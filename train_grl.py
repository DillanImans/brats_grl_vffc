import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import gc
import nibabel as nib
import tqdm as tqdm
from utils.meter import AverageMeter
from utils.general import save_checkpoint, load_pretrained_model, resume_training
from brats import get_datasets, get_datasets_independent, get_datasets_grl_based
from tqdm import tqdm

from monai.data import  decollate_batch
import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast


from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.networks.nets import SwinUNETR, VNet, AttentionUnet, UNETR
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from networks.models.UNet.model import UNet3D, GRLUNet3D
from networks.models.UX_Net.network_backbone import UXNET
from networks.models.nnformer.nnFormer_tumor import nnFormer
from networks.models.SegResNet.segresnet import SegResNet, GRLSegResNet

try:
    from thesis.models.SegUXNet.model import SegUXNet
except ModuleNotFoundError:
    print('model not available, please train with other models')
    
from functools import partial
from utils.augment import DataAugmenter
from utils.schedulers import SegResNetScheduler, PolyDecayScheduler

# Configure logger
import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs("logger", exist_ok=True)
file_handler = logging.FileHandler(filename="logger/train_logger.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def save_best_model(dir_name, model, name="best_model"):
    save_path = os.path.join(dir_name, name)
    os.makedirs(save_path, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict, f"{save_path}/{name}.pkl")


def save_checkpoint(dir_name, state, name="checkpoint"):
    save_path = os.path.join(dir_name, name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(state, f"{save_path}/{name}.pth.tar")


def create_dirs(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "best-model"), exist_ok=True)


def init_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class Solver:
    def __init__(self, model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-5):
        self.lr = lr
        self.weight_decay = weight_decay
        self.all_solvers = {
            "Adam": torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True),
            "AdamW": torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True),
            "SGD": torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
        }

    def select_solver(self, name):
        return self.all_solvers[name]

def dann_alpha_schedule(current_epoch: int, max_epochs: int) -> float:
    """
    Returns alpha in [0..1] increasing during training (DANN style)
    """
    p = float(current_epoch) / float(max_epochs)
    alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1
    return abs(alpha)


def train_epoch_domain_adaptation(model, loader, optimizer, device, epoch: int,
                                  max_epochs: int, schedule_alpha: bool = True,
                                    lambda_domain: float = 1.0, augment: bool = False):
    """
    Train the model for one epoch.
    (Print statements are minimized to reduce overhead.)
    """
    model.train()
    epoch_loss_meter = AverageMeter()
    count = 0

    scaler = GradScaler('cuda')

    if schedule_alpha:
        alpha = dann_alpha_schedule(epoch, max_epochs)
    else:
        alpha = 1

    with tqdm(loader, desc=f"DomainAdapt-Train Epoch {epoch + 1}", leave=False) as progress_bar:
        for batch_data in progress_bar:
            images = batch_data['image'].to(device)
            seg_labels = batch_data['label'].to(device)
            domain_labels = batch_data['domain_label'].to(device)

            if augment:
                augmenter = DataAugmenter().to(device)
                image, seg_labels = augmenter(image, seg_labels)

            optimizer.zero_grad()

            with autocast("cuda"):
                seg_loss, domain_loss = model(
                    images,
                    seg_labels = seg_labels,
                    domain_labels = domain_labels,
                    alpha = alpha,
                )
                total_loss = seg_loss + lambda_domain * domain_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss_meter.update(total_loss.item(), images.size(0))
            count += images.size(0)
            progress_bar.set_postfix({
                "seg_loss": f"{seg_loss.item():.4f}",
                "dom_loss": f"{domain_loss.item():.4f}",
                "alpha": f"{alpha:.3f}"
            })
       
        return epoch_loss_meter.avg


def val(model, loader, acc_func, device, model_inferer=None, post_sigmoid=None, post_pred=None):
    """
    Validation phase on loader. Return average dice metric for BraTS (ET, WT, TC).
    """
    model.eval()
    run_acc = AverageMeter()

    with torch.no_grad():
        for batch_data in loader:
            
            logits = model_inferer(batch_data["image"].to(device))
            masks = decollate_batch(batch_data["label"].to(device))

            prediction_lists = decollate_batch(logits)
            predictions = [post_pred(post_sigmoid(pred)) for pred in prediction_lists]

            acc_func.reset()
            acc_func(y_pred=predictions, y=masks)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

    return run_acc.avg


def save_data(training_loss, et, wt, tc, val_mean_acc, epochs, cfg):
    data = {}
    NAMES = ["training_loss", "WT", "ET", "TC", "mean_dice", "epochs"]
    data_lists = [training_loss, wt, et, tc, val_mean_acc, epochs]
    for i in range(len(NAMES)):
        data[f"{NAMES[i]}"] = data_lists[i]
    data_df = pd.DataFrame(data)
    save_path = os.path.join(cfg.training.exp_name, "csv")
    os.makedirs(save_path, exist_ok=True)
    data_df.to_csv(os.path.join(save_path, "training_data.csv"))
    return data


def trainer(
    cfg,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    device,
    model_inferer=None,
    max_epochs=100,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    val_every=10,
    schedule_alpha = False
):
    """
    Train (max_epochs) and validate every 'val_every' epochs.
    """
    val_acc_max = 0
    dices_tc, dices_wt, dices_et, mean_dices = [], [], [], []
    epoch_losses, train_epochs = [], []

    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()

        train_loss = train_epoch_domain_adaptation(
            model = model,
            loader = train_loader,
            optimizer = optimizer,
            device = device,
            epoch = epoch,
            max_epochs = max_epochs,
            schedule_alpha = cfg.training.schedule_alpha,
            lambda_domain = cfg.training.lambda_domain,
            augment = False
        )

        epoch_time = (time.time() - epoch_start_time) / 60.0

        if (epoch % val_every == 0) or (epoch == 0):
            epoch_losses.append(train_loss)
            train_epochs.append(epoch)

            val_acc = val(
                model=model,
                loader=val_loader,
                acc_func=acc_func,
                device=device,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_et = val_acc[0]
            dice_wt = val_acc[1]
            dice_tc = val_acc[2]
            mean_dice = np.mean(val_acc)

            dices_et.append(dice_et)
            dices_wt.append(dice_wt)
            dices_tc.append(dice_tc)
            mean_dices.append(mean_dice)

            logger.info(
                f"Epoch {epoch+1}/{max_epochs}, train_loss={train_loss:.4f}, "
                f"time={epoch_time:.2f} min, Val => ET={dice_et:.4f}, WT={dice_wt:.4f}, TC={dice_tc:.4f}, mean={mean_dice:.4f}"
            )

            if mean_dice > val_acc_max:
                val_acc_max = mean_dice
                save_best_model(cfg.training.exp_name, model, "best-model")

            scheduler.step()

            save_checkpoint(
                cfg.training.exp_name,
                {
                    "epoch": epoch + 1,
                    "max_epochs": max_epochs,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                "checkpoint",
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{max_epochs}, train_loss={train_loss:.4f}, time={epoch_time:.2f} min"
            )

    logger.info(f"Training Finished! Best Validation Mean Dice: {val_acc_max:.4f}")
    save_data(
        training_loss=train_loss,
        et=dices_et,
        wt=dices_wt,
        tc=dices_tc,
        val_mean_acc=mean_dices,
        epochs=train_epochs,
        cfg=cfg
    )

    return val_acc_max


def run(
    cfg,
    model,
    device,
    loss_func,
    acc_func,
    optimizer,
    train_loader,
    val_loader,
    scheduler,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
    max_epochs=100,
    val_every=10
):
    create_dirs(cfg.training.exp_name)

    start_epoch = 0
    if cfg.training.resume:
        logger.info("Resuming training...")
        checkpoint_path = os.path.join(cfg.training.exp_name, "checkpoint", "checkpoint.pth.tar")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        max_epochs = cfg.training.new_max_epochs
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info(f"Resuming from epoch={start_epoch}/{max_epochs}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")

    best_val_mean = trainer(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        scheduler,
        device=device,
        model_inferer=model_inferer,
        max_epochs=max_epochs,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        val_every=10
    )
    return best_val_mean


@hydra.main(config_name="configs", config_path="conf", version_base=None)
def main(cfg: DictConfig):
    """Main function for distributed training."""

    init_random(seed=cfg.training.seed)

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True

    if cfg.model.architecture == "unet3d":
        model = UNet3D(in_channels = 1, num_classes = 3).to(device)
    elif cfg.model.architecture == "segres_net":
        model = SegResNet(spatial_dims=3, 
                    init_filters=32, 
                    in_channels=1, 
                    out_channels=3, 
                    dropout_prob=0.2, 
                    blocks_down=(1, 2, 2, 4), 
                    blocks_up=(1, 1, 1)).to(device)
    elif cfg.model.architecture == "grlsegres_net":
        model = GRLSegResNet(
            spatial_dims = 3,
            init_filters = 32,
            in_channels = 1,
            out_channels = 3,
            dropout_prob = 0.2,
            blocks_down = (1, 2, 2, 4),
            blocks_up = (1, 1, 1),
            num_domains = 2,
            alpha = 1.0
        ).to(device)
    elif cfg.model.architecture == 'grl_unet3d':
        model = GRLUNet3D(in_channels = 1, num_classes = 3, num_domains = 3,
                           level_channels=[64, 128, 256], bottleneck_channel=512).to(device)

    else:
        raise NotImplementedError("Please implement your chosen architecture init here...")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    dataset_dir = cfg.dataset.dataset_folder


    # This is for stacked modalities
    # train_dataset = get_datasets(dataset_dir, "train", target_size=(128, 128, 128))
    # val_dataset = get_datasets(dataset_dir, "train_val", target_size=(128, 128, 128))


    # This is for single modalities
    # train_dataset = get_datasets_independent(
    #     dataset_folder = dataset_dir,
    #     mode = 'train',
    #     target_size = (128, 128, 128),
    #     version = 'brats2020',
    #     modality = 't2',
    #     seedAh = 42
    # )

    # val_dataset = get_datasets_independent(
    #     dataset_folder=dataset_dir,
    #     mode="train_val",
    #     target_size=(128, 128, 128),
    #     version="brats2020",
    #     modality="t2",
    #     seedAh = 42
    # )

    train_dataset = get_datasets_grl_based(
        dataset_folder = dataset_dir,
        mode = 'train',
        target_size = (128, 128, 128),
        version = 'brats2020',
        modality = 't2',
        seedAh = 42
    )

    
    val_dataset = get_datasets_grl_based(
        dataset_folder = dataset_dir,
        mode = 'train_val',
        target_size = (128, 128, 128),
        version = 'brats2020',
        modality = 't2',
        seedAh = 42
    )


    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) \
        if world_size > 1 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=False
    )

    loss_func = DiceLoss(to_onehot_y=False, sigmoid=True)
    acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    solver = Solver(model=model, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    optimizer = solver.select_solver(cfg.training.solver_name)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs)

    roi = cfg.model.roi
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi] * 3,
        sw_batch_size=cfg.training.sw_batch_size,
        predictor=model,
        overlap=cfg.model.infer_overlap
    )

    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    post_sigmoid = Activations(sigmoid=True)

    if rank == 0:
        logger.info(f"Starting training with world_size={world_size}, local_rank={local_rank}")
        logger.info(f"Train dataset size = {len(train_dataset)}, Val dataset size = {len(val_dataset)}")

    best_val_mean = run(
        cfg,
        model,
        device,
        loss_func,
        acc_func,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        model_inferer=model_inferer,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        max_epochs=cfg.training.max_epochs,
        val_every=cfg.training.val_every
    )

    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        logger.info(f"Finished training. Best validation mean dice = {best_val_mean:.4f}")


if __name__ == "__main__":
    main()
