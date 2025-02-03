"""
==========================
Data loading and processing
===========================

credit: https://github.com/faizan1234567/CKD-TransBTS/blob/main/BraTS.py
"""
import torch
import os
from torch.utils.data.dataset import Dataset, ConcatDataset
from utils.all_utils import pad_or_crop_image, minmax, load_nii, pad_image_and_label, listdir, get_brats_folder
from math import comb
from copy import deepcopy
import numpy as np
import random

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
)

######################################

class BraTSNULLIFIED(Dataset):
    def __init__(self, patients_dir, patient_ids, mode, target_size = (128, 128, 128), version="brats2023"):
        super(BraTS,self).__init__()
        self.patients_dir = patients_dir
        self.patients_ids = patient_ids
        self.mode = mode
        self.target_size = target_size
        self.version = version
        self.datas = []
        if version == "brats2023":
            self.pattens =["-t1n","-t1c","-t2w","-t2f"]
        elif version == "brats2019" or version == "brats2020":
            self.pattens =["_t1","_t1ce","_t2","_flair"]
        if (mode == "train" or mode == "train_val" or mode == "test") and version == "brats2023" :
            self.pattens += ["-seg"]
        elif (mode == "train" or mode == "train_val" or mode == "test") and (version == "brats2019" or version == "brats2020"):
            self.pattens += ["_seg"]

        for patient_id in patient_ids:
            if version == "brats2023":
                paths = [f"{patient_id}{patten}.nii.gz" for patten in self.pattens]
            elif version == "brats2019" or version == "brats2020":
                paths = [f"{patient_id}{patten}.nii" for patten in self.pattens]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if mode == "train" or mode == "train_val" or mode == "val" or mode == "test" or mode == "visualize" else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        patient = self.datas[idx]
        patient_id = patient["id"]
        crop_list = []
        pad_list = []
        patient_image = {key:torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient[key]}")) for key in patient if key not in ["id", "seg"]}
        patient_label = torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient['seg']}").astype("int8"))
        patient_image = torch.stack([patient_image[key] for key in patient_image])  
        if self.mode == "train" or self.mode == "train_val" or self.mode == "test":
            ed_label = 2 # Peritumoral Edema 
            ncr_label = 1 # NCR or NET (necrotic and non-enhancing tumor core )
            bg_label = 0  # Background
            if self.version == "brats2023" or self.version == "brats2024":
                et_label = 3 #  GD-enhancing tumor
                et = patient_label == et_label
            elif self.version == "brats2020" or self.version == "brats2019":
                et_label = 4 #  GD-enhancing tumor
                et = patient_label == et_label
            tc = torch.logical_or(patient_label == ncr_label, patient_label == et_label)
            wt = torch.logical_or(tc, patient_label == ed_label)
            patient_label = torch.stack([et, tc, wt])

        # Removing black area from the edge of the MRI
        nonzero_index = torch.nonzero(torch.sum(patient_image, axis=0)!=0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:,0], nonzero_index[:,1], nonzero_index[:,2]
        zmin, ymin, xmin = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()

        for i in range(patient_image.shape[0]):
            patient_image[i] = minmax(patient_image[i])
        
        patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        if self.mode == "train" or self.mode == "train_val" or self.mode == "test":
            patient_image, patient_label, pad_list, crop_list = pad_or_crop_image(patient_image, patient_label, target_size=self.target_size)
        elif self.mode == "test_pad":
            d, h, w = patient_image.shape[1:]
            pad_d = (128-d) if 128-d > 0 else 0
            pad_h = (128-h) if 128-h > 0 else 0
            pad_w = (128-w) if 128-w > 0 else 0
            patient_image, patient_label, pad_list = pad_image_and_label(patient_image, patient_label, target_size=(d+pad_d, h+pad_h, w+pad_w))

        return dict(
            patient_id = patient["id"],
            image = patient_image.to(dtype=torch.float32),
            label = patient_label.to(dtype=torch.float32),
            nonzero_indexes = ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            box_slice = crop_list,
            pad_list = pad_list
        )

    def __len__(self):
        return len(self.datas)

def get_datasets(dataset_folder, mode, target_size = (128, 128, 128), version= "brats2020"):
    dataset_folder = get_brats_folder(dataset_folder, mode, version= version)
    assert os.path.exists(dataset_folder), "Dataset Folder Does Not Exist1"
    patients_ids = [x for x in listdir(dataset_folder)]
    return BraTS(dataset_folder, patients_ids, mode, target_size=target_size, version="brats2020")


######################################


# This is for T1CE and T2. Anyways the changes are quite obvious.
class BraTSNullified2(Dataset):
    def __init__(self, patients_dir, patient_ids, mode, target_size = (128, 128, 128), version="brats2020"):
        super(BraTS,self).__init__()
        self.patients_dir = patients_dir
        self.patients_ids = patient_ids
        self.mode = mode
        self.target_size = target_size
        self.version = version
        self.datas = []
        if version == "brats2023":
            self.pattens =["-t1n","-t1c","-t2w","-t2f"]
        elif version == "brats2019" or version == "brats2020":
            # self.pattens =["_t1","_t1ce","_t2","_flair"]
            self.pattens = ['_t1ce', '_t2']
        if (mode == "train" or mode == "train_val" or mode == "test") and version == "brats2023" :
            self.pattens += ["-seg"]
        elif (mode == "train" or mode == "train_val" or mode == "test") and (version == "brats2019" or version == "brats2020"):
            self.pattens += ["_seg"]

        for patient_id in patient_ids:
            if version == "brats2023":
                paths = [f"{patient_id}{patten}.nii.gz" for patten in self.pattens]
            elif version == "brats2019" or version == "brats2020":
                paths = [f"{patient_id}{patten}.nii" for patten in self.pattens]
            patient = dict(
                # id=patient_id, t1=paths[0], t1ce=paths[1],
                # t2=paths[2], flair=paths[3], seg=paths[4] if mode == "train" or mode == "train_val" or mode == "val" or mode == "test" or mode == "visualize" else None
                id = patient_id, t1ce = paths[0], t2 = paths[1], seg = paths[2] if mode == 'train' or mode == "train_val" or mode == "val" or mode == "test" or mode == "visualize" else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        patient = self.datas[idx]
        patient_id = patient["id"]
        crop_list = []
        pad_list = []
        patient_image = {key:torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient[key]}")) for key in patient if key not in ["id", "seg"]}
        patient_label = torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient['seg']}").astype("int8"))
        patient_image = torch.stack([patient_image[key] for key in patient_image])  
        if self.mode == "train" or self.mode == "train_val" or self.mode == "test":
            ed_label = 2 # Peritumoral Edema 
            ncr_label = 1 # NCR or NET (necrotic and non-enhancing tumor core )
            bg_label = 0  # Background
            if self.version == "brats2023" or self.version == "brats2024":
                et_label = 3 #  GD-enhancing tumor
                et = patient_label == et_label
            elif self.version == "brats2020" or self.version == "brats2019":
                et_label = 4 #  GD-enhancing tumor
                et = patient_label == et_label
            tc = torch.logical_or(patient_label == ncr_label, patient_label == et_label)
            wt = torch.logical_or(tc, patient_label == ed_label)
            patient_label = torch.stack([et, tc, wt])

        # Removing black area from the edge of the MRI
        nonzero_index = torch.nonzero(torch.sum(patient_image, axis=0)!=0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:,0], nonzero_index[:,1], nonzero_index[:,2]
        zmin, ymin, xmin = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()

        for i in range(patient_image.shape[0]):
            patient_image[i] = minmax(patient_image[i])
        
        patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        if self.mode == "train" or self.mode == "train_val" or self.mode == "test":
            patient_image, patient_label, pad_list, crop_list = pad_or_crop_image(patient_image, patient_label, target_size=self.target_size)
        elif self.mode == "test_pad":
            d, h, w = patient_image.shape[1:]
            pad_d = (128-d) if 128-d > 0 else 0
            pad_h = (128-h) if 128-h > 0 else 0
            pad_w = (128-w) if 128-w > 0 else 0
            patient_image, patient_label, pad_list = pad_image_and_label(patient_image, patient_label, target_size=(d+pad_d, h+pad_h, w+pad_w))

        return dict(
            patient_id = patient["id"],
            image = patient_image.to(dtype=torch.float32),
            label = patient_label.to(dtype=torch.float32),
            nonzero_indexes = ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            box_slice = crop_list,
            pad_list = pad_list
        )

    def __len__(self):
        return len(self.datas)

def get_datasets(dataset_folder, mode, target_size = (128, 128, 128), version= "brats2020"):
    dataset_folder = get_brats_folder(dataset_folder, mode, version= version)
    assert os.path.exists(dataset_folder), "Dataset Folder Does Not Exist1"
    patients_ids = [x for x in listdir(dataset_folder)]
    return BraTS(dataset_folder, patients_ids, mode, target_size=target_size, version="brats2020")


######################################

# Brats Independent Domain
class BraTSIndependent(Dataset):
    def __init__(
            self,
            patients_dir,
            patient_ids,
            mode,
            target_size = (128, 128, 128),
            version = 'brats2020',
            modality = 't1',
            seed = None
    ):
        super(BraTSIndependent, self).__init__()

        self.patients_dir = patients_dir
        self.patient_ids = patient_ids
        self.mode = mode
        self.target_size = target_size
        self.version = version
        self.modality = modality.lower()
        self.seed = seed

        self.rng = random.Random(seed) if seed is not None else random

        if self.version in ['brats2020']:
            pattern_map = {
                "t1":    "_t1",
                "t1ce":  "_t1ce",
                "t2":    "_t2",
                "flair": "_flair",
            }
            seg_pattern = "_seg"
        else:
            raise ValueError(
                f"Version {self.version} is not supported"
            )

        self.modality_pattern = pattern_map[self.modality]
        self.seg_pattern = seg_pattern

        self.datas = []

        for patient_id in self.patient_ids:
            image_path = f"{patient_id}{self.modality_pattern}.nii"
            label_path = f"{patient_id}{self.seg_pattern}.nii"

            if self.mode in ['train', 'train_val', 'val', 'test', 'visualize']:
                seg_path = label_path
            else:
                seg_path = None

            self.datas.append(
                dict(
                    id=patient_id,
                    image=image_path,
                    seg=seg_path
                )
            )

    def __getitem__(self, idx):
        patient = self.datas[idx]
        patient_id = patient["id"]

        if self.seed is not None:
            self.rng.seed(self.seed + idx)

        patient_image = torch.tensor(
            load_nii(f"{self.patients_dir}/{patient_id}/{patient['image']}")
        ).float()

        if patient['seg'] is not None:
            patient_label = torch.tensor(
                load_nii(f"{self.patients_dir}/{patient_id}/{patient['seg']}").astype('int8'))
        else:
            patient_label = None

        patient_image = patient_image.unsqueeze(0)

        if patient_label is not None and self.mode in ["train", "train_val", "test", "val", "visualize"]:
            ed_label = 2
            ncr_label = 1
            bg_label = 0

            et_label = 4

            et = (patient_label == et_label)
            tc = torch.logical_or(patient_label == ncr_label, et)
            wt = torch.logical_or(tc, patient_label == ed_label)

            patient_label = torch.stack([et, tc, wt], dim=0).float()

        nonzero_index = torch.nonzero(torch.sum(patient_image, axis = 0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:, 0], nonzero_index[:, 1], nonzero_index[:, 2]
        zmin, ymin, xmin = [
            max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)
        ]
        zmax, ymax, xmax = [
            int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)
        ]

        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]

        patient_image[0] = minmax(patient_image[0])

        if patient_label is not None:
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        crop_list = []
        pad_list = []

        if self.mode in ["train", "train_val", "test"]:
            patient_image, patient_label, pad_list, crop_list = pad_or_crop_image(
                patient_image, 
                patient_label,
                target_size=self.target_size
            )
        elif self.mode == "test_pad":
            d, h, w = patient_image.shape[1:]
            pad_d = (128 - d) if (128 - d) > 0 else 0
            pad_h = (128 - h) if (128 - h) > 0 else 0
            pad_w = (128 - w) if (128 - w) > 0 else 0
            patient_image, patient_label, pad_list = pad_image_and_label(
                patient_image,
                patient_label,
                target_size=(d + pad_d, h + pad_h, w + pad_w)
            )

        if patient_label is None:
            patient_label = torch.zeros(3, *patient_image.shape[1:], dtype = torch.float32)


        return dict(
            patient_id=patient["id"],
            image=patient_image.to(dtype=torch.float32),
            label=patient_label.to(dtype=torch.float32),
            nonzero_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            box_slice=crop_list,
            pad_list=pad_list
        )
    
    def __len__(self):
        return len(self.datas)
    

def get_datasets_independent(
    dataset_folder,
    mode,
    target_size = (128, 128, 128),
    version='brats2020',
    modality = 't1',
    seedAh = 42
):
    dataset_folder = get_brats_folder(dataset_folder, mode, version = version)
    assert os.path.exists(dataset_folder), "Dataset folder does not exist."

    patients_ids = [x for x in listdir(dataset_folder)]

    return BraTSIndependent(
        patients_dir = dataset_folder,
        patient_ids = patients_ids,
        mode = mode,
        target_size = target_size,
        version = version,
        modality = modality,
        seed = seedAh
    )


######################################

class BraTSDuplicateChannel(BraTSIndependent):
    def __init__(
        self,
        patients_dir,
        patient_ids,
        mode,
        target_size=(128, 128, 128),
        version="brats2020",
        modality="t1",   
    ):
        super().__init__(
            patients_dir=patients_dir,
            patient_ids=patient_ids,
            mode=mode,
            target_size=target_size,
            version=version,
            modality=modality
        )

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        image = sample['image']

        if image.shape[0] == 1:
            image_2ch = image.repeat(2, 1, 1, 1)
            sample['image'] = image_2ch

        return sample


def get_datasets_independent_2channel(
    dataset_folder,
    mode,
    target_size = (128, 128, 128),
    version = 'brats2020',
    modality = 't1'
):
    dataset_folder = get_brats_folder(dataset_folder, mode, version = version)
    assert os.path.exists(dataset_folder), f"Dataset folder {dataset_folder} does not exist."

    patient_ids = [x for x in listdir(dataset_folder)]
    return BraTSDuplicateChannel(
        patients_dir=dataset_folder,
        patient_ids=patient_ids,
        mode=mode,
        target_size=target_size,
        version=version,
        modality=modality
    )


######################################

def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** (n - i)) * ((1 - t) ** i)

def bezier_curve(points, nTimes = 100000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)
    poly_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(nPoints)])
    xvals = np.dot(xPoints, poly_array)
    yvals = np.dot(yPoints, poly_array)
    return xvals, yvals

def apply_domain_curve(img_np, domain_label):
    if domain_label not in [1, 2]:
        return img_np
    
    c = 1
    if img_np.ndim == 4:
        c = img_np.shape[0]

    v = random.random()
    w = random.random()

    if domain_label == 1:
        #   [-1, -1], [-v, v], [v, -v], [1, 1]
        points = [
            [-1, -1],
            [-v,  v],
            [ v, -v],
            [ 1,  1]
        ]
    elif domain_label == 2:  # domain_label == 2
        # Flip sign for each point => [1, 1], [v, -v], [-v, v], [-1, -1]
        points = [
            [ 1,  1],
            [ w, -w],
            [-w,  w],
            [-1, -1]
        ]

    xvals, yvals = bezier_curve(points, nTimes = 100000)

    idx_sort = np.argsort(xvals)
    xvals = xvals[idx_sort]
    yvals = yvals[idx_sort]

    for ch in range(c):
        if c == 1:
            channel_data = img_np
        else:
            channel_data = img_np[ch]

        cmin, cmax = channel_data.min(), channel_data.max()

        if cmin != cmax:
            channel_norm = 2.0 * (channel_data - cmin) / (cmax - cmin) - 1.0
        else:
            if c > 1:
                img_np[ch] = channel_data
            else:
                img_np = channel_data
            continue

        channel_nonlinear = np.interp(channel_norm, xvals, yvals)

        if c == 1:
            img_np = channel_nonlinear
        else:
            img_np[ch] = channel_nonlinear

    return img_np.astype(np.float32)



monai_3d_augment = Compose([
    RandFlipd(
        keys = ['image', 'label'],
        prob = 0.5,
        spatial_axis = 0
    ),
    RandFlipd(
        keys = ['image', 'label'],
        prob = 0.5,
        spatial_axis = 1
    ),
    RandFlipd(
        keys = ['image', 'label'],
        prob = 0.5,
        spatial_axis = 2
    ),
    RandRotate90d(
        keys = ['image', 'label'],
        prob = 0.5,
        spatial_axes = (1, 2)
    ),
])


class BraTSBezierAugDataset(BraTSIndependent):
    def __init__(
        self,
        patients_dir,
        patient_ids,
        mode,
        target_size = (128, 128, 128),
        version = 'brats2020',
        modality = 't2',
        domain_label = 0,
        augment = False,
        seed = None
    ):
        super().__init__(
            patients_dir = patients_dir,
            patient_ids = patient_ids,
            mode = mode,
            target_size = target_size,
            version = version,
            modality = modality,
            seed = seed
        )

        self.domain_label = domain_label
        self.augment = augment
        self.seed = seed

    def __getitem__(self, idx):

        if self.seed is not None:
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
            np.random.seed(self.seed + idx)

        sample = super().__getitem__(idx)

        image = sample['image']
        label = sample['label']

        image = image.to(dtype = torch.float32)
        label = label.to(dtype = torch.float32)

        if self.augment:
            image_np = image.cpu().numpy()

            image_np = apply_domain_curve(image_np, domain_label = self.domain_label)

            image = torch.from_numpy(image_np).to(dtype = torch.float32, device = image.device)

        sample['image'] = image
        sample['domain_label'] = torch.tensor(self.domain_label, dtype = torch.long)

        return sample
    
    
class BraTSBezierAug3DDataset(BraTSIndependent):
    def __init__(
        self,
        patients_dir,
        patient_ids,
        mode,
        target_size=(128, 128, 128),
        version='brats2020',
        modality='t2',
        domain_label=0,
        augment=True,
        seed=None,
    ):
        super().__init__(
            patients_dir=patients_dir,
            patient_ids=patient_ids,
            mode=mode,
            target_size=target_size,
            version=version,
            modality=modality,
            seed=seed
        )
        self.domain_label = domain_label
        self.augment = augment
        self.seed = seed

        self.monai_3d_augment = monai_3d_augment

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        image = sample['image']
        label = sample['label']

        if self.seed is not None:
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
            np.random.seed(self.seed + idx)

        if self.augment:
            data_dict = {"image": image, "label": label}
            
            # If we want the flips and rotates
            # data_dict = self.monai_3d_augment(data_dict)
            
            image = data_dict["image"]
            label = data_dict["label"]

            if self.domain_label in [1, 2]:
                img_np = image.cpu().numpy()
                img_np = apply_domain_curve(img_np, self.domain_label)
                image = torch.from_numpy(img_np).to(image.device)

        sample["image"] = image
        sample["label"] = label
        sample["domain_label"] = torch.tensor(self.domain_label, dtype=torch.long)
        return sample



def get_datasets_grl_based(
    dataset_folder,
    mode,
    target_size = (128, 128, 128),
    version = 'brats2020',
    modality = 't1',
    seedAh = 42
):
    dataset_folder = get_brats_folder(dataset_folder, mode, version = version)
    assert os.path.exists(dataset_folder), f"Dataset folder {dataset_folder} does not exist."

    patient_ids = [x for x in listdir(dataset_folder)]

    n = len(patient_ids)
    half_n = n // 2
    quarter_n = n // 4

    # subset_ids_0 = patient_ids[:half_n]
    # subset_ids_1 = patient_ids[half_n:half_n + quarter_n]
    # subset_ids_2 = patient_ids[half_n+quarter_n:]

    subset_ids_1 = patient_ids[:half_n]
    subset_ids_2 = patient_ids[half_n:]


    # ds0 = BraTSBezierAug3DDataset(
    #     patients_dir=dataset_folder,
    #     patient_ids=subset_ids_0,
    #     mode=mode,
    #     target_size=target_size,
    #     version=version,
    #     modality=modality,
    #     domain_label=0,
    #     augment=True,
    #     seed = seedAh
    # )

    ds1 = BraTSBezierAug3DDataset(
        patients_dir=dataset_folder,
        patient_ids=subset_ids_1,
        mode=mode,
        target_size=target_size,
        version=version,
        modality=modality,
        domain_label=1,
        augment=True,
        seed = seedAh
    )

    ds2 = BraTSBezierAug3DDataset(
        patients_dir=dataset_folder,
        patient_ids=subset_ids_2,
        mode=mode,
        target_size=target_size,
        version=version,
        modality=modality,
        domain_label=2,
        augment=True,
        seed = seedAh
    )

    # combined_dataset = ConcatDataset([ds0, ds1, ds2])
    combined_dataset = ConcatDataset([ds1, ds2])

    return combined_dataset
