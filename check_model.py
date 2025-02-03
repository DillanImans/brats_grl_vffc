import torch
import torch.nn as nn
from torchinfo import summary

from brats import get_datasets, get_datasets_independent, get_datasets_grl_based
from networks.models.SegResNet.segresnet import SegResNet, GRLSegResNet

from myCodeArchives.vffc import Bottleneck
from myCodeArchives.nnAdapts import conv_nd, conv_nd_class, avg_pool_nd, norm_nd_class, DropPath


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GRLSegResNet(
    spatial_dims = 3,
    init_filters = 32,
    in_channels = 1,
    out_channels = 3,
    dropout_prob = 0.2,
    blocks_down = (1, 2, 2, 4),
    blocks_up = (1, 1, 1),
    num_domains = 3,
    alpha = 1.0,
).to(device)

dataset = get_datasets_grl_based(
    dataset_folder = '/home/monetai/Desktop/dillan/dataBratsFormatted',
    mode = 'train',
    target_size = (128, 128, 128),
    version = 'brats2020',
    modality = 't2',
    seedAh = 42
)

chosen = 0
sample = dataset[chosen]

image_tensor = sample['image'].unsqueeze(0).to(device)
label_tensor = sample['label'].unsqueeze(0).to(device)
domain_label_tensor = torch.tensor([sample['domain_label']], device = device)

print(image_tensor.shape)
print(label_tensor.shape)
print(domain_label_tensor.shape)

model = GRLSegResNet(
        spatial_dims = 3,
        init_filters = 32,
        in_channels = 1,
        out_channels = 3,
        dropout_prob = 0.2,
        blocks_down = (1, 2, 2, 4),
        blocks_up = (1, 1, 1),
        num_domains = 3,
        alpha = 1.0
    ).cuda()

dummyInput = torch.randn(1, 1, 128, 128, 128).cuda()
out = model(dummyInput)
print('Output shape:', out.shape)