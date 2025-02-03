from torch import nn
import torch
import time
import torch.fft


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambda_ = 1.0):
        super(GRL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    def __init__(self, in_channels, num_domains):
        super(DomainClassifier, self).__init__()
        self.globalpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_domains)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.globalpool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class FrequencyFilterModule(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyFilterModule, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        # depth, height, width
        x_freq = torch.fft.fftn(x, dim = (-3, -2, -1), norm = 'ortho')

        x_freq_filtered = x_freq

        x_filtered = torch.fft.ifftn(x_freq_filtered, dim = (-3, -2, -1), norm = 'ortho').real
        
        return x_filtered


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels = in_channels, out_channels = out_channels//2, kernel_size = (3, 3, 3), padding = 1)
        self.bn1 = nn.BatchNorm3d(num_features = out_channels//2)
        self.conv2 = nn.Conv3d(in_channels = out_channels // 2, out_channels = out_channels, kernel_size = (3, 3, 3), padding = 1)
        self.bn2 = nn.BatchNorm3d(num_features = out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = 2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    def __init__(self, in_channels, res_channels = 0, last_layer = False, num_classes = None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (last_layer == True and num_classes != None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels = in_channels, out_channels = in_channels, kernel_size = (2, 2, 2), stride = 2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features = in_channels//2)
        self.conv1 = nn.Conv3d(in_channels = in_channels + res_channels, out_channels = in_channels // 2, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels = in_channels // 2, out_channels = in_channels // 2, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.last_layer = last_layer

        if last_layer:
            self.conv3 = nn.Conv3d(in_channels = in_channels//2, out_channels = num_classes, kernel_size = (1, 1, 1))

    def forward(self, input, residual = None):
        out = self.upconv1(input)
        if residual != None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        
        return out


class UNet3DGRL(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 5, num_domains = 3, level_channels = [64, 128, 256], bottleneck_channel = 512, lambda_ = 1.0) -> None:
        super(UNet3DGRL, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels


        # Encoder
        self.a_block1 = Conv3DBlock(in_channels = in_channels, out_channels = level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels = level_1_chnls, out_channels = level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels = level_2_chnls, out_channels = level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels = level_3_chnls, out_channels = bottleneck_channel, bottleneck = True)


        # Maybe put on bottleneck?
        # self.frequency_filter = FrequencyFilterModule(in_channels = bottleneck_channel)


        # Decoder
        self.s_block3 = UpConv3DBlock(in_channels = bottleneck_channel, res_channels = level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels = level_3_chnls, res_channels = level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels = level_2_chnls, res_channels = level_1_chnls, num_classes = num_classes, last_layer = True)


        # DANN
        self.grl = GRL(lambda_ = lambda_)
        self.domain_classifier = DomainClassifier(in_channels = bottleneck_channel, num_domains = num_domains)


    def forward(self, input):

        # Encoder
        out, residual_level1 = self.a_block1(input)
        # out = self.frequency_filter(out)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        bottleneck_features, _ = self.bottleNeck(out)


        # Decoder
        out = self.s_block3(bottleneck_features, residual_level3)
        out = self.s_block2(out, residual_level2)
        segmentation_output = self.s_block1(out, residual_level1)

    
        # DAN branch
        reversed_features = self.grl(bottleneck_features)
        domain_output = self.domain_classifier(reversed_features)

        domain_output = None

        return segmentation_output, domain_output