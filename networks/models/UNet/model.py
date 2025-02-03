'''
Implementation of 3D-UNet Architecture for biomedical image segmentation for BraTS datasets
paper: https://arxiv.org/abs/1606.06650

3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
code borrowed from: https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
from torchsummary import summary

from myCodeArchives.nnAdapts import norm_nd_class
from myCodeArchives.vffc import Bottleneck
from monai.losses import DiceLoss


class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        """convolution->batch norm-> relu"""
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res

class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out
        
class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    
    def forward(self, input):
        #Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        #Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out
    

# testing 
if __name__ == '__main__':
    #Configurations according to the Xenopus kidney dataset
    model = UNet3D(in_channels=4, num_classes=3)
    input = torch.rand(1, 4, 128, 128, 128)
    out = model(input)
    print(out.shape)


    # start_time = time.time()
    # summary(model=model, input_size=(1, 4, 128, 128, 128), batch_size=1, device="cpu")
    # print("--- %s seconds ---" % (time.time() - start_time))



################################################################

class GradReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    
def grad_reverse(x, alpha = 1.0):
    return GradReverseFunction.apply(x, alpha)

class GRLUNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, num_domains = 3, alpha = 1.0,
                 level_channels = [64, 128, 256], bottleneck_channel = 512):
        super().__init__()
        self.alpha = alpha
        self.num_domains = num_domains

        level_1_chnls, level_2_chnls, level_3_chnls = level_channels
        self.a_block1 = Conv3DBlock(in_channels, level_1_chnls)
        self.a_block2 = Conv3DBlock(level_1_chnls, level_2_chnls)
        self.a_block3 = Conv3DBlock(level_2_chnls, level_3_chnls)
        
        # New VFFC Bottleneck
        norm_layer = norm_nd_class(3)

        self.bottleneck_proj = nn.Conv3d(
            in_channels = level_3_chnls,
            out_channels = bottleneck_channel,
            kernel_size = 1
        )

        self.bottleneck = Bottleneck(
            feats_num_bottleneck=bottleneck_channel,
            norm_layer=norm_layer,
            padding_type='reflect',
            resnet_conv_kwargs={
                "ratio_gin": 0.5,
                "ratio_gout": 0.5
            },
            inline=True,
            drop_path_rate=0
        )

        self.s_block3 = UpConv3DBlock(bottleneck_channel, level_3_chnls)
        self.s_block2 = UpConv3DBlock(level_3_chnls, level_2_chnls)
        self.s_block1 = UpConv3DBlock(level_2_chnls, level_1_chnls, 
                                     num_classes=num_classes, last_layer=True)
        
        self.domain_head = nn.Sequential(
            nn.Linear(bottleneck_channel, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

        self.seg_loss_fn = DiceLoss(sigmoid = True, to_onehot_y = False)
        self.domain_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, seg_labels = None, domain_labels = None, alpha = None):

        # Encoder
        x, res1 = self.a_block1(x)
        x, res2 = self.a_block2(x)
        x, res3 = self.a_block3(x)

        x = self.bottleneck_proj(x)
        x = self.bottleneck(x)
        features = x

        x = self.s_block3(x, res3)
        x = self.s_block2(x, res2)
        seg_logits = self.s_block1(x, res1)

        if seg_labels is None and domain_labels is None:
            return seg_logits
        
        pooled = F.adaptive_avg_pool3d(features, (1, 1, 1)).flatten(1)
        reversed_feat = grad_reverse(pooled, alpha = self.alpha if alpha is None else alpha)
        domain_logits = self.domain_head(reversed_feat)

        seg_loss = self.seg_loss_fn(seg_logits, seg_labels) if seg_labels is not None else 0
        domain_loss = self.domain_loss_fn(domain_logits, domain_labels) if domain_labels is not None else 0

        return seg_loss, domain_loss
    
