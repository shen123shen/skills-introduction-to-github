import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from InceptionNext import inceptionnext_tiny
up_kwargs = {'mode': 'bilinear', 'align_corners': False}

from torchsummary import summary


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    return param_size


class MDI_Net(nn.Module):
    def __init__(self, out_planes=1, encoder='inceptionnext_tiny'):
        super(MDI_Net, self).__init__()
        self.encoder = encoder
        if self.encoder == 'inceptionnext_tiny':
            mutil_channel = [96, 192, 384, 768]
            self.backbone = inceptionnext_tiny()

        self.dropout = torch.nn.Dropout(0.3)  # 添加 Dropout
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.mlfa1 = MLFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)
        self.mlfa2 = MLFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)
        self.mlfa3 = MLFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)


        self.decoder4 = BasicConv2d(mutil_channel[3], mutil_channel[2], 3, padding=1)
        self.decoder3 = BasicConv2d(mutil_channel[2], mutil_channel[1], 3, padding=1)
        self.decoder2 = BasicConv2d(mutil_channel[1], mutil_channel[0], 3, padding=1)
        self.decoder1 = nn.Sequential(nn.Conv2d(mutil_channel[0], 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, out_planes, kernel_size=1, stride=1))

        self.fu1 = DGIA(96, 192,  96)
        self.fu2 = DGIA(192, 384, 192)
        self.fu3 = DGIA(384, 768,  384)
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        '''
        x1 : (2,96,56,56)
        x2 : (2,192,28,28)
        x3 : (2,384,14,14)
        x4 : (2,768,7,7)
        '''

        x1, x2, x3, x4 = self.mlfa1(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.mlfa2(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.mlfa3(x1, x2, x3, x4)

        x_f_3 = self.fu3(x3, x4)
        x_f_2 = self.fu2(x2, x_f_3)
        x_f_1 = self.fu1(x1, x_f_2)

        d1 = self.decoder1(x_f_1)
        d1 = self.dropout(d1)  # 在解码器阶段应用 Dropout
        d1 = F.interpolate(d1, scale_factor=4, mode='bilinear')  # (1,1,224,224)
        return d1


class DGIA(nn.Module):
    def __init__(self, l_dim, g_dim, out_dim):
        super(DGIA,self).__init__()
        self.extra_l = LKFE(l_dim)
        self.bn = nn.BatchNorm2d(out_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3x3 = BasicConv2d(g_dim, out_dim, 3, padding=1)
        self.selection = nn.Conv2d(out_dim, 1, 1)
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self,l,g):
        l = self.extra_l(l)
        g = self.conv3x3(self.upsample(g))
        weight = self.avg_pool(g)  # (1,384,14,14)
        output = l * weight + g
        return output


class LKFE(nn.Module):
    # 初始化，输入特征图通道数，输出特征图通道数，DW卷积的步长=1或2
    def __init__(self, dim):
        super(LKFE, self).__init__()
        self.conv0 = nn.Conv2d(2*dim//3, 2*dim//3, 5, padding=2, groups=2*dim//3)
        self.conv_spatial = nn.Conv2d(2*dim//3, 2*dim//3, 7, stride=1, padding=9, groups=2*dim//3, dilation=3)
        self.conv1 = nn.Conv2d(2*dim//3, dim // 3, 1)
        self.conv2 = nn.Conv2d(2*dim//3, dim // 3, 1)

        self.split_indexes = (dim // 3, 2*dim//3)
        self.branch1 = nn.Sequential()
        self.conv1x1 = nn.Sequential(
            # 1*1卷积通道数不变
            nn.Conv2d(in_channels=2*dim//3, out_channels=2*dim//3, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2*dim//3),  # 对输出的每个通道做BN
            nn.ReLU(inplace=True))
        self.norm =nn.BatchNorm2d(dim)
    def forward(self, x):
        x_id, x_k= torch.split(x, self.split_indexes, dim=1)
        attn1 = self.conv0(x_k)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        x_id = self.branch1(x_id)
        attn = torch.cat((x_id, attn1, attn2), dim=1)
        out = channel_shuffle(attn, 2)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # (1,768,14,14)
        x = self.conv(x) # (1,384,14,14)
        x = self.bn(x)
        return x



def channel_shuffle(x, groups): # groups:表示将输入通道分成多少组进行洗牌。
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)  # 为了将通道分组，每组包含 channels_per_group 个通道。
    # 将张量的 groups 和 channels_per_group 维度交换，这是洗牌操作的关键步骤。这样做的目的是将不同组的通道混合在一起。
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
            self,
            num_in_filters,  # 输入特征图的通道数
            num_out_filters,  # 输出特征图的通道数，即卷积层中卷积核的数量。
            kernel_size,
            stride=(1, 1),  # 表示卷积核每次移动一个像素。
            activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=8
            # 使用 "same" 作为 padding 参数的值，意味着自动计算并应用适当的填充量，以保持输出特征图的空间尺寸与输入特征图相同。
        )
        self.num_in_filters =num_in_filters
        self.num_out_filters = num_out_filters
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):  # (2,1792,56,56)  1920->1792
        x=channel_shuffle(x,gcd(self.num_in_filters,self.num_out_filters))
        x = self.conv1(x)  # (2,128,56,56)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.sqe(x)


class Conv2d_channel(torch.nn.Module):
    """
    2D pointwise Convolutional layers
    """

    def __init__(self, num_in_filters, num_out_filters):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))


# 实现了通道注意力（简称 SE）机制，SE块的实现可以帮助网络集中注意力在最有信息量的特征上，抑制不太有用的特征，从而提高网络的性能和泛化能力。
# 通常放在 卷积--归一化--激活之后
class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()
        # 自适应平均池化操作,表示 1*1 大小的输出  -
        # 使用`AdaptiveAvgPool2d`进行全局平均池化，以获取每个通道的全局空间信息。
        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # 全局平均池化层，用于将每个通道的空间信息压缩成一个单一的全局特征。

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio
        # 两个全连接层，用于实现降维和升维操作。第一个全连接层将通道数减少到
        # num_channels // reduction_ratio，第二个全连接层将其恢复到原始通道数。
        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()

        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)

    def forward(self, inp):  # inp输入张量 (2,128,56,56)

        batch_size, num_channels, H, W = inp.size()
        # 将池化后得到的特征图通过view函数。将其变为形状为(batch_size, num_channels)batch_size表示输入张量的批次大小，
        # num_channels表示卷积后的特征通道数，最后再经过激活函数
        # 通过两个全连接层（`Linear`）和`LeakyReLU`激活函数来学习通道之间的关系。
        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.fc2(out)
        # 使用`Sigmoid`函数生成权重，并通过逐元素乘法将这些权重应用于输入特征图。
        out = self.sigmoid(out)
        # 将输入特征图 inp 与这些权重进行逐元素乘法，以调整每个通道的贡献。
        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))
        # 最后，使用批量归一化（`BatchNorm2d`）和`LeakyReLU`激活函数输出最终的特征图。
        out = self.bn(out)
        out = self.act(out)

        return out


class MLFA(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    # lenn: 表示重复上述操作的次数，默认为1。
    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
                in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels
        self.in_filters3_4 = (
                in_filters3 + in_filters4
        )
        self.in_filters2_3_4 = (
                in_filters2 + in_filters3 + in_filters4
        )
        # 一个上采样层，使用双线性插值进行上采样。
        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        # 一个平均池化层，用于下采样。
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        '''
        四个 ModuleList，分别用于存储每个层级的卷积层 (cnv_blks1, cnv_blks2, cnv_blks3, cnv_blks4)、
        合并卷积层 (cnv_mrg1, cnv_mrg2, cnv_mrg3, cnv_mrg4)、
        批量归一化层 (bns1, bns2, bns3, bns4) 和
        合并操作后的批量归一化层 (bns_mrg1, bns_mrg2, bns_mrg3, bns_mrg4)。
        '''

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            # 存储每个层级的卷积层
            self.cnv_blks1.append(
                # 卷积-标准化-激活-注意力
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
            )
            # 合并卷积层
            self.cnv_mrg1.append(Conv2d_batchnorm(in_filters1, in_filters1, (1, 1)))
            # 批量归一化层
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            # 合并操作后的批量归一化
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters2_3_4, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters3_4, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)

    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape

        '''
        图（D）是第三层级的MLFC3,可知在像素上，x1=2*x2=4*x3=8*x4,调整不同层级特征图的大小后进行拼接
        再进行逐点卷积-->再与X3(MLFC3为第三层)拼接，再进行逐点卷积
        '''

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                x1,
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,  # 通道
                        )
                    )
                )
            )  # x_c1 (2,128,56,56)
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                x2,
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                x3,
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                # 合并操作后的批量归一化层
                self.bns_mrg1[i](
                    # 合并卷积层
                    torch.mul(x_c1, x1).view(batch_size, self.in_filters1, h1, w1) + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    torch.mul(x_c2, x2).view(batch_size, self.in_filters2, h2, w2) + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    torch.mul(x_c3, x3).view(batch_size, self.in_filters3, h3, w3) + x3
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)

        return x1, x2, x3, x4



