import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import spectral_norm
from models import resnet_face as rf


############################################################################
# Re-usable blocks
############################################################################


class TensorTransform(nn.Module):
    # Used to convert between default color space and VGG colorspace

    def __init__(self, res=256, mean=[.485, .456, .406], std=[.229, .224, .225]):
        super(TensorTransform, self).__init__()

        self.mean = torch.zeros([3, res, res]).cuda()
        self.mean[0, :, :] = mean[0]
        self.mean[1, :, :] = mean[1]
        self.mean[2, :, :] = mean[2]

        self.std = torch.zeros([3, res, res]).cuda()
        self.std[0, :, :] = std[0]
        self.std[1, :, :] = std[1]
        self.std[2, :, :] = std[2]

    def forward(self, x):
        norm_ready = (x * .5) + .5
        norm_ready *= 255
        result = (norm_ready - self.mean)
        return result


class SReLU(nn.Module):
    # Relu with ability to subtract the mean

    def __init__(self, slope=.01, sub=None):
        super(SReLU, self).__init__()
        self.slope = slope
        self.sub = sub

    def forward(self, x):
        x = F.leaky_relu(x, self.slope)
        if self.sub is not None:
            x.sub_(self.sub)
        return x


class PreShuffConv(nn.Module):
    # Convolution which is compatible with pixel shuffle and spectral normalization

    def __init__(self, ni, nf, kernel_size=3):
        super(PreShuffConv, self).__init__()
        conv_list = [nn.Conv2d(ni, nf // 4,
                               kernel_size,
                               padding=kernel_size // 2,
                               bias=False) for i in range(4)]

        self.conv_list = nn.ModuleList(
            [spectral_norm(conv) for conv in conv_list])

    def forward(self, x):
        conv_list = [conv(x) for conv in self.conv_list]
        bs, filts, height, width = conv_list[0].shape[0], \
                                   conv_list[0].shape[1] * 4, \
                                   conv_list[0].shape[2], \
                                   conv_list[0].shape[3]

        # Interlace our conv results so that after pixel shuffle, each conv is shuffled into its own pixel
        return torch.stack(conv_list, dim=2).view(bs, filts, height, width)

    def init_icnr(self):
        print('Applying Preshuff ICNR')
        for i in range(len(self.conv_list)):
            self.conv_list[i].weight.data.copy_(self.conv_list[0].weight.data)


class ConvBlock(nn.Module):
    # Conv and Relu with option for PreShuffConv

    def __init__(self, ni, nf, kernel_size=3, quad_conv=True, stride=2, bias=True):
        super(ConvBlock, self).__init__()
        self.quad_conv = quad_conv
        if quad_conv:
            self.conv = PreShuffConv(ni, nf, kernel_size=kernel_size)
        else:
            self.conv = spectral_norm(
                nn.Conv2d(ni, nf,
                          kernel_size,
                          padding=kernel_size // 2,
                          stride=stride,
                          bias=bias))

        self.relu = SReLU(slope=.1, sub=.4)

    def forward(self, x):
        # store input for res
        x = self.relu(self.conv(x))

        return x


class ResBlock(nn.Module):
    # Upres block which uses pixel shuffle with res connection
    def __init__(self, c, kernel_size=3):
        super(ResBlock, self).__init__()

        self.conv_a = spectral_norm(nn.Conv2d(c, c,
                                              kernel_size,
                                              padding=kernel_size // 2,
                                              stride=1,
                                              bias=False))

        self.conv_b = spectral_norm(nn.Conv2d(c, c,
                                              kernel_size,
                                              padding=kernel_size // 2,
                                              stride=1,
                                              bias=False))

        self.relu = SReLU(slope=.1, sub=.4)

        self.bn_a = nn.BatchNorm2d(c)
        self.bn_b = nn.BatchNorm2d(c)

    def forward(self, x):
        # store input for res
        input_tensor = x

        x = self.relu(self.conv_a(x))
        x = self.bn_a(x)
        x = self.conv_b(x)
        x = x + (input_tensor * .2)
        x = self.relu(x)
        x = self.bn_b(x)

        return x


class UpResBlock(nn.Module):
    # Upres block which uses pixel shuffle
    def __init__(self, ic, oc, kernel_size=3):
        super(UpResBlock, self).__init__()

        self.oc = oc
        self.conv = ConvBlock(ic, oc * 4,
                              kernel_size=kernel_size,
                              quad_conv=True,
                              stride=1,
                              bias=False)

        self.bn = nn.BatchNorm2d(oc * 4)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        # store input for res
        x = self.conv(x)
        x = self.bn(x)

        x = self.ps(x)
        return x


class DownRes(nn.Module):
    # DownRes block, with or without batchnorm

    def __init__(self, ic, oc, kernel_size=3, enc=False):
        super(DownRes, self).__init__()

        self.kernel_size = kernel_size
        self.oc = oc
        self.use_bn = enc

        if enc:
            self.conv = ConvBlock(ic, oc,
                                  kernel_size=kernel_size,
                                  quad_conv=False,
                                  bias=False)

            self.bn = nn.BatchNorm2d(oc)
        else:
            self.conv = ConvBlock(ic, oc,
                                  kernel_size=kernel_size,
                                  quad_conv=False,
                                  bias=True)

    def forward(self, x):

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)

        return x


class SelfAttention(nn.Module):
    # Self Attention Based of SAGAN

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = spectral_norm(nn.Conv2d(in_channels=in_dim,
                                                  out_channels=in_dim // 8,
                                                  kernel_size=1))

        self.key_conv = spectral_norm(nn.Conv2d(in_channels=in_dim,
                                                out_channels=in_dim // 8,
                                                kernel_size=1))

        self.value_conv = spectral_norm(nn.Conv2d(in_channels=in_dim,
                                                  out_channels=in_dim,
                                                  kernel_size=1))

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, c, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, c, width, height)

        out = self.gamma * out + x
        return out


############################################################################
# Encoder and Decoder
############################################################################


class Decoder(nn.Module):
    # Decodes latent vector into output image

    def __init__(self, layers=4, max_filts=1024, min_filts=64, channels=3, attention=False):
        super(Decoder, self).__init__()
        operations = []
        print('decoder')
        filt_count = min_filts

        for a in range(layers):
            print('up_block')
            operations += [UpResBlock(int(min(max_filts, filt_count * 2)),
                                      int(min(max_filts, filt_count)))]

            if a in [layers - 2, layers - 3] and attention:
                print('attn')
                operations += [SelfAttention(int(min(max_filts, filt_count * 2)))]

                if a == layers - 3:
                    operations += [ResBlock(int(min(max_filts, filt_count * 2)))]
            filt_count = int(filt_count * 2)

        operations.reverse()

        operations += [nn.ReflectionPad2d(3)]

        self.rgb_out = spectral_norm(
            nn.Conv2d(in_channels=min_filts,
                      out_channels=channels,
                      kernel_size=7,
                      padding=0,
                      stride=1))

        self.alpha_out = spectral_norm(
            nn.Conv2d(in_channels=min_filts,
                      out_channels=1,
                      kernel_size=7,
                      padding=0,
                      stride=1))

        self.model = nn.Sequential(*operations)

    def forward(self, x):
        x = self.model(x)
        rgb = self.rgb_out(x)
        alpha = self.alpha_out(x)
        return torch.cat([F.sigmoid(alpha), F.tanh(rgb)], dim=1)


class Encoder(nn.Module):
    # Encodes input image into a latent vector
    def __init__(self, channels=3, filts_min=64, filts=1024, layers=5, attention=False):
        super(Encoder, self).__init__()
        operations = []
        print('encoder')
        in_operations = [nn.ReflectionPad2d(3),
                         spectral_norm(
                             nn.Conv2d(in_channels=channels,
                                       out_channels=filts_min,
                                       kernel_size=7,
                                       stride=1))]

        filt_count = filts_min

        for a in range(layers):
            print('down block')
            operations += [DownRes(ic=min(filt_count, filts),
                                   oc=min(filt_count * 2, filts),
                                   kernel_size=3,
                                   enc=True)]

            if a in [1, 2] and attention:
                print('attn')
                operations += [SelfAttention(min(filt_count * 2, filts))]
            print(min(filt_count * 2, filts))
            filt_count = int(filt_count * 2)

        operations = in_operations + operations
        self.operations = nn.Sequential(*operations)

        self.linear_operations = nn.Sequential(nn.Linear(4 * 4 * 1024, 2048),
                                               nn.Linear(2048, 4 * 4 * 1024))

        self.output_operations = UpResBlock(int(min(filts, filt_count)),
                                            int(min(filts, filt_count // 2)))

    def forward(self, x):
        bs = x.shape[0]
        x = self.operations(x)
        x = self.linear_operations(x.view(bs, -1))
        x = self.output_operations(x.view(bs, 1024, 4, 4))
        x = self.output_operations(x)
        return x


############################################################################
# Discriminator and VGGFace
############################################################################


class Discriminator(nn.Module):
    # Discriminator roughly based on SRGAN

    def __init__(self, channels=3, filts_min=64, filts=256, kernel_size=4, layers=4, attention=False):
        super(Discriminator, self).__init__()
        print('discriminator')
        operations = []

        in_operations = [nn.ReflectionPad2d(3),
                         spectral_norm(
                             nn.Conv2d(in_channels=channels,
                                       out_channels=filts_min,
                                       kernel_size=7,
                                       stride=1))]

        filt_count = filts_min

        for a in range(layers):
            operations += [DownRes(ic=min(filt_count, filts),
                                   oc=min(filt_count * 2, filts),
                                   kernel_size=3)]

            if a > 2 and attention:
                print('attn')
                operations += [SelfAttention(min(filt_count * 2, filts))]
            print(min(filt_count * 2, filts))
            filt_count = int(filt_count * 2)

        out_operations = [
            spectral_norm(
                nn.Conv2d(in_channels=min(filt_count, filts),
                          out_channels=1,
                          padding=1,
                          kernel_size=kernel_size,
                          stride=1))]

        operations = in_operations + operations + out_operations
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        # Run operations, and return relu activations for loss function
        x = self.operations(x)
        return x


def resnet_face():
    return rf.resnet50_ft_dag(weights_path='resnet50_ft_dag.pth')


############################################################################
# Hook and Losses
############################################################################


class SetHook:
    # Register hook inside of network to retrieve features

    feats = None
    def __init__(self, block):
        self.hook_reg = block.register_forward_hook(self.hook)

    def hook(self, module, hook_input, output):
        self.feats = output

    def close(self):
        self.hook_reg.remove()


def emboss(x, axis=2):
    # create embossed version of image in X or Y

    img_nrows = x.shape[2]
    img_ncols = x.shape[3]
    if axis == 2:
        return torch.abs(x[:, :, :img_nrows - 1, :img_ncols - 1, ] - x[:, :, 1:, :img_ncols - 1])
    elif axis == 3:
        return torch.abs(x[:, :, :img_nrows - 1, :img_ncols - 1, ] - x[:, :, :img_nrows - 1, 1:])
    else:
        return None


def edge_loss(fake_img, real_img, edge_weight):
    # Calculate Edge Loss

    result_face = F.l1_loss(emboss(fake_img, axis=2),
                            emboss(real_img, axis=2)) * edge_weight

    result_face += F.l1_loss(emboss(fake_img, axis=3),
                             emboss(real_img, axis=3)) * edge_weight
    return result_face


def recon_loss(fake_img, real_img, recon_weight):
    # Calculate L1 Loss
    result_face = F.l1_loss(fake_img, real_img) * recon_weight
    return result_face


class PerceptualLoss(nn.Module):
    # Store Hook, Calculate Perceptual Loss

    def __init__(self, vgg, ct_wgt, perceptual_layer_ids, weight_list, hooks=None, use_instance_norm=False):
        super().__init__()
        self.m, self.ct_wgt = vgg, ct_wgt
        self.use_instance_norm = use_instance_norm
        self.IN = nn.InstanceNorm2d(3)
        if not hooks:
            self.cfs = [SetHook(vgg[i]) for i in perceptual_layer_ids]
        else:
            print('Using custom hooks')
            self.cfs = hooks

        ratio = ct_wgt / sum(weight_list)
        weight_list = [a * ratio for a in weight_list]
        self.weight_list = weight_list

    def forward(self, fake_img, real_img, disc_mode=False):
        if sum(self.weight_list) > 0.0:
            self.m(real_img.data)
            targ_feats = [o.feats.data.clone() for o in self.cfs]
            fake_result = self.m(fake_img)
            inp_feats = [o.feats for o in self.cfs]

            if self.use_instance_norm:
                result_perc = [F.l1_loss(self.IN(inp).view(-1), self.IN(targ).view(-1)) * layer_weight for
                               inp, targ, layer_weight in
                               zip(inp_feats, targ_feats, self.weight_list)]
            else:
                result_perc = [F.l1_loss(inp.view(-1), targ.view(-1)) * layer_weight for inp, targ, layer_weight in
                               zip(inp_feats, targ_feats, self.weight_list)]
        else:
            result_perc = [torch.zeros(1).cuda() for layer_weight in self.weight_list]
            fake_result = torch.zeros(1).cuda()

        if not disc_mode:
            return result_perc
        else:
            return result_perc, fake_result

    def close(self):
        [o.remove() for o in self.sfs]
