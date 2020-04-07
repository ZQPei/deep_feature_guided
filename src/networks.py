import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from collections import OrderedDict

from .vgg import vgg19
from .multiscale_discriminator import MultiscaleDiscriminator


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class VGGEncoder(BaseNetwork):
    def __init__(self, pretrained=False, pretrained_path="./pretrained/vgg19_places2_ep83_acc87.pth", requires_grad=False):
        super(VGGEncoder, self).__init__()

        # vgg = vgg19(pretrained=False)
        # vgg.load_state_dict(torch.load("./pretrained/vgg19-dcbb9e9d.pth"))
        vgg = vgg19(pretrained=False, input_size=256, num_classes=365, init_weights=True)
        if pretrained:
            print("Loading vgg pretrained parameters from %s ..."%pretrained_path)
            state_dict = torch.load(pretrained_path, map_location="cpu")
            vgg.load_state_dict(
                OrderedDict({k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}))

        # MaxPooling layer ids are 4, 9, 18, 27, 36
        ml1, ml2, ml3, ml4, ml5 = 4, 9, 18, 27, 36    # size 256 128 64 32 16  nc 64 128 256 512 512
        ml1, ml2, ml3, ml4, ml5 = ml1 + 1, ml2 + 1, ml3 + 1, ml4 + 1, ml5 + 1    # size 128 64 32 16 8  nc 64 128 256 512 512
        feature_0 = vgg.features[:ml1]
        feature_1 = vgg.features[ml1:ml2]
        feature_2 = vgg.features[ml2:ml3]
        feature_3 = vgg.features[ml3:ml4]
        feature_4 = vgg.features[ml4:ml5]

        avg_pool = vgg.avgpool
        classifier = vgg.classifier

        self.add_module("feature_0", feature_0)
        self.add_module("feature_1", feature_1)
        self.add_module("feature_2", feature_2)
        self.add_module("feature_3", feature_3)
        self.add_module("feature_4", feature_4)

        self.add_module("avg_pool", avg_pool)
        self.add_module("classifier", classifier)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, images_masked):
        # 256 128 64 32 16 8
        fmap0 = self.feature_0(images_masked)
        fmap1 = self.feature_1(fmap0)
        fmap2 = self.feature_2(fmap1)
        fmap3 = self.feature_3(fmap2)
        fmap4 = self.feature_4(fmap3)

        y = self.avg_pool(fmap4)
        y = y.view(y.size(0), -1)
        pred = self.classifier(y)

        pred = torch.sigmoid(pred)
        tmp = torch.zeros_like(pred).to(pred)
        tmp[torch.arange(pred.size(0)), pred.argmax(dim=1)] = 1
        pred = tmp

        # for inputsize 256, return size 8
        out = [images_masked, fmap0, fmap1, fmap2, fmap3, fmap4, pred]
        return out


class FRN(BaseNetwork):
    def __init__(self, nd=5, dilation=2, use_classification=False, num_classes=365, init_weights=True, upsample_type='upsample'):
        super(FRN, self).__init__()

        nb0, nb1, nb2, nb3, nb4 = 1,1,1,1,1
        nc0, nc1, nc2, nc3, nc4 = 512, 512, 256, 128, 64
        self.dilated_conv0 = self.build_dilated_blocks(nc0, nc0, nb0, dilation=dilation)
        self.dilated_conv1 = self.build_dilated_blocks(nc1, nc1, nb1, dilation=dilation)
        self.dilated_conv2 = self.build_dilated_blocks(nc2, nc2, nb2, dilation=dilation)
        self.dilated_conv3 = self.build_dilated_blocks(nc3, nc3, nb3, dilation=dilation)
        self.dilated_conv4 = self.build_dilated_blocks(nc4, nc4, nb4, dilation=dilation)

        self.upsample1 = self.build_upsample_blocks(nc0, nc1)
        self.upsample2 = self.build_upsample_blocks(nc1, nc2)
        self.upsample3 = self.build_upsample_blocks(nc2, nc3)
        self.upsample4 = self.build_upsample_blocks(nc3, nc4)


    def forward(self, inputs):
        images_masked, fmap0, fmap1, fmap2, fmap3, fmap4, pred = inputs
        x0 = self.dilated_conv0(fmap4)
        x1 = self.dilated_conv1(fmap3 + self.upsample1(x0))
        x2 = self.dilated_conv2(fmap2 + self.upsample2(x1))
        x3 = self.dilated_conv3(fmap1 + self.upsample3(x2))
        x4 = self.dilated_conv4(fmap0 + self.upsample4(x3))

        out = [images_masked, x4, x3, x2, x1, x0, pred]
        return out


    def build_dilated_blocks(self, cin, cout, nb=2, dilation=1):
        blocks = []
        for i in range(nb):
            blocks += [nn.ReflectionPad2d(dilation),
                        nn.Conv2d(cin, cout if i == nb-1 else cin, kernel_size=3, stride=1, padding=0, dilation=dilation),
                        nn.InstanceNorm2d(cout if i == nb-1 else cin, track_running_stats=False),
                        nn.ReLU(True)]
        return nn.Sequential(*blocks)

    def build_upsample_blocks(self, cin, cout):
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(cout, track_running_stats=False),
            nn.ReLU(True),
        )


class Decoder(BaseNetwork):
    def __init__(self, nd=5, output_size=256, dilation=2, use_classification=False, num_classes=365, init_weights=True, upsample_type='upsample', add_mask_info=False, use_latent=False, all_feature=False):
        super(Decoder, self).__init__()

        self.use_latent = use_latent

        self.min_size = output_size // (2 ** nd)

        nb0, nb1, nb2, nb3, nb4 = 2,1,1,1,1
        nc0, nc1, nc2, nc3, nc4 = 512, 512, 256, 128, 64

        ns0, ns1, ns2, ns3, ns4 = 512, 512, 256, 128, 64
        if all_feature:
            ns0, ns1, ns2, ns3, ns4 = 1472, 1472, 1472, 1472, 1472
        self.all_feature = all_feature
        if use_classification:
            ns0, ns1, ns2, ns3, ns4 = ns0 + num_classes, ns1 + num_classes, ns2 + num_classes, ns3 + num_classes, ns4 + num_classes
        self.use_classification = use_classification
        if add_mask_info:
            ns0, ns1, ns2, ns3, ns4 = ns0 + 1, ns1 + 1, ns2 + 1, ns3 + 1, ns4 + 1
        self.add_mask_info = add_mask_info

        self.head = nn.Conv2d(3, 512, 3, 1, padding=1)

        self.middle_0, self.upsample_0 = self.build_blocks(nc0, nc0, ns0, num_blks=nb0, dilation=dilation, upsample_type='upsample', resblk=SPADEResnetBlock)
        self.middle_1, self.upsample_1 = self.build_blocks(nc0, nc1, ns1, num_blks=nb1, dilation=dilation, upsample_type='upsample', resblk=SPADEResnetBlock)
        self.middle_2, self.upsample_2 = self.build_blocks(nc1, nc2, ns2, num_blks=nb2, dilation=dilation, upsample_type='upsample', resblk=SPADEResnetBlock)
        self.middle_3, self.upsample_3 = self.build_blocks(nc2, nc3, ns3, num_blks=nb3, dilation=dilation, upsample_type='upsample', resblk=SPADEResnetBlock)
        self.middle_4, self.upsample_4 = self.build_blocks(nc3, nc4, ns4, num_blks=nb4, dilation=dilation, upsample_type='upsample', resblk=SPADEResnetBlock)

        self.final = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(nc4, 3, 3, padding=0)
        )

        if init_weights:
            self.init_weights()


    def forward(self, inputs, labelmap):
        images_masked, fmap0, fmap1, fmap2, fmap3, fmap4, pred = inputs
        x = fmap4

        if not self.use_latent:
            x = F.interpolate(fmap4, size=(self.min_size, self.min_size))
        else:
            x = F.interpolate(images_masked, size=(self.min_size, self.min_size))
            x = self.head(x)


        if self.all_feature:
            fmap4 = fmap3 = fmap2 = fmap1 = fmap0 = self.concat_fmap(fmap0, fmap1, fmap2, fmap3, fmap4)

        x0 = self.middle_0([x, self.concat_labelmap(fmap4, labelmap)])[0]
        x1 = self.upsample_0(x0)
        x1 = self.middle_1([x1, self.concat_labelmap(fmap3, labelmap)])[0]
        x2 = self.upsample_1(x1)
        x2 = self.middle_2([x2, self.concat_labelmap(fmap2, labelmap)])[0]
        x3 = self.upsample_2(x2)
        x3 = self.middle_3([x3, self.concat_labelmap(fmap1, labelmap)])[0]
        x4 = self.upsample_3(x3)
        x4 = self.middle_4([x4, self.concat_labelmap(fmap0, labelmap)])[0]
        outputs = self.upsample_4(x4)

        outputs = self.final(outputs)
        outputs = (torch.tanh(outputs) + 1) / 2

        return outputs, x4, x3, x2, x1, x0, pred  # size 256 128 64 32 16 8  # nc 3 64 128 256 512 512


    def concat_labelmap(self, fmap, labelmap):
        if labelmap is not None:
            fmap = torch.cat([fmap, F.interpolate(labelmap, size=fmap.shape[2:])], dim=1)
        return fmap

    def concat_fmap(self, *fmap):
        max_size = fmap[0].shape[-1]
        fmap = torch.cat([F.interpolate(f, size=max_size) for f in fmap], dim=1)
        return fmap


    def build_blocks(self, cin, cout, nc, resblk, num_blks=1, dilation=1, upsample_type='upsample'):
        blocks = []
        for i in range(num_blks):
            blocks += [resblk(cin, cout if i == num_blks - 1 else cin, nc, dilation=dilation)]
        res_layers = nn.Sequential(*blocks)

        if upsample_type == 'upsample':
            upsample_layer = nn.Upsample(scale_factor=2)
        elif upsample_layer == 'deconv':
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels=cout, out_channels=cout, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(64, track_running_stats=False),
                nn.ReLU(True),
            )

        return res_layers, upsample_layer


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x, labelmap):
        if labelmap is not None:
            x = torch.cat([x, labelmap], dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class Classifier(BaseNetwork):
    def __init__(self, pretrained_path="./pretrained/vgg19_places2_ep83_acc87.pth", requires_grad=True):
        super(Classifier, self).__init__()

        print("Loading vgg pretrained parameters from %s ..."%pretrained_path)
        # vgg = vgg19(pretrained=False)
        # vgg.load_state_dict(torch.load("./pretrained/vgg19-dcbb9e9d.pth"))
        vgg = vgg19(pretrained=False, input_size=256, num_classes=365, init_weights=False)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        vgg.load_state_dict(
            OrderedDict({k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}))

        features = vgg.features
        avg_pool = vgg.avgpool
        classifier = vgg.classifier

        self.add_module("features", features)
        self.add_module("avg_pool", avg_pool)
        self.add_module("classifier", classifier)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SPADE(nn.Module):
    def __init__(self, norm_nc, input_nc, use_multi_conv=False, num_conv=1):
        super(SPADE, self).__init__()

        self.use_multi_conv = use_multi_conv

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False) # affine means no params norm layer

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        # nhidden = input_nc // 2 if input_nc != 3 else 64
        
        ks = 3
        pw = ks // 2

        if use_multi_conv:
            raise NotImplementedError

        else:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(input_nc, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU(True)
            )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, fmap=None):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        if self.use_multi_conv:
            assert NotImplementedError

        else:
            fmap = F.interpolate(fmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(fmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, input_nc=64, dilation=1, norm_type='spectral'):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1, dilation=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_type:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        # input_nc = 4
        # input_nc = 64
        self.norm_0 = SPADE(fin, input_nc)
        self.norm_1 = SPADE(fmiddle, input_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, input_nc)

    # note the resnet block with SPADE also takes in |fmap|,
    def forward(self, x):
        x, fmap = x
        x_s = self.shortcut(x, fmap)

        dx = self.conv_0(self.actvn(self.norm_0(x, fmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, fmap)))

        out = x_s + dx

        return out, fmap

    def shortcut(self, x, fmap):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, fmap))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module



class SPADE_REV(nn.Module):
    def __init__(self, norm_nc, input_nc, use_multi_conv=False, num_conv=1):
        super(SPADE_REV, self).__init__()

        self.use_multi_conv = use_multi_conv

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, track_running_stats=False) # affine means no params norm layer

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        # nhidden = input_nc // 2 if input_nc != 3 else 64
        
        ks = 3
        pw = ks // 2

        if use_multi_conv:
            raise NotImplementedError

        else:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(input_nc, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU(True)
            )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, fmap=None):
        # Part 1. generate parameter-free normalized activations
        # normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        if self.use_multi_conv:
            assert NotImplementedError

        else:
            fmap = F.interpolate(fmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(fmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = self.param_free_norm(x * (1 + gamma) + beta)

        return out


class SPADEResnetBlock_REV(nn.Module):
    def __init__(self, fin, fout, input_nc=64, dilation=1, norm_type='spectral'):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1, dilation=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_type:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        # input_nc = 4
        # input_nc = 64
        self.norm_0 = SPADE_REV(fin, input_nc)
        self.norm_1 = SPADE_REV(fmiddle, input_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE_REV(fin, input_nc)

    # note the resnet block with SPADE also takes in |fmap|,
    def forward(self, x):
        x, fmap = x
        x_s = self.shortcut(x, fmap)

        dx = self.conv_0(self.actvn(self.norm_0(x, fmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, fmap)))

        out = x_s + dx

        return out, fmap

    def shortcut(self, x, fmap):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, fmap))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


if __name__ == "__main__":
    fpn = FPN()
    x0 = torch.randn(1,512, 4, 4)
    x1 = F.interpolate(x0, scale_factor=2)
    x2 = F.interpolate(x1, scale_factor=2)
    x3 = F.interpolate(x2, scale_factor=2)
    x4 = F.interpolate(x3, scale_factor=2)
    
    import ipdb; ipdb.set_trace()

    fpn([1,x4,x3,x2,x1,x0,1])