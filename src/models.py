import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from .networks import VGGEncoder, Decoder, Discriminator, MultiscaleDiscriminator, Classifier, FRN
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, FeatureLoss, GANLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.testing = False

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')


    def train(self):
        super(BaseModel, self).train()
        self.testing = False


    def eval(self):
        super(BaseModel, self).train(False)
        self.testing = False


    def test(self):
        super(BaseModel, self).train(False)
        self.testing = True


    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s encoder and decoder...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.encoder.load_state_dict(data['encoder'])
            self.decoder.load_state_dict(data['decoder'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self, extra=None):
        if extra is not None:
            assert isinstance(extra, str)

        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }, self.gen_weights_path if extra is None else self.gen_weights_path[:-4]+'_'+extra+'.pth')

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path if extra is None else self.dis_weights_path[:-4]+'_'+extra+'.pth')


class InpaintModel(BaseModel):
    def __init__(self, config):
        super(InpaintModel, self).__init__("InpaintModel", config)

        # self.t = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

        # generator input: [rgb(3)]
        # discriminator input: [rgb(3)]
        encoder = VGGEncoder(pretrained=config.VGG_PRETRAINED,
                                            pretrained_path=config.VGG_PRETRAINED_PATH, 
                                            requires_grad=config.VGG_REQUITES_GRAD)
        decoder = Decoder(output_size=config.INPUT_SIZE, dilation=config.DILATION,
                                            use_classification=config.USE_CLASSIFICATION,
                                            num_classes=config.CLASS_NUM,
                                            upsample_type=config.UPSAMPLE_TYPE,
                                            add_mask_info=config.ADD_MASK_INFO,
                                            all_feature=config.ALL_FEATURE,
                                            use_latent=config.USE_LATENT_VECTOR)

        if config.USE_FRN:
            frn = FRN(dilation=config.DILATION)

        if not config.MULTI_SCALE_DIS:
            discriminator = Discriminator(in_channels=3+config.CLASS_NUM if config.USE_CLASSIFICATION else 3, use_sigmoid=config.GAN_LOSS != 'hinge')
        else:
            discriminator = MultiscaleDiscriminator(input_nc=3+config.CLASS_NUM if config.USE_CLASSIFICATION else 3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, 
                    use_sigmoid=False, num_D=2, getIntermFeat=True)

        if config.WITH_CLASSIFIER:
            assert config.USE_CLASSIFICATION
            classifier = Classifier(pretrained_path=config.VGG_PRETRAINED_PATH,
                                    requires_grad=config.CLASSIFIER_REQUIRES_GRAD)


        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        if not config.MULTI_SCALE_DIS:
            adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        else:
            gan_loss = GANLoss(gan_mode=config.GAN_LOSS)
        if config.WITH_CLASSIFIER:
            cls_loss = nn.CrossEntropyLoss()
        if config.MULTI_TASK:
            ce_loss = nn.CrossEntropyLoss()


        if len(config.GPU) > 1:
            encoder = nn.DataParallel(encoder, config.GPU)
            decoder = nn.DataParallel(decoder, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)
            if config.WITH_CLASSIFIER:
                classifier = nn.DataParallel(classifier, config.GPU)
            # l1_loss = nn.DataParallel(l1_loss, config.GPU)
            # perceptual_loss = nn.DataParallel(perceptual_loss, config.GPU)
            # if not config.MULTI_SCALE_DIS:
            #     adversarial_loss = nn.DataParallel(adversarial_loss, config.GPU)
            # else:
            #     gan_loss = nn.DataParallel(gan_loss, config.GPU)
            # if config.WITH_CLASSIFIER:
            #     cls_loss = nn.DataParallel(cls_loss, config.GPU)
            # if config.MULTI_TASK:
            #     ce_loss = nn.DataParallel(ce_loss, config.GPU)


        self.add_module('encoder', encoder)
        self.add_module('decoder', decoder)
        if config.USE_FRN:
            self.add_module('frn', frn)
        self.add_module('discriminator', discriminator)
        if config.WITH_CLASSIFIER:
            self.add_module('classifier', classifier)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        if not config.MULTI_SCALE_DIS:
            self.add_module('adversarial_loss', adversarial_loss)
        else:
            self.add_module('gan_loss', gan_loss)
        if config.WITH_CLASSIFIER:
            self.add_module('cls_loss', cls_loss)
        if config.MULTI_TASK:
            self.add_module('ce_loss', ce_loss)


        gen_parameters = list(decoder.parameters()) 
        if config.USE_FRN:
            gen_parameters += list(frn.parameters())
        if config.VGG_REQUITES_GRAD:
            gen_parameters += list(encoder.parameters())
        self.gen_optimizer = optim.Adam(
            params=gen_parameters,
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

        if config.WITH_CLASSIFIER and config.CLASSIFIER_REQUIRES_GRAD:
            self.cls_optimizer = optim.Adam(params=classifier.parameters(),
                lr=float(config.LR) * float(config.D2G_LR),
                betas=(config.BETA1, config.BETA2)
            )


    def process(self, data):
        if self.training:
            self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        if self.config.WITH_CLASSIFIER and self.config.CLASSIFIER_REQUIRES_GRAD:
            self.cls_optimizer.zero_grad()


        # preprocess inputs
        data = self.preprocess_input(data)
        images = data['image']
        masks = data['mask']
        images_masked = data['images_masked']
        labelmap = data['labelmap']
        labels = data['label']


        # process outputs
        middle = self.encoder(images_masked)
        if self.config.USE_FRN:
            middle = self.frn(middle)

        if labelmap is None and self.config.USE_CLASSIFICATION:
            labelmap = self.pred2labelmap(middle[-1], size=images_masked.shape[2:])

        if self.config.ADD_MASK_INFO and self.config.USE_CLASSIFICATION:
            outputs, x4, x3, x2, x1, x0, pred = self.decoder(middle, torch.cat([masks, labelmap], dim=1))
        elif self.config.ADD_MASK_INFO:
            outputs, x4, x3, x2, x1, x0, pred = self.decoder(middle, masks)
        else:
            outputs, x4, x3, x2, x1, x0, pred = self.decoder(middle, labelmap)
        gen_loss = 0
        dis_loss = 0
        cls_loss = None

        if self.testing == True:
            return outputs


        # classifier loss
        if self.config.WITH_CLASSIFIER and self.config.CLASSIFIER_REQUIRES_GRAD:
            y_pred = self.classifier(outputs.detach())
            cls_loss = self.cls_loss(y_pred, labels)
        

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real, labelmap)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake, labelmap)                    # in: [rgb(3)]
        if not self.config.MULTI_SCALE_DIS:
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        else:
            dis_real_loss = self.gan_loss(dis_real, True, for_discriminator=True)
            dis_fake_loss = self.gan_loss(dis_fake, False, for_discriminator=True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake, labelmap)                    # in: [rgb(3)]
        if not self.config.MULTI_SCALE_DIS:
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        else:
            gen_gan_loss = self.gan_loss(gen_fake, True, for_discriminator=False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


        # classification loss
        if self.config.WITH_CLASSIFIER:
            y_pred = self.classifier(outputs)
            gen_cls_loss = self.cls_loss(y_pred, labels) * self.config.CLASSIFIER_LOSS_WEIGHT
            gen_loss += gen_cls_loss

        
        # frn loss
        if self.config.USE_FRN and self.config.FRN_LOSS:
            with torch.no_grad():
                middle_original = self.encoder(images.sub(self.mean[None, :, None, None].to(images)).div_(self.std[None, :, None, None].to(images)))
            frn_loss = 0.
            frn_loss_fmap0 = F.l1_loss(middle[1], middle_original[1])
            frn_loss_fmap1 = F.l1_loss(middle[2], middle_original[2])
            frn_loss_fmap2 = F.l1_loss(middle[3], middle_original[3])
            frn_loss_fmap3 = F.l1_loss(middle[4], middle_original[4])
            frn_loss_fmap4 = F.l1_loss(middle[5], middle_original[5])
            frn_loss = frn_loss_fmap0 + frn_loss_fmap1+ frn_loss_fmap2 + frn_loss_fmap3 + frn_loss_fmap4
            gen_loss += frn_loss

        
        # ce loss
        if self.config.MULTI_TASK:
            ce_loss = self.ce_loss(pred, labels)
            gen_loss += ce_loss


        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        if self.config.WITH_CLASSIFIER:
            logs += [("l_cls", gen_cls_loss.item())]

        if self.config.USE_FRN and self.config.FRN_LOSS:
            logs += [("l_frn", frn_loss.item())]

        if self.config.MULTI_TASK:
            logs += [("ce_loss", ce_loss.item())]
            

        return outputs, gen_loss, dis_loss, cls_loss, logs

    
    def preprocess_input(self, data):
        images, masks, labels = data['image'], data['mask'], data['label']

        images_masked = (images * (1 - masks).float())

        # Normalizae masked image as vgg model did
        images_masked = images_masked.sub(self.mean[None, :, None, None].to(images_masked)).div_(self.std[None, :, None, None].to(images_masked))
        # for i in range(images_masked.size(0)):
        #     images_masked[i] = self.t(images_masked[i])
        images_masked = (images_masked * (1 - masks).float())

        if labels.ne(-1).all():
            bs, _, h, w = images_masked.shape
            nc = self.config.CLASS_NUM
            labelmap = torch.zeros(bs, nc, h, w).float()
            labelmap[torch.arange(bs), labels] = 1.
        else:
            labelmap = None

        data['images_masked'] = images_masked.to(images)
        data['labelmap'] = labelmap.to(images) if labelmap is not None else None

        return data


    def backward(self, gen_loss=None, dis_loss=None, cls_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

        if cls_loss is not None:
            cls_loss.backward()
            self.cls_optimizer.step()


    def pred2labelmap(self, pred, size):
        bs, nc = pred.shape
        h, w = size
        labelmap = pred.view(bs, nc, 1, 1).repeat(1,1,h,w)
        return labelmap