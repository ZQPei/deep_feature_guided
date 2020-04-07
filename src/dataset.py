import os
import glob
import scipy
import torch
import torchvision
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, image_flist, mask_flist, augment=True, training=True, debug=False):
        super(Dataset, self).__init__()
        self.mode = config.MODE
        self.augment = augment
        self.training = training
        self.center_crop = config.CENTER_CROP_WHEN_TRAIN


        # flist
        self.image_data = self.load_flist(image_flist)
        self.label_data = None
        self.mask_data = self.load_flist(mask_flist)

        assert self.image_data.shape[0] > 0
        if self.image_data.ndim > 1:
            self.label_data = self.image_data[:,1]
            self.image_data = self.image_data[:,0]

        if debug:
            self.image_data = self.image_data[:50]
            self.label_data = self.label_data[:50] if self.label_data is not None else None
            self.mask_data = self.mask_data[:50]


        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK

        self.mask_reverse = config.MASK_REVERSE

        self.t = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.use_classification = config.USE_CLASSIFICATION
        if self.use_classification:
            self.class_dict = config.CLASS_DICT
            self.class_num = config.CLASS_NUM


        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if not self.training:
            self.mask = 6


    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.image_data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.image_data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        fn = self.image_data[index]
        img = imread(fn)

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if self.training:
            img = self.resize(img, size, size, centerCrop=self.center_crop)
        else:
            img = self.resize(img, size, size, centerCrop=False)


        # load mask
        mask = self.load_mask(img, index)


        # load class index
        if self.use_classification and self.label_data is not None:
            label = self.label_data[index]
            label = int(label)
        elif self.use_classification:
            try:
                label = self.load_class(fn)
            except:
                label = -1
        else:
            label = -1


        # augment data
        if self.augment:
            # random horizontal flip
            if np.random.binomial(1, 0.5) > 0:
                img = img[:, ::-1, ...]

            mask = self.augment_mask(mask)

        im_tensor = self.to_tensor(img, normalize=False)
        mask_tensor = self.to_tensor(mask, self.mask_reverse)


        data = {
            'image': im_tensor,
            'mask': mask_tensor,
            'label': label,
        }

        return data

    
    def load_class(self, fn):
        class_name = fn.split("/")[-2]
        return self.class_dict[class_name]


    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            # mask = imread(self.mask_data[mask_index])
            # mask = self.resize(mask, imgh, imgw, centerCrop=False)
            # mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            mask_pil = Image.open(self.mask_data[mask_index])
            mask_pil = mask_pil.resize([imgw, imgh])
            mask = np.array(mask_pil)
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            # mask = imread(self.mask_data[index])
            # mask = self.resize(mask, imgh, imgw, centerCrop=False)
            # mask = rgb2gray(mask)
            # mask = (mask > 0).astype(np.uint8) * 255
            mask_pil = Image.open(self.mask_data[index])
            mask_pil = mask_pil.resize([imgw, imgh])
            mask = np.array(mask_pil)
            return mask


    def to_tensor(self, img, reverse=False, normalize=False):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()

        if reverse:
            return 1 - img_t

        if normalize:
            return self.t(img_t)

        return img_t


    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    @staticmethod
    def hflip(img):
        return img[:, ::-1, ...]

    @staticmethod    
    def vflip(img):
        return img[::-1, :, ...]

    def rotate(self, img, degree):
        if degree == 0:
            pass
        elif degree == 90:
            img = self.vflip(img.T)
        elif degree == 180:
            img = self.hflip(self.vflip(img))
        elif degree == 270:
            img = self.hflip(img.T)

        return img

    
    def augment_mask(self, mask):
        c = np.random.choice(np.arange(7))
        if c == 0:
            pass
        elif c == 1:
            mask = self.rotate(mask, 90)
        elif c == 2:
            mask = self.rotate(mask, 180)
        elif c == 3:
            mask = self.rotate(mask, 270)
        elif c == 4:
            mask = self.hflip(mask)
        elif c == 5:
            mask = self.vflip(mask)
        elif c == 6:
            mask = mask.T

        return mask


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

