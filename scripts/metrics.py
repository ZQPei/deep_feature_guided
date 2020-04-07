import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
from scipy.misc import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray, gray2rgb

import os
import cv2
import scipy


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--log-file', type=str, default='')
    parser.add_argument('--win-size', type=int, default=11)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    parser.add_argument('--gray', action='store_true', help='use gray img')
    parser.add_argument('--centercrop', action='store_true', help='center crop img')
    parser.add_argument('--ext', type=str, default='.png', help='saved images ext')
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


def load_flist(flist):
    if isinstance(flist, list):
        return flist

    if isinstance(flist, np.ndarray):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob(flist + '/*.jpg')) + list(glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                flist = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                if flist.ndim > 1 :
                    flist = flist[:, 0]
                return flist.tolist()
            except:
                return [flist]

    return []


def resize(img, height, width, centerCrop=False):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = scipy.misc.imresize(img, [height, width])

    return img


def to_float(img):
    assert img.dtype == np.uint8
    return img.astype(np.float32) / 255.0


def print_fun(*args, log_file=None):
    print(*args)
    if log_file:
        with open(log_file, 'a+') as f:
            print(*args, file=f)


def print_args(args):
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))


def calculate(path_true, path_pred, output_path, log_file, image_size=256, ssim_winsize=11, gray_mode=False, debug_mode=False, centercrop=False, ext='.png'):
    psnr = []
    ssim = []
    mae = []
    names = []
    index = 1

    if not log_file:
        log_file = os.path.join(output_path, 'metrics.txt')


    # files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
    files = load_flist(path_true)#[:50]
    for fn in sorted(files):
        name = basename(str(fn))
        names.append(name)

        img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
        img_pred = (imread(path_pred + '/' + basename(str(fn)[:-4]+ext)) / 255.0).astype(np.float32)

        if img_gt.ndim == 2:
            img_gt = gray2rgb(img_gt)


        img_gt = resize(img_gt, image_size, image_size, centerCrop=centercrop).astype(np.float32) / 255.0
        img_pred = resize(img_pred, image_size, image_size, centerCrop=False).astype(np.float32) / 255.0

        if gray_mode:
            img_gt = rgb2gray(img_gt)
            img_pred = rgb2gray(img_pred)

        if debug_mode:
            plt.subplot('121')
            plt.imshow(img_gt)
            plt.title('Groud truth')
            plt.subplot('122')
            plt.imshow(img_pred)
            plt.title('Output')
            plt.show()

        psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
        ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=ssim_winsize, multichannel=True))
        mae.append(compare_mae(img_gt, img_pred))
        if np.mod(index, 100) == 0:
            print(
                str(index) + ' images processed',
                "PSNR: %.4f" % round(np.mean(psnr), 4),
                "SSIM: %.4f" % round(np.mean(ssim), 4),
                "MAE: %.4f" % round(np.mean(mae), 4),
            )
        index += 1

    np.savez(output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
    print_fun(
        "PSNR: %.4f" % round(np.mean(psnr), 4),
        "PSNR Variance: %.4f" % round(np.var(psnr), 4),
        "SSIM: %.4f" % round(np.mean(ssim), 4),
        "SSIM Variance: %.4f" % round(np.var(ssim), 4),
        "MAE: %.4f" % round(np.mean(mae), 4),
        "MAE Variance: %.4f" % round(np.var(mae), 4),
        log_file=log_file
    )

    print()
    for i in range(6):
        start = i * 2000
        end = (i + 1) * 2000
        print_fun("mask ratio: [%2d%% - %2d%%]"%(i*10,(i+1)*10), log_file=log_file)
        print_fun(
            "PSNR: %.4f" % round(np.mean(psnr[start:end]), 4),
            "PSNR Variance: %.4f" % round(np.var(psnr[start:end]), 4),
            "SSIM: %.4f" % round(np.mean(ssim[start:end]), 4),
            "SSIM Variance: %.4f" % round(np.var(ssim[start:end]), 4),
            "MAE: %.4f" % round(np.mean(mae[start:end]), 4),
            "MAE Variance: %.4f" % round(np.var(mae[start:end]), 4),
            log_file=log_file
        )


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    path_true = args.data_path
    path_pred = args.output_path
    output_path = args.output_path
    log_file = args.log_file
    image_size = args.image_size
    win_size = args.win_size
    gray = args.gray
    debug = args.debug
    centercrop = args.centercrop
    ext = args.ext


    calculate(path_true, path_pred, output_path, log_file, 
                image_size=image_size,
                ssim_winsize=win_size,
                gray_mode=gray, debug_mode=debug, centercrop=centercrop,
                ext=ext)

    # import ipdb; ipdb.set_trace()