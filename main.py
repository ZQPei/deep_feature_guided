import os
import cv2
import random
import numpy as np
import torch
import argparse
import importlib

from shutil import copyfile
from src import Config, get_model


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """


    config = load_config(mode)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)


    if mode == 1:
        # copy network files as a backup when training
        os.system("cp -r ./src %s/"%config.PATH)

    elif mode == 2:
        # select source code when testing
        global Config, get_model
        src = importlib.import_module("checkpoints.%s.src"%(os.path.basename(config.PATH)))
        Config = src.Config
        get_model = src.get_model


    # cuda visble devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        CUDA_VISIBLE_DEVICES = [int(x) for x in list(os.environ.get("CUDA_VISIBLE_DEVICES")) if x.isdigit()]
        config.GPU = list(range(len(CUDA_VISIBLE_DEVICES)))


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # import inpainting model
    InpaintingModel = get_model(model=config.MODEL)


    # build the model and initialize
    model = InpaintingModel(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        model.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1], help='1: deep_feature_guided')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], help='1: train, 2: test, 3: eval')
    parser.add_argument('--input-size', type=int, default=256, help='1: train, 2: test, 3: eval')
    parser.add_argument('--debug', action='store_true', help='debug')

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    os.makedirs(args.path, exist_ok=True)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml', config_path)


    # load config file
    config = Config(config_path)


    # class info
    if config.USE_CLASSIFICATION:
        config.CLASS_DICT = importlib.import_module("datasets.%s"%config.DATASET).CLASS_DICT


    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else config.MODEL

        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else config.MODEL

    # activate debug
    config.DEBUG = args.debug

    # activate debuging
    if config.DEBUG:
        config.SAVE_INTERVAL = 10
        config.METRIC_INTERVAL = 10
        config.SAMPLE_INTERVAL = 10
        config.EVAL_INTERVAL = 10
        config.TEST_INTERVAL = 10
        config.LOG_INTERVAL = 10
        config.MAX_ITERS = 30

    return config


if __name__ == "__main__":
    main()
