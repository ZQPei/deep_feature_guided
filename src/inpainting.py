import os
import sys
import numpy as np
import torch

from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintModel
from .psnr import PSNR
from .ssim import SSIM
from .utils import Progbar, create_dir, stitch_images, imsave, AverageMeta, print_fun
from .htmls import HTML


class BaseInpainting(object):
    def __init__(self, config):
        self.config = config
        self.model = config.MODEL
        self.debug = config.DEBUG
        self.html = config.HTML

        if self.debug:
            config.TRAIN_FLIST = config.VAL_FLIST


        # dataset
        if self.config.MODE == 1 or self.config.MODE == 3:
            # train mode or eval mode
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True, debug=config.DEBUG)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_MASK_FLIST, augment=False, training=True, debug=config.DEBUG)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, training=False, debug=config.DEBUG)
        else:
            # test mode
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, training=False, debug=config.DEBUG)


        # # dataloader
        # if self.config.MODE == 1 or self.config.MODE == 3:
        #     self.train_loader = DataLoader(
        #         dataset=self.train_dataset,
        #         batch_size=self.config.BATCH_SIZE,
        #         num_workers=4,
        #         drop_last=True,
        #         shuffle=True
        #     )
        #     self.val_loader = DataLoader(
        #         dataset=self.val_dataset,
        #         batch_size=self.config.BATCH_SIZE,
        #         drop_last=True,
        #         shuffle=True
        #     )
        # else:
        #     self.test_loader = DataLoader(
        #         dataset=self.test_dataset,
        #         batch_size=1,
        #     )


        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        create_dir(self.samples_path)
        create_dir(self.results_path)

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.txt')

    @property
    def model_name(self):
        return self.__class__.__name__

    def load(self):
        self.inpaint_model.load()

    def save(self, niter=None):
        self.inpaint_model.save(niter)
        

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def sample(self, it=None):
        raise NotImplementedError


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item) for item in logs]))

    def cuda(self, data):
        return {key: item.to(self.config.DEVICE) for key, item in data.items()}

    def postprocess(self, img, to_byte=True):
        img = img.permute(0, 2, 3, 1)
        if to_byte:
            # [0, 1] => [0, 255]
            img = img * 255.0
            img = img.byte()
        return img

    def print(self):
        # print model to log_file
        saved_stdout = sys.stdout
        with open(self.log_file, "a") as f:
            sys.stdout = f
            print(self.inpaint_model)
        sys.stdout = saved_stdout


class DeepFeatureGuided(BaseInpainting):
    def __init__(self, config):
        super(DeepFeatureGuided, self).__init__(config)

        self.inpaint_model = InpaintModel(config).to(config.DEVICE)


        self.psnr = PSNR(1.).to(config.DEVICE).to(config.DEVICE)
        self.ssim = SSIM(window_size=config.SSIM_WINSIZE, size_average=True).to(config.DEVICE)
        

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        # metrics when bs is 1
        if train_loader.batch_size == 1:
            metrics_path = os.path.join(self.results_path, "metrics_train.txt")
            psnr_dict = {i:AverageMeta() for i in range(6)}
            ssim_dict = {i:AverageMeta() for i in range(6)}
            mae_dict = {i:AverageMeta() for i in range(6)}


        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for data in train_loader:
                self.inpaint_model.train()


                data = self.cuda(data)

                # train
                outputs, gen_loss, dis_loss, cls_loss, logs = self.inpaint_model.process(data)
                outputs_merged = (outputs * data['mask']) + (data['image'] * (1 - data['mask']))

                # metrics
                psnr = self.psnr(data['image'], outputs_merged)
                ssim = self.ssim(data['image'], outputs_merged)
                mae = (torch.sum(torch.abs(data['image'] - outputs_merged)) / torch.sum(data['image'])).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('ssim', ssim.item()))
                logs.append(('mae', mae.item()))

                # metrics when bs is 1
                if train_loader.batch_size == 1:
                    mask = data['mask']
                    mask_ratio = mask.gt(0).sum().item() / (mask.nelement())
                    psnr_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(psnr.item())
                    ssim_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(ssim.item())
                    mae_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(mae.item())


                # backward
                self.inpaint_model.backward(gen_loss, dis_loss, cls_loss)
                iteration = self.inpaint_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(data['image']), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample(iteration)

                # metrics when bs is 1
                if train_loader.batch_size == 1 and iteration % self.config.METRIC_INTERVAL == 0:
                    print_fun('='*20 + ' METRICS %07d' %self.inpaint_model.iteration + '='*20, log_file=metrics_path)
                    print_fun("[0.0-0.1) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[0].value(), ssim_dict[0].value(), mae_dict[0].value()), log_file=metrics_path)
                    print_fun("[0.1-0.2) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[1].value(), ssim_dict[1].value(), mae_dict[1].value()), log_file=metrics_path)
                    print_fun("[0.2-0.3) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[2].value(), ssim_dict[2].value(), mae_dict[2].value()), log_file=metrics_path)
                    print_fun("[0.3-0.4) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[3].value(), ssim_dict[3].value(), mae_dict[3].value()), log_file=metrics_path)
                    print_fun("[0.4-0.5) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[4].value(), ssim_dict[4].value(), mae_dict[4].value()), log_file=metrics_path)
                    print_fun("[0.5-0.6) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[5].value(), ssim_dict[5].value(), mae_dict[5].value()), log_file=metrics_path)


                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()
                    print('\nend eval...\n')

                # test model at checkpoints
                if self.config.TEST_INTERVAL and iteration % self.config.TEST_INTERVAL == 0:
                    print('\nstart testing...\n')
                    self.test(iteration)
                    print('\nend testing...\n')

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
                if self.config.EXTRA_SAVE_INTERVAL and iteration % self.config.EXTRA_SAVE_INTERVAL == 0:
                    self.save("iter%d"%iteration)

        print('\nEnd training....')


    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)

        # metrics when bs is 1
        if self.config.RUN_METRIC:
            metrics_path = os.path.join(self.results_path, "metrics_eval_iter%07d.txt" %self.inpaint_model.iteration)
            psnr_dict = {i:AverageMeta() for i in range(6)}
            ssim_dict = {i:AverageMeta() for i in range(6)}
            mae_dict = {i:AverageMeta() for i in range(6)}

        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        with torch.no_grad():
            for data in val_loader:
                iteration += 1

                data = self.cuda(data)

                # eval
                outputs, _, _, _, logs = self.inpaint_model.process(data)
                outputs_merged = (outputs * data['mask']) + (data['image'] * (1 - data['mask']))

                # metrics
                psnr = self.psnr(data['image'], outputs_merged)
                ssim = self.ssim(data['image'], outputs_merged)
                mae = (torch.sum(torch.abs(data['image'] - outputs_merged)) / torch.sum(data['image'])).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('ssim', ssim.item()))
                logs.append(('mae', mae.item()))

                # metrics when bs is 1
                if self.config.RUN_METRIC:
                    mask = data['mask']
                    mask_ratio = mask.gt(0).sum().item() / (mask.nelement())
                    psnr_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(psnr.item())
                    ssim_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(ssim.item())
                    mae_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(mae.item())

                logs = [("it", iteration), ] + logs
                progbar.add(len(data['image']), values=logs)


            # metrics when bs is 1
            if self.config.RUN_METRIC:
                print_fun('='*20 + ' METRICS ' + '='*20, log_file=metrics_path)
                print_fun("[0.0-0.1) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[0].value(), ssim_dict[0].value(), mae_dict[0].value()), log_file=metrics_path)
                print_fun("[0.1-0.2) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[1].value(), ssim_dict[1].value(), mae_dict[1].value()), log_file=metrics_path)
                print_fun("[0.2-0.3) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[2].value(), ssim_dict[2].value(), mae_dict[2].value()), log_file=metrics_path)
                print_fun("[0.3-0.4) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[3].value(), ssim_dict[3].value(), mae_dict[3].value()), log_file=metrics_path)
                print_fun("[0.4-0.5) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[4].value(), ssim_dict[4].value(), mae_dict[4].value()), log_file=metrics_path)
                print_fun("[0.5-0.6) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[5].value(), ssim_dict[5].value(), mae_dict[5].value()), log_file=metrics_path)


    def test(self, it=None):
        self.inpaint_model.test()

        sub_path = 'iter_%s'%(str(it)) if it is not None else self.model_name
        outputs_path = os.path.join(self.results_path, sub_path)
        masked_path = os.path.join(self.results_path, 'masked_images')
        create_dir(outputs_path)
        create_dir(masked_path)


        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        if self.config.RUN_METRIC:
            metrics_path = os.path.join(self.results_path, "metrics_test_iter%07d.txt"%it if it is not None else "metrics_%s.txt"%self.model_name)
            psnr_dict = {i:AverageMeta() for i in range(6)}
            ssim_dict = {i:AverageMeta() for i in range(6)}
            mae_dict = {i:AverageMeta() for i in range(6)}

        index = 0
        with torch.no_grad():
            for data in test_loader:
                name = self.test_dataset.load_name(index)
                save_name = name[:-4]+self.config.SAVE_EXT
                
                data = self.cuda(data)

                index += 1

                outputs = self.inpaint_model.process(data)
                outputs_merged = (outputs * data['mask']) + (data['image'] * (1 - data['mask']))


                if self.config.RUN_METRIC:
                    # metrics
                    psnr = self.psnr(data['image'], outputs_merged)
                    ssim = self.ssim(data['image'], outputs_merged)
                    mae = (torch.sum(torch.abs(data['image'] - outputs_merged)) / torch.sum(data['image'])).float()

                    mask = data['mask']
                    mask_ratio = mask.gt(0).sum().item() / (mask.nelement())
                    psnr_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(psnr.item())
                    ssim_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(ssim.item())
                    mae_dict[mask_ratio // 0.1 if mask_ratio // 0.1 < 6 else 5].update(mae.item())


                output = self.postprocess(outputs_merged)[0]
                path = os.path.join(outputs_path, save_name)
                print(index, name)

                # imsave(output, path[:-4]+'.jpg')
                # imsave(output, path[:-4]+'.png')
                # imsave(output, path[:-4]+'.bmp')
                imsave(output, path)

                if not os.path.exists(os.path.join(masked_path, save_name)):
                    masked = self.postprocess(data['image'] * (1 - data['mask']) + data['mask'])[0]

                    imsave(masked, os.path.join(masked_path, save_name))


            if self.config.RUN_METRIC:
                print_fun('='*20 + ' METRICS ' + '='*20, log_file=metrics_path)
                print_fun("[0.0-0.1) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[0].value(), ssim_dict[0].value(), mae_dict[0].value()), log_file=metrics_path)
                print_fun("[0.1-0.2) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[1].value(), ssim_dict[1].value(), mae_dict[1].value()), log_file=metrics_path)
                print_fun("[0.2-0.3) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[2].value(), ssim_dict[2].value(), mae_dict[2].value()), log_file=metrics_path)
                print_fun("[0.3-0.4) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[3].value(), ssim_dict[3].value(), mae_dict[3].value()), log_file=metrics_path)
                print_fun("[0.4-0.5) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[4].value(), ssim_dict[4].value(), mae_dict[4].value()), log_file=metrics_path)
                print_fun("[0.5-0.6) psnr: %.4f ssim: %.4f mae:%.4f" %(psnr_dict[5].value(), ssim_dict[5].value(), mae_dict[5].value()), log_file=metrics_path)


            if self.config.RUN_METRIC and self.config.RUN_METRIC_SCRIPTS:
                try:
                    from scripts.metrics import calculate
                    calculate(self.test_dataset.image_data, outputs_path, outputs_path, metrics_path,
                                image_size=self.config.INPUT_SIZE, ssim_winsize=self.config.SSIM_WINSIZE,
                                gray_mode=self.config.GRAY_MODE, debug_mode=False, ext=self.config.SAVE_EXT)

                    # os.system("python scripts/metrics.py --data-path %s --output-path %s --image-size %d --log-file %s" \
                    #             %(self.config.TEST_FLIST, self.results_path, self.config.INPUT_SIZE, os.path.join(self.results_path, "metrics.txt")))
                except:
                    print("\nCalculating metrics failed... Please run the metric.py script manully!")

            
            if self.html:
                html_title = 'results_it%s'%(str(it)) if it is not None else 'results'
                html_fname = os.path.join(self.results_path, "%s.html" %html_title)
                html = HTML(html_title)
                flist = self.test_dataset.image_data
                path_ext_dict = {
                    'masked_images': ['masked_images', self.config.SAVE_EXT],
                    sub_path: [sub_path, self.config.SAVE_EXT],
                }
                html.compare(flist, **path_ext_dict)
                html.save(html_fname)

        print('\nEnd test....')


    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.test()

        with torch.no_grad():

            data = next(self.sample_iterator)
            
            data = self.cuda(data)

            iteration = self.inpaint_model.iteration
            inputs = (data['image'] * (1 - data['mask'])) + data['mask']
            outputs = self.inpaint_model.process(data)
            outputs_merged = (outputs * data['mask']) + (data['image'] * (1 - data['mask']))

            if it is not None:
                iteration = it

            image_per_row = 2
            if self.config.SAMPLE_SIZE <= 6:
                image_per_row = 1

            sample_image = stitch_images(
                self.postprocess(data['image']),
                self.postprocess(inputs),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row = image_per_row
            )


            path = os.path.join(self.samples_path, self.model_name)
            name = os.path.join(path, str(iteration).zfill(7) + ".png")
            create_dir(path)
            print('\nsaving sample ' + name)
            sample_image.save(name)
