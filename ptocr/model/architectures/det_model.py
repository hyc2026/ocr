"""
@graduation project
@School of Computer Science and Engineering, Beihang University
@author  : hyc
@File    : det_model.py
@contact : hyc2026@yeah.net
"""
import torch
import torch.nn as nn
from ptocr.utils.util_function import create_module


class DetModel(nn.Module):
    def __init__(self, config):
        super(DetModel, self).__init__()
        self.algorithm = config['base']['algorithm']

        self.backbone = create_module(config['backbone']['function'])(config['base']['pretrained'])
        self.head = create_module(config['head']['function'])(config['base']['in_channels'], config['base']['inner_channels'])

        if (config['base']['algorithm']) == 'DB':
            self.seg_out = create_module(config['segout']['function'])(config['base']['inner_channels'], config['base']['k'], config['base']['adaptive'])
        elif (config['base']['algorithm']) == 'PAN':
            self.seg_out = create_module(config['segout']['function'])(config['base']['inner_channels'], config['base']['classes'])
        elif (config['base']['algorithm']) == 'PSE':
            self.seg_out = create_module(config['segout']['function'])(config['base']['inner_channels'], config['base']['classes'])
        else:
            assert True == False, ('not support this algorithm !!!')

    def forward(self, data):
        if self.training:
            if self.algorithm == "DB":
                img, gt, gt_mask, thresh_map, thresh_mask = data
                if torch.cuda.is_available():
                    img, gt, gt_mask, thresh_map, thresh_mask = img.cuda(), gt.cuda(), gt_mask.cuda(), thresh_map.cuda(), thresh_mask.cuda()
                gt_batch = dict(gt=gt)
                gt_batch['mask'] = gt_mask
                gt_batch['thresh_map'] = thresh_map
                gt_batch['thresh_mask'] = thresh_mask

            elif self.algorithm == "PSE":
                img, gt_text, gt_kernels, train_mask = data
                if torch.cuda.is_available():
                    img, gt_text, gt_kernels, train_mask = img.cuda(), gt_text.cuda(), gt_kernels.cuda(), train_mask.cuda()
                gt_batch = dict(gt_text=gt_text)
                gt_batch['gt_kernel'] = gt_kernels
                gt_batch['train_mask'] = train_mask

            elif self.algorithm == "PAN":
                img, gt_text, gt_text_key, gt_kernel, gt_kernel_key, train_mask = data
                if torch.cuda.is_available():
                    img, gt_text, gt_text_key, gt_kernel, gt_kernel_key, train_mask = \
                        img.cuda(), gt_text.cuda(), gt_text_key.cuda(), gt_kernel.cuda(), gt_kernel_key.cuda(), train_mask.cuda()

                gt_batch = dict(gt_text=gt_text)
                gt_batch['gt_text_key'] = gt_text_key
                gt_batch['gt_kernel'] = gt_kernel
                gt_batch['gt_kernel_key'] = gt_kernel_key
                gt_batch['train_mask'] = train_mask

        else:
            img = data

        x = self.backbone(img)
        x = self.head(x)
        x = self.seg_out(x, img)

        if self.training:
            return x, gt_batch
        return x


class DetLoss(nn.Module):
    def __init__(self, config):
        super(DetLoss, self).__init__()
        self.algorithm = config['base']['algorithm']
        if (config['base']['algorithm']) == 'DB':
            self.loss = create_module(config['loss']['function'])(config['loss']['l1_scale'], config['loss']['bce_scale'])
        elif (config['base']['algorithm']) == 'PAN':
            self.loss = create_module(config['loss']['function'])(config['loss']['kernel_rate'], config['loss']['agg_dis_rate'])
        elif (config['base']['algorithm']) == 'PSE':
            self.loss = create_module(config['loss']['function'])(config['loss']['text_tatio'])
        else:
            assert True == False, ('not support this algorithm !!!')

    def forward(self, pre_batch, gt_batch):
        return self.loss(pre_batch, gt_batch)

