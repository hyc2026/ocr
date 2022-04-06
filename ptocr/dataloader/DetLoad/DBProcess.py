# -*- coding:utf-8 _*-
"""
@graduation project
@School of Computer Science and Engineering, Beihang University
@author  : hyc
@File    : DBProcess.py
@contact : hyc2026@yeah.net
"""
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from .MakeBorderMap import MakeBorderMap
from .transform_img import Random_Augment
from .MakeSegMap import MakeSegMap
import torchvision.transforms as transforms
from ptocr.utils.util_function import resize_image

class DBProcessTrain(data.Dataset):
    def __init__(self,config):
        super(DBProcessTrain,self).__init__()
        self.crop_shape = config['base']['crop_shape']
        self.MBM = MakeBorderMap()
        self.TSM = Random_Augment(self.crop_shape)
        self.MSM = MakeSegMap(shrink_ratio = config['base']['shrink_ratio'])
        img_list, label_list = self.get_base_information(config['trainload']['train_file'])
        self.img_list = img_list
        self.label_list = label_list
        
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_bboxes(self, gt_path):
        polys = []
        tags = []
        with open(gt_path, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
                gt = json.loads(line)
                if "#" in gt["transcription"]:
                    tags.append(True)
                else:
                    tags.append(False)
                # box = [int(gt[i]) for i in range(len(gt)//2*2)]
                box = [point for point in gt["points"]]
                polys.append(box)
        return np.array(polys), tags

    def get_base_information(self, train_txt_file):
        label_list = []
        img_list = []
        with open(train_txt_file, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                img_list.append(line[0])
                result = self.get_bboxes(line[1])
                label_list.append(result)
        return img_list, label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        
        img = Image.open(self.img_list[index]).convert('RGB')
        img = np.array(img)[:,:,::-1]
        
        polys, dontcare = self.label_list[index]
        # img 720 * 1280 * 3 ; self.crop_shape[0] 640
        img, polys = self.TSM.random_scale(img, polys, self.crop_shape[0])
        img, polys = self.TSM.random_rotate(img, polys)
        img, polys = self.TSM.random_flip(img, polys)
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        img, polys, dontcare = self.TSM.random_crop_db(img, polys, dontcare)
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        img, gt, gt_mask = self.MSM.process(img, polys, dontcare)
        img, thresh_map, thresh_mask = self.MBM.process(img, polys, dontcare)

        # 把数组转化成PIL图片
        img = Image.fromarray(img).convert('RGB')
        # 调整亮度brightness，对比度contrast，饱和度saturation和色相hue
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        # plt.imshow(img)
        # plt.show()

        # 把(0, 255)映射到(-1, 1)
        img = self.TSM.normalize_img(img)

        gt = torch.from_numpy(gt).float()
        gt_mask = torch.from_numpy(gt_mask).float()
        thresh_map = torch.from_numpy(thresh_map).float()
        thresh_mask = torch.from_numpy(thresh_mask).float()

        return img, gt, gt_mask, thresh_map, thresh_mask


class DBProcessTest(data.Dataset):
    def __init__(self,config):
        super(DBProcessTest,self).__init__()
        self.img_list = self.get_img_files(config['testload']['test_file'])
        self.TSM = Random_Augment(config['base']['crop_shape'])
        self.test_size = config['testload']['test_size']
        self.config =config

    def get_img_files(self, test_txt_file):
        img_list = []
        with open(test_txt_file, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip('\n')
                img_list.append(line)
        return img_list
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        ori_img = cv2.imread(self.img_list[index])
        img = resize_image(ori_img, self.config['base']['algorithm'], self.test_size, stride=self.config['testload']['stride'])
        img = Image.fromarray(img).convert('RGB')
        img = self.TSM.normalize_img(img)
        return img, ori_img


if __name__ == "__main__":
    config={'base': {'gpu_id': '2', 'algorithm': 'DB', 'pretrained': True, 'in_channels': [256, 512, 1024, 2048], 'inner_channels': 256, 'k': 50, 'adaptive': True, 'crop_shape': [640, 640], 'shrink_ratio': 0.4, 'n_epoch': 1200, 'start_val': 600, 'show_step': 20, 'checkpoints': './checkpoint', 'save_epoch': 100, 'restore': False, 'restore_file': './DB.pth.tar'}, 'backbone': {'function': 'ptocr.model.backbone.det_resnet,resnet50'}, 'head': {'function': 'ptocr.model.head.det_DBHead,DB_Head'}, 'segout': {'function': 'ptocr.model.segout.det_DB_segout,SegDetector'}, 'architectures': {'model_function': 'ptocr.model.architectures.det_model,DetModel', 'loss_function': 'ptocr.model.architectures.det_model,DetLoss'}, 'loss': {'function': 'ptocr.model.loss.db_loss,DBLoss', 'l1_scale': 10, 'bce_scale': 1}, 'optimizer': {'function': 'ptocr.optimizer,SGDDecay', 'base_lr': 0.002, 'momentum': 0.99, 'weight_decay': 0.0005}, 'optimizer_decay': {'function': 'ptocr.optimizer,adjust_learning_rate_poly', 'factor': 0.9}, 'trainload': {'function': 'ptocr.dataloader.DetLoad.DBProcess,DBProcessTrain', 'train_file': './dataset/icdar2015/dec_train_list.txt', 'num_workers': 10, 'batch_size': 8}, 'testload': {'function': 'ptocr.dataloader.DetLoad.DBProcess,DBProcessTest', 'test_file': './dataset/icdar2015/dec_test_list.txt', 'test_gt_path': './dataset/icdar2015/dec_test_gt/', 'test_size': 736, 'stride': 32, 'num_workers': 5, 'batch_size': 4}, 'postprocess': {'function': 'ptocr.postprocess.DBpostprocess,DBPostProcess', 'is_poly': False, 'thresh': 0.5, 'box_thresh': 0.6, 'max_candidates': 1000, 'unclip_ratio': 2, 'min_size': 3}, 'infer': {'model_path': './checkpoint/ag_DB_bb_resnet50_he_DB_Head_bs_8_ep_1200_bk/DB_best.pth.tar', 'path': './dataset/icdar2015/test_image', 'save_path': './result'}}
    pt = DBProcessTrain(config)
    # print(pt.img_list)
    # print(pt.label_list)
    '''
    [(array([[537, 140, 615, 143, 613, 171, 535, 167],
       [572, 165, 642, 168, 643, 188, 573, 185],
       [675, 212, 716, 213, 717, 231, 676, 230],
       [691, 246, 702, 246, 703, 260, 691, 260]]), [False, True, False, True])]
    '''
    pt[0]
