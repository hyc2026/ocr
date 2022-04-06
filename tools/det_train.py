# -*- coding:utf-8 _*-
"""
@graduation project
@School of Computer Science and Engineering, Beihang University
@author  : hyc
@File    : det_train.py
@contact : hyc2026@yeah.net
"""
import sys
sys.path.append('./')
import cv2
import torch
import os
import random
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import torch.utils.data
from ptocr.utils.util_function import create_module, create_loss_bin, set_seed, save_checkpoint, create_dir
from ptocr.utils.metrics import runningScore
from ptocr.utils.logger import Logger
from ptocr.utils.cal_iou_acc import cal_DB, cal_PAN_PSE
from tools.cal_rescall.script import cal_recall_precison_f1
from ptocr.utils.util_function import create_process_obj, merge_config, load_model

np.seterr(divide='ignore', invalid='ignore')
GLOBAL_WORKER_ID = None
GLOBAL_SEED = 123456

torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def ModelTrain(train_data_loader, t_model, t_criterion, model, criterion, optimizer, loss_bin, args, config, epoch):
    if config['base']['algorithm'] == 'DB' or config['base']['algorithm'] == 'SAST':
        running_metric_text = runningScore(2)
    else:
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)
    for batch_idx, data in enumerate(train_data_loader):
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        if data is None:
            continue
        # 使用pytorch训练模型时，定义的时候有forward，使用的时候直接用模型自己。
        pre_batch, gt_batch = model(data)  # model.forward(data)
            
        loss, metrics = criterion(pre_batch, gt_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in loss_bin.keys():
            if key in metrics.keys():
                loss_bin[key].loss_add(metrics[key].item())
            else:
                loss_bin[key].loss_add(loss.item())
        if config['base']['algorithm'] == 'DB':
            iou, acc = cal_DB(pre_batch['binary'], gt_batch['gt'], gt_batch['mask'], running_metric_text)
        else:
            iou, acc = cal_PAN_PSE(pre_batch['pre_kernel'], gt_batch['gt_kernel'], pre_batch['pre_text'], gt_batch['gt_text'],
                                   gt_batch['train_mask'], running_metric_text, running_metric_kernel)
        if batch_idx % config['base']['show_step'] == 0:
            log = '({}/{}/{}/{}) | '.format(epoch, config['base']['n_epoch'], batch_idx, len(train_data_loader))
            bin_keys = list(loss_bin.keys())

            for i in range(len(bin_keys)):
                log += bin_keys[i] + ':{:.4f}'.format(loss_bin[bin_keys[i]].loss_mean()) + ' | '

            log +=  'ACC:{:.4f}'.format(acc) + ' | '
            log +=  'IOU:{:.4f}'.format(iou) + ' | '
            log += 'lr:{:.8f}'.format(optimizer.param_groups[0]['lr'])
            # print(log)
            file = open("log3.txt", "a+")
            file.write(log)
            file.write('\n')
            file.close()
    loss_write = []
    for key in list(loss_bin.keys()):
        loss_write.append(loss_bin[key].loss_mean())
    loss_write.extend([acc, iou])
    return loss_write


def ModelEval(test_dataset, test_data_loader, model, imgprocess, checkpoints, config):
    bar = tqdm(total=len(test_data_loader))
    for batch_idx, (imgs, ori_imgs) in enumerate(test_data_loader):
        bar.update(1)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        with torch.no_grad():
            out = model(imgs)
        scales = []
        if isinstance(out,dict):
            img_num = out['f_score'].shape[0]
        else:
            img_num = out.shape[0]
        for i in range(img_num):
            scale = (ori_imgs[i].shape[1] * 1.0 / out.shape[3], ori_imgs[i].shape[0] * 1.0 / out.shape[2])
            scales.append(scale)
        out = create_process_obj(config['base']['algorithm'], out)
        bbox_batch, score_batch = imgprocess(out, scales)
            
        for i in range(len(bbox_batch)):
            bboxes = bbox_batch[i]
            img_show = ori_imgs[i].numpy().copy()
            idx = i + out.shape[0] * batch_idx
            image_name = test_dataset.img_list[idx].split('/')[-1].split('.')[0]
            with open(os.path.join(checkpoints, 'val', 'res_txt', 'res_' + image_name + '.txt'), 'w+',
                      encoding='utf-8') as fid_res:
                for bbox in bboxes:
                    bbox = bbox.reshape(-1, 2).astype(np.int)
                    img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
                    bbox_str = [str(x) for x in bbox.reshape(-1)]
                    bbox_str = ','.join(bbox_str) + '\n'
                    fid_res.write(bbox_str)
            # cv2.imwrite(os.path.join(checkpoints, 'val', 'res_img', image_name + '.jpg'), img_show)
    bar.close()
    result_dict = cal_recall_precison_f1(config['testload']['test_gt_path'],os.path.join(checkpoints, 'val', 'res_txt'))
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']


def TrainValProgram(args):
    config = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config = merge_config(config, args)  # args覆盖config里参数
    os.environ["CUDA_VISIBLE_DEVICES"] = config['base']['gpu_id']
    create_dir(config['base']['checkpoints'])
    checkpoints = os.path.join(config['base']['checkpoints'],
                               "my_ag_%s_bb_%s_he_%s_bs_%d_ep_%d_%s" % (config['base']['algorithm'],
                                                      config['backbone']['function'].split(',')[-1],
                                                      config['head']['function'].split(',')[-1],
                                                      config['trainload']['batch_size'],
                                                      config['base']['n_epoch'],
                                                      args.log_str))
    create_dir(checkpoints)
    model = create_module(config['architectures']['model_function'])(config)
    criterion = create_module(config['architectures']['loss_function'])(config)
    train_dataset = create_module(config['trainload']['function'])(config)
    test_dataset = create_module(config['testload']['function'])(config)
    # The optimizer instructs the parameters how to update their values
    # knowing the gradient with a function named step()
    optimizer = create_module(config['optimizer']['function'])(config, model)
    optimizer_decay = create_module(config['optimizer_decay']['function'])
    img_process = create_module(config['postprocess']['function'])(config)  # 用于验证
   
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['trainload']['batch_size'],
        # 加载数据时是否打乱
        shuffle=True,
        # 加载数据的子进程数
        num_workers=config['trainload']['num_workers'],
        # 在每一个worker子进程数据加载前以worker_id作为参数调用worker_init_fn
        worker_init_fn=worker_init_fn,
        # 如果数据集大小不能被批大小整除，则删除最后一个不完整的批
        drop_last=True,
        # 数据加载器将在返回张量之前将其复制到CUDA固定内存中
        pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['testload']['batch_size'],
        shuffle=False,
        num_workers=config['testload']['num_workers'],
        drop_last=True,
        pin_memory=True)

    # 一个数组，每种算法的loss由哪些部分组成
    loss_bin = create_loss_bin(config['base']['algorithm'], False)

    if torch.cuda.is_available():
        if len(config['base']['gpu_id'].split(',')) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()
        
    start_epoch = 0
    rescall, precision, hmean = 0, 0, 0
    best_rescall, best_precision, best_hmean = 0, 0, 0

    if config['base']['restore']:
        print('Resuming from checkpoint.')
        assert os.path.isfile(config['base']['restore_file']), 'Error: no checkpoint file found!'
        checkpoint = torch.load(config['base']['restore_file'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_rescall = checkpoint['rescall']
        best_precision = checkpoint['precision']
        best_hmean = checkpoint['hmean']
        if not os.path.exists(os.path.join(checkpoints, 'log.txt')):
            open(os.path.join(checkpoints, 'log.txt'), "a+")
        log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['base']['algorithm'], resume=True)
    else:
        print('Training from scratch.')
        log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['base']['algorithm'])
        title = list(loss_bin.keys())
        title.extend(['piexl_acc','piexl_iou','t_rescall','t_precision','t_hmean','b_rescall','b_precision','b_hmean'])
        log_write.set_names(title)
        
    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    for epoch in range(start_epoch, config['base']['n_epoch']):
        model.train()  # 设置模型需要训练
        if args.t_config is not None:
            t_model.train()
        else:
            t_model = None
        distil_loss = None
        optimizer_decay(config, optimizer, epoch)  # 函数内部修改optimizer的lr
        loss_write = ModelTrain(train_data_loader, t_model, distil_loss, model, criterion, optimizer, loss_bin, args, config, epoch)

        if epoch >= config['base']['start_val']:
            create_dir(os.path.join(checkpoints, 'val'))
            create_dir(os.path.join(checkpoints, 'val', 'res_img'))
            create_dir(os.path.join(checkpoints, 'val', 'res_txt'))
            model.eval()
            rescall, precision, hmean = ModelEval(test_dataset, test_data_loader, model, img_process, checkpoints, config)
            print('rescall:', rescall, 'precision', precision, 'hmean', hmean)
            if hmean > best_hmean:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lr': config['optimizer']['base_lr'],
                    'optimizer': optimizer.state_dict(),
                    'hmean': hmean,
                    'rescall': rescall,
                    'precision': precision
                }, checkpoints, config['base']['algorithm'] + '_best' + '.pth.tar')
                best_hmean = hmean
                best_precision = precision
                best_rescall = rescall

        loss_write.extend([rescall, precision, hmean, best_rescall, best_precision, best_hmean])
        log_write.append(loss_write)
        for key in loss_bin.keys():
            loss_bin[key].loss_clear()
        if epoch % config['base']['save_epoch'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lr': config['optimizer']['base_lr'],
                'optimizer': optimizer.state_dict(),
                'hmean': 0,
                'rescall': 0,
                'precision': 0
             }, checkpoints, config['base']['algorithm']+'_'+str(epoch)+'.pth.tar')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--t_config', default=None, help='config file path')
    parser.add_argument('--t_model_path', default=None, help='teacher model path')
    parser.add_argument('--t_ratio', nargs='?', type=float, default=0.5)
    parser.add_argument('--log_str', help='log title')
    parser.add_argument('--sr_lr', nargs='?', type=float, default=None)
    
    parser.add_argument('--pruned_model_dict_path', help='config file path',default=None)
    parser.add_argument('--prune_type', type=str, help='prune type,total or backbone')
    parser.add_argument('--prune_model_path', help='model file path')
    
    parser.add_argument('--n_epoch', nargs='?', type=int, default=600)
    parser.add_argument('--start_epoch', nargs='?', type=int, default=None)
    parser.add_argument('--start_val', nargs='?', type=int, default=400)
    parser.add_argument('--base_lr', nargs='?', type=float, default=0.001)
    parser.add_argument('--gpu_id', help='config file path')

    if os.path.exists("log3.txt"):
        os.remove("log3.txt")

    args = parser.parse_args()
    TrainValProgram(args)
