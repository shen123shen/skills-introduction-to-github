from random import random
from PIL import Image
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from lib.MDI_Net import MDI_Net
from utils.dataloader import get_loader, test_dataset,get_test_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn as nn

# 设置随机种子以确保结果的可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def test(test_loader, model, optimizer, epoch):
    model.eval()
    # 创建一个空列表，用于存储每个批次的损失值。
    loss_list = []
    loss_P2_record = AvgMeter()
    gts_list = []
    P1_list = []
    with torch.no_grad():
        for i, pack in enumerate(tqdm(test_loader), start=1):

            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- forward ----
            P1 = model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss = loss_P1

            # ---- recording loss ----
            loss_P2_record.update(loss_P1.data, 1)
            loss_list.append(loss_P2_record.show())
            # ----------添加P1,gts---------
            gts_list.append(gts.squeeze(1).cpu().detach().numpy())  # 将真实标签添加到 gts 列表中。
            if type(P1) is tuple:  # 检查模型输出是否为元组，如果是，则取第一个元素作为输出。
                P1 = P1[0]
            P1 = P1.squeeze(1).cpu().detach().numpy()  # 将输出转换为NumPy数组。
            P1_list.append(P1)  # 将预测结果添加到 preds 列表中。


    mean_loss = np.mean([l.cpu().numpy() for l in loss_list])
    preds = np.array(P1_list).reshape(-1)
    gts = np.array(gts_list).reshape(-1)
    # 使用 np.where 函数将预测结果 preds 和真实标签 gts 二值化。预测结果大于或等于阈值 config.threshold 时被视为1（正类）否则被视为0（负类）
    y_pre = np.where(preds >= 0.5, 1, 0)
    # 同样，真实标签大于或等于0.5时被视为1，否则被视为0。
    y_true = np.where(gts >= 0.5, 1, 0)
    # 使用 confusion_matrix 函数计算二值化后的预测结果和真实标签之间的混淆矩阵。
    confusion = confusion_matrix(y_true, y_pre)

    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    # 正确预测的样本数占总样本数的比例。
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    # sensitivity（敏感性/召回率）：正确预测为正类的样本数占所有实际正类样本数的比例。
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    # specificity（特异性）：正确预测为负类的样本数占所有实际负类样本数的比例。
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    # f1_or_dsc（F1分数或Dice系数）：综合考虑预测的精确性和召回率，是模型性能的一个指标。
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    # miou（平均交并比）：是模型预测的正类和实际正类之间的交并比的平均值。
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    log_info = f'{datetime.now()}  Test epoch: {epoch}, loss: {mean_loss:.4f}, miou: {miou:4f}, f1_or_dsc: {f1_or_dsc:4f}, accuracy: {accuracy:4f}, \
    specificity: {specificity:4f}, sensitivity: {sensitivity:4f}, confusion_matrix: {confusion}'

    print(log_info)
    logging.info(log_info)
    # print('mean_loss',mean_loss)
    return mean_loss

if __name__ == '__main__':
    #set_seed(42)  # Set the random seed for reproducibility
    model_name = 'MDI_Net'
    parser = argparse.ArgumentParser()

    parser.add_argument('--testsize', type=int,
                        default=224, help='testing size')

    parser.add_argument('--gpu_ids', type=int,
                        default=0, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--save_model', type=str,
                        default='/home/ta/Project/SwinPA-Sparse/result' + model_name + '/')

    parser.add_argument('--pth_path', type=str,
                        default='/home/ta/Project/MDI-Net-main/result/ISIC2018/MDI_Net/rank_1.pth')

    opt = parser.parse_args()

    logging.basicConfig(filename='/home/ta/Project/MDI-Net-main/test/log/ISIC2018/MDI_Net.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('opt:{}'.format(opt))

    torch.cuda.set_device(opt.gpu_ids)  # set your gpu device
    model = MDI_Net().cuda()

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)



    print('----------prepare dataset-----------')
    data_path = '/home/ta/datasets/ISIC2018/Test_Folder'
    image_test = '{}/img/'.format(data_path)
    gt_test = '{}/labelcol/'.format(data_path)
    test_loader = get_test_loader(image_test, gt_test, batchsize=1, trainsize=opt.testsize)
    total_step = len(test_loader)

    print(opt.augmentation)

    save_model = (opt.save_model)
    if not os.path.exists(save_model):
        os.makedirs(save_model)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    print('---------------test--------------')
    model.load_state_dict(torch.load(opt.pth_path,  map_location=torch.device('cpu')))
    for epoch in range(1, 2):
    	test(test_loader, model, optimizer, epoch)
