import copy
import sys

import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from lib.MDI_Net import MDI_Net
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn as nn

# ======================================================================
# # import accelerate
# from accelerate import Accelerator
# from accelerate.utils import set_seed

# ======================================================================

early_stop__eps = 1e-4  # 早停的指标阈值  1e-3 表示0.001
early_stop_patience = 15  # 早停的epoch阈值
threshold_lr = 1e-6  # 早停的学习率阈值


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    loss_list = []  # 创建一个空列表，用于存储每个批次的损失值。
    size_rates = [0.75, 1, 1.25]  # 表示图像将被缩放为原始尺寸的0.75、1（原始尺寸）和1.25倍。
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)  # 根据缩放比例调整后的图像尺寸。
            # 如果 rate 不等于 1，使用 F.interpolate 对图像和标签进行尺寸调整。
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear')
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear')
            # ---- forward ----
            P1 = model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss = loss_P1
            # ---- backward ----
            loss.backward()
            # 使用 clip_gradient(optimizer, opt.clip) 进行梯度裁剪，防止梯度爆炸。
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            # 如果 rate 等于 1（即原始尺寸），则更新 loss_P2_record。
            if rate == 1:
                loss_P2_record.update(loss_P1.data, opt.batchsize)
                loss_list.append(loss_P2_record.show())
        # ---- train visualization ----
        # 可视化训练进度：每20个步骤或在最后一个步骤，打印当前的时间、epoch、步骤数和当前的损失。
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                         ' lateral-5: {:0.4f}]'.
                         format(datetime.now(), epoch, opt.epoch, i, total_step,
                                loss_P2_record.show()))
    mean_loss = np.mean([l.cpu().numpy() for l in loss_list])
    print('{} Epoch [{:03d}/{:03d}]  '
          ' Train_mean_loss: {:0.4f}'.
          format(datetime.now(), epoch, opt.epoch,
                 mean_loss))
    logging.info('{} Epoch [{:03d}/{:03d}] '
                 ' Train_mean_loss: {:0.4f}]'.
                 format(datetime.now(), epoch, opt.epoch,
                        mean_loss))

    # if epoch % 5 == 0:
    #     save_path = (opt.train_save)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     torch.save(model.state_dict(), save_path + str(epoch) + 'Kvasir.pth' )
    global dict_plot


def test(test_loader, model, optimizer, epoch):
    model.eval()
    # 创建一个空列表，用于存储每个批次的损失值。
    loss_list = []
    #     save_path = (opt.train_save)
    loss_P2_record = AvgMeter()
    dice_list = []
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
            loss_P2_record.update(loss_P1.data, opt.batchsize)
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

    log_info = f'{datetime.now()}  val epoch: {epoch}, loss: {mean_loss:.4f}, miou: {miou:4f}, f1_or_dsc: {f1_or_dsc:4f}, accuracy: {accuracy:4f}, \
    specificity: {specificity:4f}, sensitivity: {sensitivity:4f}, confusion_matrix: {confusion}'

    print(log_info)
    logging.info(log_info)
    # print('mean_loss',mean_loss)
    return f1_or_dsc


# 初始化变量
best_dice = second_dice = third_dice = 0
best_epoch = second_epoch = third_epoch = 0
best_model = second_model = third_model = None
counter = 0


def save_model(model_state_dict, epoch, dice, rank):
    filename = f"rank_{rank}.pth"
    save_path = os.path.join(opt.train_save, filename)
    torch.save(copy.deepcopy(model_state_dict), save_path)  # 使用deepcopy确保独立复制
    print(f"Model epoch_{epoch}, dice_{dice} saved to {save_path}")

def train_and_evaluate(model, epochs=50):
    global best_dice, second_dice, third_dice, best_epoch, second_epoch, third_epoch, best_model, second_model, third_model
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch} started at {datetime.now()}")
        train(train_loader, model, optimizer, epoch, opt.test_path)
        dice = test(test_loader, model, optimizer, epoch)

        # 更新最好的三个模型
        if dice > best_dice:
            third_dice, third_epoch, third_model = second_dice, second_epoch, second_model
            second_dice, second_epoch, second_model = best_dice, best_epoch, best_model
            best_dice, best_epoch, best_model = dice, epoch, copy.deepcopy(model.state_dict())
        elif dice > second_dice:
            third_dice, third_epoch, third_model = second_dice, second_epoch, second_model
            second_dice, second_epoch, second_model = dice, epoch, copy.deepcopy(model.state_dict())
        elif dice > third_dice:
            third_dice, third_epoch, third_model = dice, epoch, copy.deepcopy(model.state_dict())

        # 保存当前最好的三个模型
        save_model(best_model, best_epoch, best_dice, 1)
        save_model(second_model, second_epoch, second_dice, 2)
        save_model(third_model, third_epoch, third_dice, 3)

        # 保存当前最好的三个模型
        save_model(best_model, best_epoch, best_dice, 1)
        save_model(second_model, second_epoch, second_dice, 2)
        save_model(third_model, third_epoch, third_dice, 3)

        if dice < best_dice:
            counter += 1
            log_info = f'val_dice is already {counter} times not increase'
            print(log_info)
            logging.info(log_info)
        else:
            counter = 0
        if counter > 100:
            logging.info('\t early_stopping!')
            break

        # 每10轮输出一次当前最好的三个模型的信息
        if epoch % 10 == 0:
            log_info1 = f"Current best dice: {best_dice} at epoch {best_epoch}"
            log_info2 = f"Second best dice: {second_dice} at epoch {second_epoch}"
            log_info3 = f"Third best dice: {third_dice} at epoch {third_epoch}"
            print(log_info1)
            print(log_info2)
            print(log_info3)
            logging.info(log_info1)
            logging.info(log_info2)
            logging.info(log_info3)

    # 训练结束后，加载最好的模型参数
    model.load_state_dict(best_model)
    print("Loaded best model parameters")


if __name__ == '__main__':
    max_val_dice=0
    start_epoch = 1
    min_epoch = 1
    counter = 0

    dict_plot = {'CVC-ClinicDB': [], 'ISIC2018': [], 'Kvasir': [], 'BUSI': [],
                 'test': []}
    name = ['CVC-ClinicDB', 'ISIC2018', 'Kvasir', 'BUSI', 'test']
    ##################model_name#############################
    model_name = 'MDI_Net'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=1000, help='epoch number')

    parser.add_argument('--gpu_id', type=int,
                        default=0, help='epoch number')

    parser.add_argument('--dataset', type=str,
                        default='ISIC2018', help='choosing dataset ISIC2018 or Polyp')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=224, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='/home/ta/datasets/ISIC2018/Train_Folder',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/home/ta/datasets/ISIC2018/Val_Folder',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='/home/ta/Project/MDI-Net-main/result/ISIC2018/' + model_name + '/')

    opt = parser.parse_args()
    logging.basicConfig(filename='/home/ta/Project/MDI-Net-main/log/ISIC2018/MDI_Net.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    # ---- build models ----

    torch.cuda.set_device(opt.gpu_id)  # set your gpu device
    model = MDI_Net().cuda()

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)

    # --------- dataset----------------------
    image_root = '{}/img/'.format(opt.train_path)
    gt_root = '{}/labelcol/'.format(opt.train_path)
    print(opt.augmentation)
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=False)
    total_step = len(train_loader)

    image_test = '{}/img/'.format(opt.test_path)
    gt_test = '{}/labelcol/'.format(opt.test_path)
    print(opt.augmentation)
    test_loader = get_loader(image_test, gt_test, batchsize=1, trainsize=opt.trainsize,
                             augmentation=False)

    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename_best = os.path.join(save_path, 'best.pth')
    filename_latest = os.path.join(save_path, 'latest.pth')
    print("#" * 20, "Start Training", "#" * 20)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    train_and_evaluate(model, epochs=opt.epoch)
