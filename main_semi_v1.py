from tqdm import tqdm
import network
import utils
import os
import random
import sys
import numpy as np
from itertools import cycle
from datetime import datetime

from torch.utils import data
from datasets import VOCSegmentation
# from utils import ext_transforms as et
from metrics import Metrics, DistMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from PIL import Image
import matplotlib.pyplot as plt


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        l_list = "subset_train_aug/train_aug_labeled_1-{}.txt".format(opts.labeled_ratio)
        unl_list = "subset_train_aug/train_aug_unlabeled_1-{}.txt".format(opts.labeled_ratio)
        l_train_dst = VOCSegmentation(root=opts.data_root, data_list=l_list, image_set='train',
                                      base_size=opts.base_size, crop_size=opts.crop_size, is_training=True)
        unl_train_dst = VOCSegmentation(root=opts.data_root, data_list=unl_list, image_set='train',
                                        base_size=opts.base_size, crop_size=opts.crop_size, is_training=True)
        train_dst = {'l': l_train_dst, 'unl': unl_train_dst}
        val_dst = VOCSegmentation(root=opts.data_root, data_list=None, image_set='val',
                                  base_size=opts.base_size, crop_size=opts.crop_size, is_training=False)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0
    total_itrs = len(loader)
    overlap = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            l_bs = images.shape[0]

            outputs = model(images, l_bs)

            # lam = np.random.beta(2, 2)
            # rand_index = torch.randperm(images.size()[0]).to(device)
            # bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            # images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            #
            # outputs1 = model(images, l_bs)

            # eval_metrics has already implemented DDP synchronized
            metrics.update(outputs, labels)

            preds = {}
            for num_member in range(0, opts.num_members):
                preds["final_" + str(num_member)] = outputs["final_" + str(num_member)].detach() \
                    .max(dim=1)[1].cpu().numpy()
            preds["final"] = outputs["final"].detach().max(dim=1)[1].cpu().numpy()
            # diff = abs(preds["final_1"] - preds["final_0"])
            # diff = np.where(diff > 0, 1, 0)
            # diff = np.mean(diff)
            # overlap.append(diff)
            targets = labels.cpu().numpy()

            # metrics.update(targets, preds)

            if i % 100 == 0 and opts.rank == 0:
                print("Val Itrs %d/%d" % (i, total_itrs))

            if opts.save_val_results and img_id < 50 and opts.rank == 0:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds["final"][i]
                    # mask = np.where(target > 200, 0, 1)
                    # pred = pred * mask
                    #
                    # pse_outputs_0 = F.softmax(outputs["final_0"], dim=1)
                    # pse_outputs_1 = F.softmax(outputs["final_1"], dim=1)
                    # pse_outputs_0 = outputs["final_0"]
                    # pse_outputs_1 = outputs["final_1"]
                    pse_outputs = outputs["final"]
                    # pse_outputs_0[:, :, bbx1:bbx2, bby1:bby2] = pse_outputs_0[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # pse_outputs_1[:, :, bbx1:bbx2, bby1:bby2] = pse_outputs_1[rand_index, :, bbx1:bbx2, bby1:bby2]

                    # pseudo_label_0, unc_weight_0 = pseudo_weight(pse_outputs_0)
                    # pseudo_label_1, unc_weight_1 = pseudo_weight(pse_outputs_1)
                    pseudo_label, unc_weight = pseudo_weight(pse_outputs)
                    # pseudo_label_0 = pseudo_label_0[i].cpu().numpy()
                    # pseudo_label_1 = pseudo_label_1[i].cpu().numpy()
                    pseudo_label = pseudo_label[i].cpu().numpy()
                    # unc_weight_0 = unc_weight_0[i].cpu().numpy()
                    # unc_weight_1 = unc_weight_1[i].cpu().numpy()
                    unc_weight = unc_weight[i].cpu().numpy()
                    # mask = np.where(target > 250, 0, 1)
                    fig = plt.figure()
                    plt.axis('off')
                    fig = plt.gcf()
                    fig.set_size_inches(3.2 / 6, 3.2 / 6)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.margins(0, 0)
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    plt.imshow(1 - unc_weight)
                    # plt.colorbar()
                    plt.savefig('results/%d_unc_weight_0.png' % img_id, transparent=True, dpi=600, pad_inches=0)
                    plt.close()

                    # fig = plt.figure()
                    # plt.axis('off')
                    # fig = plt.gcf()
                    # fig.set_size_inches(3.2 / 3, 3.2 / 3)
                    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    # plt.margins(0, 0)
                    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    # plt.imshow(1 - unc_weight_1)
                    # # plt.colorbar()
                    # plt.savefig('results/%d_unc_weight_1.png' % img_id, transparent=True, dpi=300, pad_inches=0)
                    # plt.close()

                    # pseudo_label_0 = pseudo_label_0[i].cpu().numpy().astype(np.uint8)
                    # pseudo_label_1 = pseudo_label_1[i].cpu().numpy().astype(np.uint8)
                    # pseudo_label_0 = loader.dataset.decode_target(pseudo_label_0[i]).astype(np.uint8)
                    # pseudo_label_1 = loader.dataset.decode_target(pseudo_label_1[i]).astype(np.uint8)
                    # Image.fromarray(unc_weight_0).save('results/%d_unc_weight_0.png' % img_id)
                    # Image.fromarray(unc_weight_1).save('results/%d_unc_weight_1.png' % img_id)
                    #
                    # target = np.where(target > 254, 0, target)
                    error = abs(target - pred)
                    error = np.where(error > 200, 0, error)
                    error = np.where(error > 0, 22, 0)
                    error = loader.dataset.decode_target(error).astype(np.uint8)
                    Image.fromarray(error).save('results/%d_error.png' % img_id)

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    # pseudo_label_0 = loader.dataset.decode_target(pseudo_label_0).astype(np.uint8)
                    # pseudo_label_1 = loader.dataset.decode_target(pseudo_label_1).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                    # Image.fromarray(pseudo_label_0).save('results/%d_pred1.png' % img_id)
                    # Image.fromarray(pseudo_label_1).save('results/%d_pred2.png' % img_id)

                    # fig = plt.figure()
                    # plt.imshow(image)
                    # plt.axis('off')
                    # plt.imshow(pred, alpha=0.7)
                    # ax = plt.gca()
                    # ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    img_id += 1
        # overlap_result = np.mean(overlap)
        # print(overlap_result)
        score = metrics.get_results()
    return score, ret_samples


def main(gpu, ngpus_per_node, opts):
    opts.rank = gpu + ngpus_per_node * opts.nodes
    if opts.dist:
        dist.init_process_group(backend='nccl', init_method=opts.dist_url, world_size=opts.world_size,
                                rank=opts.rank)

    current_time = datetime.now().strftime('%b%d_%H_%M')
    if gpu == 0:
        sys.stdout = Logger(f"./log_new/{opts.name}_{current_time}.log", stream=sys.stdout)
        sys.stderr = Logger(f"./log_new/{opts.name}_{current_time}.log", stream=sys.stderr)

    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Setup random seed
    random.seed(opts.random_seed)
    init_seeds(opts.random_seed + gpu)

    # Setup dataset and dataloader
    train_dst, val_dst = get_dataset(opts)

    if opts.dist:
        opts.batch_size = int(opts.batch_size / opts.world_size)
        opts.unl_batch_size = int(opts.unl_batch_size / opts.world_size)
        opts.val_batch_size = int(opts.val_batch_size / opts.world_size)

        l_train_sampler = data.distributed.DistributedSampler(train_dst["l"], shuffle=True)
        l_train_loader = data.DataLoader(train_dst["l"], batch_size=opts.batch_size, num_workers=2,
                                         drop_last=True, pin_memory=True, sampler=l_train_sampler)
        l_train_sampler_s = data.distributed.DistributedSampler(train_dst["l"], shuffle=True)
        l_train_loader_s = data.DataLoader(train_dst["l"], batch_size=opts.batch_size, num_workers=2,
                                           drop_last=True, pin_memory=True, sampler=l_train_sampler_s)
        ul_train_sampler = data.distributed.DistributedSampler(train_dst["unl"], shuffle=True)
        unl_train_loader = data.DataLoader(train_dst["unl"], batch_size=opts.unl_batch_size, num_workers=2,
                                           drop_last=True, pin_memory=True, sampler=ul_train_sampler)
        val_sampler = data.distributed.DistributedSampler(val_dst, shuffle=False)
        val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, num_workers=2,
                                     pin_memory=True, sampler=val_sampler)
    else:
        l_train_loader = data.DataLoader(train_dst["l"], batch_size=opts.batch_size, num_workers=1,
                                         drop_last=True, pin_memory=True, shuffle=True)
        # l_train_loader_s = data.DataLoader(train_dst["l"], batch_size=opts.batch_size, num_workers=1,
        #                                    drop_last=True, pin_memory=True, shuffle=True)
        unl_train_loader = data.DataLoader(train_dst["unl"], batch_size=opts.unl_batch_size, num_workers=1,
                                           drop_last=True, pin_memory=True, shuffle=True)
        val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, num_workers=1,
                                     pin_memory=True)
    print("Dataset: %s, Train labeled set: %d, Train unlabeled set: %d, Val set: %d" %
          (opts.dataset, len(train_dst["l"]), len(train_dst["unl"]), len(val_dst)))

    # Set up modeling (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
        network.convert_to_separable_conv(model.final)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    # utils.set_bn_momentum(model.list_conv, momentum=0.01)

    # Set up metrics
    # metrics = Metrics(opts.num_classes, opts.num_members)
    metrics = DistMetrics(opts.num_classes, ignore_index=255, num_members=opts.num_members)

    params = [{'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
              # {'params': model.list_conv.parameters(), 'lr': 0.1 * opts.lr},
              {'params': model.classifier.parameters(), 'lr': opts.lr},
              {'params': model.final.parameters(), 'lr': opts.lr}, ]

    # Set up optimizer
    optimizer = torch.optim.SGD(params=params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=modeling.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    niters_per_epoch = len(unl_train_loader)
    # niters_per_epoch = len(l_train_loader)
    total_itrs = opts.epoch * niters_per_epoch
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    criterion_unl = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    def save_ckpt(path):
        """ save current modeling
        """
        if opts.dist:
            torch.save({
                "cur_itrs": cur_itrs,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
            }, path)
        else:
            torch.save({
                "cur_itrs": cur_itrs,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
            }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    best_score = 0.0
    cur_itrs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        print("Model restored from %s" % opts.ckpt)
        print(checkpoint["cur_itrs"])
        del checkpoint
    print("[!] Retrain")
    model.to(device)
    if opts.dist:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], broadcast_buffers=False)

    # ==========   Train Loop   ==========#
    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        print(metrics.to_str(val_score))
        return

    interval_loss, interval_l_loss, interval_ul_loss = 0, 0, 0
    for cur_epochs in range(opts.epoch):
        # =====  Train  =====
        model.train()
        if opts.dist:
            l_train_sampler.set_epoch(cur_epochs)
            l_train_sampler_s.set_epoch(cur_epochs + 1)
            ul_train_sampler.set_epoch(cur_epochs)

        dataloader_s = iter(l_train_loader_s)
        dataloader = iter(zip(cycle(l_train_loader), unl_train_loader))

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        tbar = tqdm(range(niters_per_epoch), bar_format=bar_format)
        for i in tbar:
            optimizer.zero_grad()
            (images, labels), (unl_images, unl_labels) = next(dataloader)
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            unl_images = unl_images.to(device, dtype=torch.float32, non_blocking=True)
            unl_bs = unl_images.shape[0]
            l_bs = images.shape[0]
            # 有0.4的概率，输入不同，0.6的概率，输入相同
            if random.random() > 0.4:
                # 输入不同图片时从另一个dataloader中取
                try:
                    images_s, labels_s = next(dataloader_s)
                except StopIteration:
                    dataloader_s = iter(l_train_loader_s)
                    images_s, labels_s = next(dataloader_s)
                images_s = images_s.to(device, dtype=torch.float32, non_blocking=True)
                labels_s = labels_s.to(device, dtype=torch.long, non_blocking=True)

                # unsup
                with torch.no_grad():
                    t_unsup_pred = model(unl_images, unl_bs)
                    logits_0 = t_unsup_pred["final_0"].detach()
                    logits_1 = t_unsup_pred["final_1"].detach()
                # cut_mix
                lam = np.random.beta(2, 2)
                lam_1 = np.random.beta(2, 2)
                rand_index = torch.randperm(unl_images.size()[0]).to(device)
                rand_index_1 = torch.randperm(unl_images.size()[0]).to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(unl_images.size(), lam)
                bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(unl_images.size(), lam_1)
                # mix images
                unl_images_1 = unl_images.clone()
                unl_images[:, :, bbx1:bbx2, bby1:bby2] = unl_images[rand_index, :, bbx1:bbx2, bby1:bby2]
                unl_images_1[:, :, bbx1_1:bbx2_1, bby1_1:bby2_1] = unl_images_1[rand_index_1, :, bbx1_1:bbx2_1,
                                                                   bby1_1:bby2_1]
                logits_1[:, :, bbx1:bbx2, bby1:bby2] = logits_1[rand_index, :, bbx1:bbx2, bby1:bby2]
                logits_0[:, :, bbx1_1:bbx2_1, bby1_1:bby2_1] = logits_0[rand_index_1, :, bbx1_1:bbx2_1, bby1_1:bby2_1]
                pseudo_label, weight = pseudo_weight(logits_0)
                weight_sum = torch.sum(weight)
                pseudo_label_1, weight_1 = pseudo_weight(logits_1)
                weight_sum_1 = torch.sum(weight_1)

                unl_images_cat = torch.cat((unl_images, unl_images_1), dim=0)
                s_unsup_pred = model(unl_images_cat, unl_bs)
                ul_loss = torch.div(torch.sum(criterion_unl(s_unsup_pred["final_0"], pseudo_label_1) * weight_1),
                                    weight_sum_1) + \
                          torch.div(torch.sum(criterion_unl(s_unsup_pred["final_1"], pseudo_label) * weight),
                                    weight_sum)
                # ul_loss = criterion(s_unsup_pred["final_0"], pseudo_label_1) + \
                #           criterion(s_unsup_pred["final_1"], pseudo_label)

                # sup
                images_cat = torch.cat((images, images_s), dim=0)
                outputs = model(images_cat, l_bs)
                l_loss = criterion(outputs["final_0"], labels) + criterion(outputs["final_1"], labels_s)

            else:
                # unsup
                with torch.no_grad():
                    t_unsup_pred = model(unl_images, unl_bs)
                    logits_0 = t_unsup_pred["final_0"].detach()
                    logits_1 = t_unsup_pred["final_1"].detach()
                lam = np.random.beta(2, 2)
                rand_index = torch.randperm(unl_images.size()[0]).to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(unl_images.size(), lam)
                unl_images[:, :, bbx1:bbx2, bby1:bby2] = unl_images[rand_index, :, bbx1:bbx2, bby1:bby2]
                logits_0[:, :, bbx1:bbx2, bby1:bby2] = logits_0[rand_index, :, bbx1:bbx2, bby1:bby2]
                logits_1[:, :, bbx1:bbx2, bby1:bby2] = logits_1[rand_index, :, bbx1:bbx2, bby1:bby2]
                pseudo_label_0, weight_0 = pseudo_weight(logits_0)
                weight_sum_0 = torch.sum(weight_0)
                pseudo_label_1, weight_1 = pseudo_weight(logits_1)
                weight_sum_1 = torch.sum(weight_1)

                s_unsup_pred = model(unl_images, unl_bs)
                ul_loss = torch.div(torch.sum(criterion_unl(s_unsup_pred["final_0"], pseudo_label_1) * weight_1),
                                    weight_sum_1) + \
                          torch.div(torch.sum(criterion_unl(s_unsup_pred["final_1"], pseudo_label_0) * weight_0),
                                    weight_sum_0)
                # ul_loss = criterion(s_unsup_pred["final_0"], pseudo_label_1) + \
                #           criterion(s_unsup_pred["final_1"], pseudo_label_0)

                # sup
                outputs = model(images, l_bs)
                l_loss = criterion(outputs["final_0"], labels) + criterion(outputs["final_1"], labels)

            if opts.dist:
                dist.barrier()
                # ul_loss = reduce_mean(ul_loss, opts.world_size)
            # ul_loss = l_loss
            loss = l_loss + 1 * ul_loss
            loss.backward()
            optimizer.step()

            lr = scheduler.get_lr()[0]
            print_str = 'Epoch{}/{}'.format(cur_epochs, opts.epoch) \
                        + ' Iter{}/{}:'.format(i, niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % l_loss.item() \
                        + ' loss_un=%.2f' % ul_loss.item()
            tbar.set_description(print_str)

            if opts.dist:
                dist.barrier()
                loss = reduce_mean(loss, opts.world_size)
                l_loss = reduce_mean(l_loss, opts.world_size)
                ul_loss = reduce_mean(ul_loss, opts.world_size)
            np_loss = loss.detach().cpu().numpy()
            np_l_loss = l_loss.detach().cpu().numpy()
            np_ul_loss = ul_loss.detach().cpu().numpy()
            interval_loss += np_loss
            interval_l_loss += np_l_loss
            interval_ul_loss += np_ul_loss

            scheduler.step()
            cur_itrs += 1

        if gpu == 0:
            interval_loss = interval_loss / niters_per_epoch
            interval_l_loss = interval_l_loss / niters_per_epoch
            interval_ul_loss = interval_ul_loss / niters_per_epoch
            print("Epoch %d, Itrs %d/%d, Loss=%f, l_Loss=%f, ul_Loss=%f" %
                  (cur_epochs, cur_itrs, total_itrs, interval_loss, interval_l_loss, interval_ul_loss))
            interval_loss, interval_l_loss, interval_ul_loss = 0, 0, 0
        if cur_epochs > opts.epoch * 2 / 3:  # and cur_epochs % 2 == 0:
            print("validation...")
            model.eval()
            val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
            print(metrics.to_str(val_score))
            max_score = metrics.max_results(val_score)
            if max_score > best_score:  # save best modeling
                best_score = max_score
                if gpu == 0:
                    save_ckpt('checkpoints/%s_%s.pth' %
                              (opts.name, current_time))
        model.train()


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def init_seeds(seed, cuda_deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def pseudo_weight(pse_outputs):
    pse_outputs = F.softmax(pse_outputs, dim=1)
    pseudo_label = torch.max(pse_outputs, dim=1)[1].long()
    uncertainty = -1.0 * torch.sum(pse_outputs * torch.log(pse_outputs + 1e-8), dim=1)
    unc_flatten = torch.flatten(uncertainty, 1)
    max_unc = torch.max(unc_flatten, dim=1)[0].reshape(-1, 1, 1)
    min_unc = torch.min(unc_flatten, dim=1)[0].reshape(-1, 1, 1)
    unc_weight = 1 - ((uncertainty - min_unc) / (max_unc - min_unc))
    a = 0.5
    one = torch.ones_like(unc_weight)
    unc_weight = torch.where(unc_weight >= a, one, unc_weight * 1 / a)
    return pseudo_label, unc_weight


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
