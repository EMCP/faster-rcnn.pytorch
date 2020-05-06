import os
import pprint
import datetime

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from lib.model.faster_rcnn.resnet import resnet_backbone
from lib.model import FasterRCNNSampler
from lib.roi_data_layer import roibatchLoader
from lib.roi_data_layer import combined_roidb
from lib.model.utils.net_utils import adjust_learning_rate, save_checkpoint, loss_is_improved


def train(data_conf, model_conf, **kwargs):
    import time
    overallstarttime = time.time()
    print("Using hyperParameters:")
    pprint.pprint(model_conf["hyperParameters"])

    np.random.seed(model_conf["hyperParameters"]["random_seed"])

    output_dir = data_conf["save_dir"] \
                 + "/" \
                 + model_conf["hyperParameters"]["net"] \
                 + "/" \
                 + data_conf["image_data_training_id"] \
                 + "_" + str(datetime.datetime.now()).replace(" ", "-").replace(":", "_")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Data loading: " + str(data_conf["image_data_training_id"]))
    imdb, roidb, ratio_list, ratio_index = combined_roidb(data_conf=data_conf, model_conf=model_conf)
    train_size = len(roidb)

    print("Data loading: train_size: " + str(len(roidb)))
    sampler_batch = FasterRCNNSampler(train_size=train_size, batch_size=model_conf["hyperParameters"]["batch_size"], data_source="")

    dataset = roibatchLoader(roidb=roidb,
                             ratio_list=ratio_list,
                             ratio_index=ratio_index,
                             num_classes=imdb.num_classes,
                             model_conf=model_conf,
                             training=True)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=model_conf["hyperParameters"]["batch_size"],
                                             sampler=sampler_batch,
                                             num_workers=model_conf["pytorch_engine"]["num_workers"])

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    if model_conf["pytorch_engine"]["enable_cuda"]:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    # initialize the network here
    fasterRCNN = resnet_backbone(data_conf=data_conf, model_conf=model_conf, classes=imdb.classes)

    fasterRCNN.create_architecture(model_conf=model_conf)

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [
                    {
                        'params': [value],
                        'lr': model_conf["hyperParameters"]["learning_rate"] * (model_conf["hyperParameters"]["double_bias"] + 1),
                        'weight_decay': model_conf["hyperParameters"]["bias_decay"] and
                                        model_conf["hyperParameters"]["weight_decay"] or 0
                    }]
            else:
                params += [{'params': [value],
                            'lr': model_conf["hyperParameters"]["learning_rate"],
                            'weight_decay': model_conf["hyperParameters"]["weight_decay"]}]

    print("Done setting bias gradients")

    if model_conf["pytorch_engine"]["enable_cuda"]:
        fasterRCNN.cuda()

    if model_conf["hyperParameters"]["optimizer"] == "adam":
        # it seems here, prior examples multiple learning rate
        # by 0.1 , and save it back into the learning rate.. why ?
        optimizer = torch.optim.Adam(params)
    elif model_conf["hyperParameters"]["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, momentum=model_conf["hyperParameters"]["momentum"])
    else:
        raise Exception("You must configure an optimizer.  For example 'sgd'")

    if model_conf["pytorch_engine"]["resume_checkpoint"]:
        print("TODO, implement resume checkpoints!")

    if model_conf["pytorch_engine"]["enable_multiple_gpus"]:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / model_conf["hyperParameters"]["batch_size"])

    if model_conf["pytorch_engine"]["enable_tfb"]:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(output_dir + "/logs")

    learning_rate = model_conf["hyperParameters"]["learning_rate"]

    best_loss_rpn_cls = None
    best_loss_rpn_box = None
    best_loss_rcnn_cls = None
    best_loss_rcnn_box = None

    for epoch in range(model_conf["hyperParameters"]["epoch_start"], model_conf["hyperParameters"]["epoch_max"] + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (model_conf["hyperParameters"]["learning_decay_step"] + 1) == 0:
            adjust_learning_rate(optimizer, model_conf["hyperParameters"]["learning_decay_gamma"])
            learning_rate *= model_conf["hyperParameters"]["learning_decay_gamma"]

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)

            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % model_conf["hyperParameters"]["display_interval"] == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (model_conf["hyperParameters"]["display_interval"] + 1)

                if model_conf["pytorch_engine"]["enable_multiple_gpus"]:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (model_conf["pytorch_engine"]["session"], epoch, step, iters_per_epoch, loss_temp, learning_rate))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                if model_conf["pytorch_engine"]["enable_tfb"]:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(model_conf["pytorch_engine"]["session"]), info, (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        if(loss_is_improved(loss_rpn_cls,
                            loss_rpn_box,
                            loss_rcnn_cls,
                            loss_rcnn_box,
                            best_loss_rpn_cls,
                            best_loss_rpn_box,
                            best_loss_rcnn_cls,
                            best_loss_rcnn_box)):
            best_loss_rpn_cls = loss_rpn_cls
            best_loss_rpn_box = loss_rpn_box
            best_loss_rcnn_cls = loss_rcnn_cls
            best_loss_rcnn_box = loss_rcnn_box

            save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(model_conf["pytorch_engine"]["session"], epoch, step))
            save_checkpoint({
                'session': model_conf["pytorch_engine"]["session"],
                'epoch': epoch + 1,
                'model': fasterRCNN.module.state_dict() if model_conf["pytorch_engine"]["enable_multiple_gpus"] else fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': model_conf["hyperParameters"]["pooling_mode"],
                'class_agnostic': model_conf["hyperParameters"]["use_class_agnostic_regression"],
            }, save_name)
            print('save model: {}'.format(save_name))

    if model_conf["pytorch_engine"]["enable_tfb"]:
        logger.close()

    overallendtime = time.time()

    print("START Time " + str(overallstarttime))
    print("END Time " + str(overallendtime))

    grandtotal_secs = overallendtime - overallstarttime
    print("TOTAL RUNTIME WAS " + str(grandtotal_secs) + " seconds .. or ")

    day = grandtotal_secs // (24 * 3600)
    time = grandtotal_secs % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    print("d:h:m:s -> %d:%d:%d:%d" % (day, hour, minutes, seconds))


if __name__ == "__main__":
    import json

    with open("cfgs/dataset.json") as f:
        config_json = json.load(f)

    with open("cfgs/config.json") as fp:
        model_conf = json.load(fp)

    train(data_conf=config_json, model_conf=model_conf)
