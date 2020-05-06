import os
import pickle
import pprint
import sys
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.ops import nms

from model.faster_rcnn.resnet import resnet_backbone
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
from utils.net_utils import vis_detections


def evaluate(data_conf, model_conf, **kwargs):
    print("Beginning evaluation of model")
    print("Using hyperParameters:")
    pprint.pprint(model_conf["hyperParameters"])

    np.random.seed(model_conf["hyperParameters"]["random_seed"])

    imdb, roidb, ratio_list, ratio_index = combined_roidb(data_conf=data_conf, model_conf=model_conf, training=False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = data_conf["save_dir"] \
                + "/" \
                + model_conf["hyperParameters"]["net"] \
                + "/" \
                + data_conf["image_data_testing_id"]

    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from: ' + str(input_dir))

    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(
                                 model_conf["hyperParameters"]["testing"]["check_session"],
                                 model_conf["hyperParameters"]["testing"]["check_epoch"],
                                 model_conf["hyperParameters"]["testing"]["check_point"]))

    fasterRCNN = resnet_backbone(data_conf=data_conf, model_conf=model_conf, classes=imdb.classes)

    fasterRCNN.create_architecture(model_conf=model_conf)

    print("load checkpoint %s" % load_name)
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])

    if 'pooling_mode' in checkpoint.keys():
        model_conf["hyperParameters"]["pooling_mode"] = checkpoint['pooling_mode']

    print('load model successfully!')
    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if model_conf["pytorch_engine"]["enable_cuda"]:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if model_conf["pytorch_engine"]["enable_cuda"]:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    if model_conf["hyperParameters"]["testing"]["enable_visualization"]:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    dataset = roibatchLoader(roidb=roidb,
                             ratio_list=ratio_list,
                             ratio_index=ratio_index,
                             num_classes=imdb.num_classes,
                             model_conf=model_conf,
                             training=False,
                             normalize=False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}

    fasterRCNN.eval()

    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    for i in range(num_images):

        data = next(data_iter)

        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(
            im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if model_conf["hyperParameters"]["testing"]["bbox_reg"]:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if model_conf["hyperParameters"]["bbox_normalize_targets_precomputed"]:
                # Optionally normalize targets by a precomputed mean and stdev
                if model_conf["hyperParameters"]["use_class_agnostic_regression"]:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_stds"]).cuda() \
                                 + torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_means"]).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        model_conf["hyperParameters"]["bbox_normalize_stds"]).cuda() \
                                 + torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_means"]).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if model_conf["hyperParameters"]["testing"]["enable_visualization"]:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if model_conf["hyperParameters"]["use_class_agnostic_regression"]:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], model_conf["hyperParameters"]["testing"]["nms"])
                cls_dets = cls_dets[keep.view(-1).long()]
                if model_conf["hyperParameters"]["testing"]["enable_visualization"]:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'.format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if model_conf["hyperParameters"]["testing"]["enable_visualization"]:
            cv2.imwrite('result.png', im2show)

    output_dir = data_conf["save_dir"] + "/detections/" + data_conf["image_data_training_id"]
    det_file = os.path.join(output_dir, 'detections.pkl')

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))


if __name__ == "__main__":
    import json

    with open("./dataset.json") as f:
        config_json = json.load(f)

    with open("./config.json") as fp:
        model_conf = json.load(fp)

    evaluate(data_conf=config_json, model_conf=model_conf)
