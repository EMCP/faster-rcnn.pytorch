import os
import pprint
import sys
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.ops import nms

from lib.model.faster_rcnn.resnet import resnet_backbone
from lib.model.rpn import bbox_transform_inv, clip_boxes
from lib.roi_data_layer import _get_image_blob_demoing
from lib.model.utils.net_utils import vis_detections


def demo(data_conf, model_conf, **kwargs):
    print("Beginning demo of model")
    print("Using hyperParameters:")
    pprint.pprint(model_conf["hyperParameters"])

    np.random.seed(model_conf["hyperParameters"]["random_seed"])

    possible_classes = np.asarray(data_conf["classes_available"])

    model_input_dir = data_conf["save_dir"] \
                + "/" \
                + model_conf["hyperParameters"]["net"] \
                + "/" \
                + data_conf["demo_model_dir"]

    if not os.path.exists(model_input_dir):
        raise Exception('There is no input directory for loading network from: ' + str(model_input_dir))

    load_name = os.path.join(model_input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(
                                 model_conf["hyperParameters"]["testing"]["check_session"],
                                 model_conf["hyperParameters"]["testing"]["check_epoch"],
                                 model_conf["hyperParameters"]["testing"]["check_point"]))

    print("load checkpoint %s" % load_name)

    model_conf["hyperParameters"]["use_pretrained_net"] = False
    fasterRCNN = resnet_backbone(data_conf=data_conf, model_conf=model_conf, classes=possible_classes)

    fasterRCNN.create_architecture(model_conf=model_conf)

    if model_conf["pytorch_engine"]["enable_cuda"]:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))

    fasterRCNN.load_state_dict(checkpoint['model'])

    if 'pooling_mode' in checkpoint.keys():
        model_conf["hyperParameters"]["pooling_mode"] = checkpoint['pooling_mode']

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

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True

    # Set up webcam or get image directories
    if data_conf["webcam_data_id_num"] >= 0:
        cap = cv2.VideoCapture(data_conf["webcam_data_id_num"])
        num_images = 0
        print('Loaded Webcam: {} == webcam id.'.format(data_conf["webcam_data_id_num"]))
    else:
        imglist = os.listdir(data_conf["demo_in_image_dir"])
        num_images = len(imglist)
        print('Loaded Photo: {} images.'.format(num_images))

    while num_images >= 0:
        total_tic = time.time()
        if data_conf["webcam_data_id_num"] == -1:
            num_images -= 1

        # Get image from the webcam or Load the demo image
        if data_conf["webcam_data_id_num"] >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame = cap.read()
            im_in = np.array(frame)
        else:
            im_file = os.path.join(data_conf["demo_in_image_dir"], imglist[num_images])
            if im_file.endswith(".png") or im_file.endswith(".jpg"):
                im_in = np.array(cv2.imread(im_file))
            else:
                continue

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        # rgb -> bgr
#        im = im_in[:, :, ::-1]
        im = im_in

        blobs, im_scales = _get_image_blob_demoing(model_conf, im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()

        det_tic = time.time()

        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if model_conf["hyperParameters"]["testing"]["bbox_reg"]:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if model_conf["hyperParameters"]["bbox_normalize_targets_precomputed"]:
                # Optionally normalize targets by a precomputed mean and stdev
                if model_conf["hyperParameters"]["use_class_agnostic_regression"]:
                    if model_conf["pytorch_engine"]["enable_cuda"]:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_stds"]).cuda() \
                                     + torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_means"]).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_stds"]) \
                                     + torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_means"])

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if model_conf["pytorch_engine"]["enable_cuda"]:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_stds"]).cuda() \
                                     + torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_means"]).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_stds"]) \
                                     + torch.FloatTensor(model_conf["hyperParameters"]["bbox_normalize_means"])
                    box_deltas = box_deltas.view(1, -1, 4 * len(possible_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)

        for j in range(1, len(possible_classes)):

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
                cls_dets = cls_dets[order]
                keep = nms(boxes=cls_boxes[order, :],
                           scores=cls_scores[order],
                           iou_threshold=model_conf["hyperParameters"]["testing"]["nms"])
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, possible_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if data_conf["webcam_data_id_num"] == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'.format(num_images + 1, len(imglist), detect_time, nms_time))
            sys.stdout.flush()

        if vis and data_conf["webcam_data_id_num"] == -1:
            result_path = os.path.join(data_conf["demo_out_image_dir"], imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
        else:
            im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", im2showRGB)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('Frame rate:', frame_rate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if data_conf["webcam_data_id_num"] >= 0:
        cap.release()
        cv2.destroyAllWindows()

    print("Demo finished")


if __name__ == "__main__":
    import json

    with open("cfgs/dataset.json") as f:
        config_json = json.load(f)

    with open("cfgs/config.json") as fp:
        model_conf = json.load(fp)

    demo(data_conf=config_json, model_conf=model_conf)
