import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.model.rpn import _RPN
from torchvision.ops.roi_pool import roi_pool
from torchvision.ops.roi_align import roi_align
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, model_conf, classes):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = model_conf["hyperParameters"]["use_class_agnostic_regression"]
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(model_conf=model_conf, din=self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(model_conf=model_conf, nclasses=self.n_classes)

        self.POOLING_SIZE = model_conf["hyperParameters"]["pooling_size"]
        self.POOLING_MODE = model_conf["hyperParameters"]["pooling_mode"]

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        # do roi pooling based on predicted rois
        if self.POOLING_MODE == 'align':
            pooled_feat = roi_align(input=base_feat, boxes=rois.view(-1, 5),output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=1.0/16.0, sampling_ratio=0)
        elif self.POOLING_MODE == 'pool':
            pooled_feat = roi_pool(base_feat, rois.view(-1, 5), output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=1.0/16.0)
        else:
            raise Exception("You have not picked a valid roi pooling mode")

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self, model_conf):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, model_conf["hyperParameters"]["truncated"])
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, model_conf["hyperParameters"]["truncated"])
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, model_conf["hyperParameters"]["truncated"])
        normal_init(self.RCNN_cls_score, 0, 0.01, model_conf["hyperParameters"]["truncated"])
        normal_init(self.RCNN_bbox_pred, 0, 0.001, model_conf["hyperParameters"]["truncated"])

    def create_architecture(self, model_conf):
        self._init_modules(model_conf=model_conf)
        self._init_weights(model_conf=model_conf)
