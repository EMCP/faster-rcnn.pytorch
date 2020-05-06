from __future__ import division

import numpy as np
import numpy.random as npr
import cv2

from utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(model_conf, roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(model_conf["hyperParameters"]["scales"]),
                                    size=num_images)
    assert (model_conf["hyperParameters"]["batch_size"] % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, model_conf["hyperParameters"]["batch_size"])

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(scales=model_conf["hyperParameters"]["scales"],
                                         pixel_means=np.array(model_conf["hyperParameters"]["pixel_means"]),
                                         max_size=model_conf["hyperParameters"]["max_size"],
                                         roidb=roidb,
                                         scale_inds=random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if model_conf["hyperParameters"]["use_all_gt"]:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    blobs['img_id'] = roidb[0]['img_id']

    return blobs


def _get_image_blob(scales, pixel_means, max_size, roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])

        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2
        # rgb -> bgr
        # im = im[:, :, ::-1]

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = scales[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im=im,
                                        pixel_means=pixel_means,
                                        target_size=target_size,
                                        max_size=max_size)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales


def _get_image_blob_demoing(modelconf, im):
    """Converts an image into a network input. USED ONLY IN DEMO RN
      Arguments:
        im (ndarray): a color image in BGR order
      Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
          in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    if im_orig.shape[2] == 4:
        # in the event you have an image with alpha channels, drop it for now
        im_orig = im_orig[:, :, :3]
    im_orig -= modelconf["hyperParameters"]["pixel_means"]

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in modelconf["hyperParameters"]["testing"]["scales"]:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > modelconf["hyperParameters"]["testing"]["max_size"]:
            im_scale = float(modelconf["hyperParameters"]["testing"]["max_size"]) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)
