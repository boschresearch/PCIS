
"""
## Copyright (c) 2021 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the AGPL-3.0 license found in the
## LICENSE file in the root directory of this source tree.


# this code is based on the repo publicly available at 
#
#     https://github.com/fizyr/keras-retinanet
#
# that was licensed under the Apache-2.0 License.
"""


import os
import sys
import numpy as np
import keras
from numpy.linalg import inv
import progressbar

assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.bin"

from ..utils.compute_overlap import compute_overlap
from ..utils.transform import transform_aabb


def remove_detections_with_gt_overlap(base_annotations, detections, remove_threshold_pred_annot_same_class):
    clean_detections = []
    pseudolabel_annotation_overlap = np.zeros(len(detections))
    for lab in range(len(detections)):
        for lab2 in range(len(base_annotations)):
            tmp_iou = compute_overlap(np.expand_dims(detections[lab], axis=0), np.expand_dims(base_annotations[lab2], axis = 0), np.array([0.0]))
            if tmp_iou > pseudolabel_annotation_overlap[lab]:
                pseudolabel_annotation_overlap[lab] = tmp_iou

    for t in range(len(pseudolabel_annotation_overlap)):
        if pseudolabel_annotation_overlap[t] < remove_threshold_pred_annot_same_class:
            clean_detections.append(detections[t])
    return clean_detections

def extract_base_annotations(filename_label_annotations, img_name):
    base_annotations = []
    det = filename_label_annotations[img_name]['annotations']
    for object_annotation in det:
        if object_annotation['lbl'] == 'person':
            tmp = np.empty(6)
            tmp[0] = object_annotation['pos'][0]
            tmp[1] = object_annotation['pos'][1]
            tmp[2] = object_annotation['pos'][2] + object_annotation['pos'][0]
            tmp[3] = object_annotation['pos'][3] + object_annotation['pos'][1]
            tmp[5] = 0  # There is only one class - person
            base_annotations.append(tmp)
    return base_annotations

def extract_boxes_and_scores(tmp_detections):
    boxes = list(tmp_detections[:, :4])
    scores = list(tmp_detections[:, 4])
    return boxes, scores

def change_box_format(boxes):
    for i in range(len(boxes)):
        boxes[i][2] = boxes[i][2] - boxes[i][0]
        boxes[i][3] = boxes[i][3] - boxes[i][1]
    return boxes

def get_detections_for_pseudo_labels(generator, model, score_threshold=0.05, max_detections=100):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    filenames = []

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):

        image, filename    = generator.load_image_return_filename(i)
        filenames.append(filename)
        if generator.visual_effect_generator is not None:
            image, _ = generator.random_visual_effect_group_entry(image.copy(), [])

        image        = generator.preprocess_image(image)
        image, transform = generator.random_transform_only_img(image, transform=None)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        # correct boxes for image scale
        boxes /= scale

        # undo transformation only after box predictions have been rescaled
        # Transform the bounding boxes in the annotations.
        if transform is not None:
            inv_transform = inv(transform)
            for index in range(boxes.shape[1]):
                boxes[0][index, :] = transform_aabb(inv_transform, boxes[0][index, :])

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    return all_detections, filenames