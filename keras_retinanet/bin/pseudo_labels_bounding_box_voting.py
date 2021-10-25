# -*- coding: utf-8 -*-

"""
## Copyright (c) 2021 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the AGPL-3.0 license found in the
## LICENSE file in the root directory of this source tree.

# this code is based on the repo publicly available at 
#     https://github.com/fizyr/keras-retinanet
# that was licensed under the Apache-2.0 License.

# The function box_voting was taken from https://github.com/facebookresearch/Detectron. 
# that was licensed under the Apache License, Version 2.0
"""

import pickle
import os
import sys
from os import listdir
from os.path import join, isdir
import numpy as np
from copy import deepcopy
import cv2
import argparse

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.bin"

from ..utils.compute_overlap import compute_overlap
from ..utils.pseudo_labels import remove_detections_with_gt_overlap, extract_base_annotations, change_box_format


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Parser for Pseudo-Label creation based on bounding box voting.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True
    subparsers.add_parser('citypersons')

    parser.add_argument('--pseudolabel_certainty_threshold', help='Predictions with confidence above this threshold are considered for clustering.', type=float, default=0.5)
    parser.add_argument('--remove_threshold_pred_annot_same_class', help='If a pseudolabel has an IOU with a known ground truth above this threshold, it is removed.', type=float, default=0.6)
    parser.add_argument('--nms_threshold', help='Threshold used in NMS.', type=float, default=0.3)
    parser.add_argument('--predictions_dir', help='Name of path in which predictions are stored.')
    parser.add_argument('--out_dir', help='Name of path in which pseudolabels are stored.')
    parser.add_argument('--base_annotation', help='Name of path in which sparse ground truth annotations are stored.')

    return parser.parse_args(args)


def combine_detections(predictions_dir, dirs_to_network_output, filenames_pseudolabel, pseudolabel_certainty_threshold, nms_threshold):
    combined_detections_total = {}
    for det_file in dirs_to_network_output:
        detections = pickle.load(open(predictions_dir + det_file + '/all_detections.pkl', 'rb'))
        for nb, file in enumerate(filenames_pseudolabel):
            if file in combined_detections_total:
                combined_detections_total[file] = np.concatenate((combined_detections_total[file], [i for i in detections[nb]]), axis=1)
            else:
                combined_detections_total[file] = []
                combined_detections_total[file] = [i for i in detections[nb]]

    combined_detections = deepcopy(combined_detections_total)
    for nb, file in enumerate(list(combined_detections_total.keys())):
        if combined_detections_total[file] != []:
            boxes = list(combined_detections_total[file][0,:,:4])
            scores = list(combined_detections_total[file][0,:,4])
            tmp_boxes = change_box_format(deepcopy(boxes))
            indices = [i[0] for i in cv2.dnn.NMSBoxes(tmp_boxes, scores, score_threshold=pseudolabel_certainty_threshold,
                                       nms_threshold=nms_threshold)]
            combined_detections[file] = [combined_detections[file][0][i] for i in range(len(combined_detections[file][0])) if i in indices]

    return combined_detections, combined_detections_total


def compute_iou_matrix(box_matrix1, box_matrix2):
    iou_matrix = np.zeros(shape=(len(box_matrix1), len(box_matrix2)))
    for i in range(len(box_matrix1)):
        for i2 in range(i, len(box_matrix2)):
            tmp_iou = compute_overlap(np.expand_dims(box_matrix1[i, :4], axis=0), np.expand_dims(box_matrix2[i2, :4], axis=0), np.array([0.0]))
            iou_matrix[i][i2] = tmp_iou
    return iou_matrix


def box_voting(top_dets, all_dets, thresh=0.5, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = compute_iou_matrix(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws ** beta) ** (1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws)) ** beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )
    return top_dets_out


def create_pseudolabels (combined_detections, combined_detections_total, filename_label_annotations, filenames_pseudolabel, remove_threshold_pred_annot_same_class, dataset_type, data, filename_to_idx):
    for nb, file in enumerate(sorted(filenames_pseudolabel)):
        base_annotations_to_include = extract_base_annotations(filename_label_annotations, file.split('/')[-1])

        det_per_model = deepcopy(combined_detections[file])
        det_total_per_model = deepcopy(combined_detections_total[file])
        if det_per_model == []:
            det_combined = []
        else:
            det_per_model = np.vstack(det_per_model)
            det_total_per_model = np.vstack(det_total_per_model)
            det_combined = box_voting(det_per_model, det_total_per_model)

        det_combined = remove_detections_with_gt_overlap(base_annotations_to_include, det_combined,
                                                         remove_threshold_pred_annot_same_class=remove_threshold_pred_annot_same_class)
        for pseudo in det_combined:
            tmp_dict = {}
            tmp_dict['id'] = 'pseudolabel'
            tmp_dict['lbl'] = 'person'
            tmp_dict['pos'] = [pseudo[0], pseudo[1], pseudo[2] - pseudo[0], pseudo[3] - pseudo[1]]
            filename_label_annotations[file.split('/')[-1]]['annotations'].append(tmp_dict)

    return filename_label_annotations


def main(args=None, data=[]):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    if not os.path.exists(args.out_dir.rsplit('/', 1)[0]):
        os.makedirs(args.out_dir.rsplit('/', 1)[0])

    dirs_to_network_output = [f for f in listdir(args.predictions_dir) if isdir(join(args.predictions_dir, f))]

    with open(args.base_annotation_dir, "rb") as f:
        print("Data: Loading annotations...")
        filename_label_annotations = pickle.load(f)

    filename_to_idx = None
    filenames_pseudolabel = list(np.load(args.predictions_dir + dirs_to_network_output[0] + '/filenames.npy'))

    combined_detections, combined_detections_total = combine_detections(args.predictions_dir, dirs_to_network_output,
                                                                        filenames_pseudolabel, args.pseudolabel_certainty_threshold,
                                                                        args.nms_threshold)
    filename_label_annotations = create_pseudolabels(combined_detections, combined_detections_total, filename_label_annotations, filenames_pseudolabel, args.remove_threshold_pred_annot_same_class, args.dataset_type, data, filename_to_idx)
    with open(args.out_dir, 'wb') as pickle_file:
       pickle.dump(filename_label_annotations, pickle_file)

    print('Pseudolabels saved to ', args.out_dir)

if __name__ == '__main__':
    main()