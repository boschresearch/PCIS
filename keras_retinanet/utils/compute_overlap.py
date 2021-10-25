"""
## Copyright (c) 2021 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the AGPL-3.0 license found in the
## LICENSE file in the root directory of this source tree.


# this code is based on the repo publicly available at 
#
#     http://github.com/rbgirshick/fast-rcnn/
#
# that was licensed under the MIT License.
"""


import numpy as np

def compute_overlap(boxes, query_boxes, iof):
    """
    Args
        boxes: (N, 4) ndarray of float
        query_boxes: (K, 4) ndarray of float
        iof: float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float64)
    for k in range(query_boxes.shape[0]):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(boxes.shape[0]):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    if iof[0] == 1.0:
                        ua = np.float64(
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1)
                        )
                        overlaps[n, k] = iw * ih / ua
                    elif iof[0] == 0.0:
                        ua = np.float64(
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1) +
                            box_area - iw * ih
                        )
                        overlaps[n, k] = iw * ih / ua
    return overlaps