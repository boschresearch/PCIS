"""
This source code is from https://github.com/DocF/Soft-NMS
Copyright (c) 2020 DocF
This source code is licensed under the MIT license found in the
3rd-party-licenses.txt file in the root directory of this source tree.
"""

import numpy as np

def nms(dets, sc, NMS_threshold=0.3, sigma=0.5, min_certainty=0.5, method=3):

    """
    nms
    :param dets:   boxes in format [y1, x1, y2, x2]
    :param sc:     scores of boxes
    :param NMS_threshold:     overlap threshold
    :param sigma:  sigma for gaussian
    :param thresh: certainty threshold
    :param method: 1=linear, 2=gaussian, 3=greedy
    :return:       indices of boxes
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > NMS_threshold] = weight[ovr > NMS_threshold] - ovr[ovr > NMS_threshold]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > NMS_threshold] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > min_certainty]
    keep = inds.astype(int)

    if len(keep) == 0:
        return ()
    else:
        return keep
