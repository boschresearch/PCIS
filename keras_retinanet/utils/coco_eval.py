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

from pycocotools.cocoeval import COCOeval

import keras
import numpy as np
import json
from time import sleep
import os
from os import listdir
from os.path import isfile, join

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def evaluate_coco(generator, model, threshold=0.05):
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator : The generator for generating the evaluation data.
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    for index in progressbar.progressbar(range(generator.size()), prefix='COCO evaluation: '):    #generator.size(
        image = generator.load_image(index)
        image = generator.preprocess_image(image)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # compute predicted labels and scores
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:  # or counter_predictions==100:      #Use this to only include 100 detections
                break

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_coco_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
            }

            # append detection to results
            results.append(image_result)

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])

    if not len(results):
        return

    mypath = './results/'
    current_dir = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    print('current dir', current_dir)

    current_dir.sort(key=os.path.getmtime)

    if len(current_dir)<3:
        # write output
        json.dump(results, open('./results/_{}_bbox_results'.format(generator.set_name) + str(len(current_dir)) + '.json', 'w'),
                  indent=4)
        json.dump(image_ids, open('./results/image_ids/_{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)
        # load results in COCO evaluation tool
        coco_true = generator.coco
        coco_pred = coco_true.loadRes('./results/_{}_bbox_results'.format(generator.set_name) + str(len(current_dir)) + '.json')
    else:
        # write output
        json.dump(results, open(current_dir[0], 'w'), indent=4)
        json.dump(image_ids, open('./results/image_ids/_{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = generator.coco
        coco_pred = coco_true.loadRes(current_dir[0])

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

