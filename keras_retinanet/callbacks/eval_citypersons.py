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

import re
import keras
from ..utils.eval_demo_citypersons import write_detections_to_json, eval_citypersons_miss_rate
from ..utils.eval import evaluate, _get_detections


class Evaluate_citypersons(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        score_threshold=0.01,
        max_detections=1000,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.verbose         = verbose

        super(Evaluate_citypersons, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        raw_detections = _get_detections(self.generator, self.model, score_threshold=self.score_threshold,
                                         max_detections=self.max_detections)
        write_detections_to_json(raw_detections, './data/tmp.json')

        eval_citypersons_miss_rate('./data/tmp.json', annType='bbox',
                                   out_dir='./data/Citypersons_official_eval')

        with open('./data/Citypersons_official_eval/results.txt', 'r') as f:
            x = f.readlines()

        line = x[0].split('=')[-1]
        miss_rate_reasonable = float(re.sub('[^0-9.]', '', line))

        logs['miss_rate_reasonable'] = miss_rate_reasonable

        if self.verbose == 1:
            print('miss_rate_reasonable: {:.4f}'.format(miss_rate_reasonable))
