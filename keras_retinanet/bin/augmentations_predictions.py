#!/usr/bin/env python

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


import argparse
import os
import sys

import keras
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.citypersons import CitypersonsGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.transform import random_transform_generator
from ..utils.pseudo_labels import get_detections_for_pseudo_labels

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True
    subparsers.add_parser('citypersons')

    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_false')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', default = '0')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).', default='./data/Citypersons_anchors')
    parser.add_argument('--save_path_predictions', help='Path for saving Pseudolabels.', default = '')
    parser.add_argument('--base_annotation_dir', type=str, help='Annotations are NOT used, only names of images.', default='./data/annotations.pickle')
    parser.add_argument('--image_dir', type=str, help='directory where images are stored')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    list_of_transform_generators = []

    transform_generator1 = random_transform_generator(
        flip_x_chance=0.0
    )
    transform_generator2 = random_transform_generator(
        flip_x_chance=1.0
    )

    list_of_transform_generators.append(transform_generator1)   #This is equivalent to no transformation
    list_of_transform_generators.append(transform_generator2)

    image_min_side = np.arange(816, 1331, 26)
    image_max_side = np.arange(1632, 2662, 52)

    for i, transform_generator in enumerate(list_of_transform_generators):
        for k in range(len(image_min_side)):

            generator = CitypersonsGenerator(
                set_name='train',
                image_dir = args.image_dir,
                image_min_side=image_min_side[k],
                image_max_side=image_max_side[k],
                transform_generator=transform_generator,
                visual_effect_generator = None,
                annotation_dir= args.base_annotation_dir,
                config=args.config,
                shuffle_groups=False)

            # optionally load anchor parameters
            anchor_params = None
            if args.config and 'anchor_parameters' in args.config:
                anchor_params = parse_anchor_parameters(args.config)

            # load the model
            if os.path.isfile(args.model):
                print('Loading model, this may take a second...')
                model = models.load_model(args.model, backbone_name=args.backbone)
            else: #If it is a directory, choose the latest model which (in case of safe best only) has the highest performance
                modelfiles = [f for f in listdir(args.model) if isfile(join(args.model, f))]
                sorted_modelfiles = sorted(modelfiles)
                print('This is the model used for creating pseudolabels', sorted_modelfiles[-1])
                print('Loading model, this may take a second...')
                model = models.load_model(args.model + '/' + sorted_modelfiles[-1], backbone_name=args.backbone)

            # optionally convert the model
            if args.convert_model:
                model = models.convert_model(model, anchor_params=anchor_params)

            model.summary()
            all_detections, filenames = get_detections_for_pseudo_labels(generator, model, score_threshold=args.score_threshold,
                                                        max_detections=args.max_detections)

            if not os.path.exists(args.save_path_predictions + str(i) + '_' + str(image_min_side[k]) + '_' + str(image_max_side[k])):
                os.makedirs(args.save_path_predictions + str(i) + '_' + str(image_min_side[k]) + '_' + str(image_max_side[k]))

            np.save(args.save_path_predictions + str(i) + '_' + str(image_min_side[k]) + '_' + str(image_max_side[k]) + '/filenames.npy', filenames)
            pickle.dump(all_detections, open(args.save_path_predictions + str(i) + '_' + str(image_min_side[k]) + '_' + str(image_max_side[k]) + '/all_detections.pkl', 'wb'))

if __name__ == '__main__':
    main()
