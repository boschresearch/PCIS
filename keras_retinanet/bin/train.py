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
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel, update_model_in_generator
from ..callbacks.eval_citypersons import Evaluate_citypersons
from ..models.retinanet import retinanet_bbox
from ..preprocessing.citypersons import CitypersonsGenerator

from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    # store weights before loading pre-trained weights
    preloaded_layers = model.layers.copy()
    preloaded_weights = []
    for pre in preloaded_layers:
        preloaded_weights.append(pre.get_weights())
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)

        # compare previews weights vs loaded weights
    for layer, pre in zip(model.layers, preloaded_weights):
        weights = layer.get_weights()

        if weights:
            if np.array_equal(weights, pre):
                print('not loaded', layer.name)
            else:
                print('loaded', layer.name)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/gpu:0'):  # as suggested here: https://github.com/fizyr/keras-retinanet/issues/521
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=False)
            # Skipt_mismatch defaults to True in the Github implementation
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=False)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.1))

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, training_generator, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []
    if args.evaluation and validation_generator:
        evaluation = Evaluate_citypersons(validation_generator)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

     #save the model
    if args.snapshots:
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_Epoch_{{epoch:02d}}_val_MR_reasonable_{{miss_rate_reasonable:03f}}.h5'.format(
                backbone=args.backbone, dataset_type=args.dataset_type)),
        verbose=1,
        save_best_only=True,
        monitor="miss_rate_reasonable",
        mode='min'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)


    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='miss_rate_reasonable',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0))

    callbacks.append(keras.callbacks.EarlyStopping(
        monitor='miss_rate_reasonable', min_delta=0, patience=5, verbose=1, mode='auto',
        baseline=None, restore_best_weights=False))

    if args.consistency_with_pseudolabel:
        callbacks.append(update_model_in_generator(prediction_model, training_generator))

    return callbacks


def create_generators(args, preprocess_image, prediction_model):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """

    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
        'annotation_dir': args.base_annotation_dir,
        'use_adaptive_image_size': args.use_adaptive_image_size,
        'image_dir': args.image_dir,
        'dataset': args.dataset_type,
        'ignore_stages_without_bbox': args.ignore_stages_without_bbox,

    }

    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(
        flip_x_chance=0.5)
    visual_effect_generator = None

    train_generator = CitypersonsGenerator(
        'train',
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        use_random_crop =args.use_random_crop,
        cut_overlapping_boxes = args.cut_overlapping_boxes,
        prediction_model=prediction_model,
        **common_args
    )
    validation_generator = CitypersonsGenerator(
        'val',
        shuffle_groups=False,
        **common_args
    )

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """
    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))
    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))
    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True
    subparsers.add_parser('citypersons')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', help='Resume training from a snapshot.')
    group.add_argument('--imagenet_weights',help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights', help='Initialize the model with weights from a file.')
    group.add_argument('--no_weights', help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', default = '0')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=15)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default= './results/exp_name/')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze first two convolutional blocks.', type=bool, default=False)
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', type=bool, default=False)
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.', default='./data/Citypersons_anchors')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true', default=False)    #If set to true, val loss will always be computed.
    parser.add_argument('--base_annotation_dir', type=str, help='Name of base_annotation_dir which stores the train annotations')
    parser.add_argument('--use_adaptive_image_size', help='If true, input size of images varies in range 608 to 1024.', type=bool, default=False)
    parser.add_argument('--use_random_crop', help='Randomly crops images during training. ', type=bool, default=False)
    parser.add_argument('--cut_overlapping_boxes', help='If an annotation overlaps the image border, it is cropped. ', type=bool, default=True)
    parser.add_argument('--image_dir', help='Directory in which images are stored.', type=str, default='')
    parser.add_argument('--consistency_with_pseudolabel', help='If true, consistency between augmented versions of the same image is ensured by pseudo-labels.', type=bool, default=False)
    parser.add_argument('--ignore_stages_without_bbox', help='Ignore loss for stages from RetinaNet for which no bbox (either ground truth or pseudo-label) exists.', type=bool, default=False)

    parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers',          help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=10)
    return check_args(parser.parse_args(args))


def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)


    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, nms_threshold = 0.5, anchor_params=anchor_params)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes= 1,
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config,
        )

    # create the generators
    if args.consistency_with_pseudolabel:
        train_generator, validation_generator = create_generators(args, backbone.preprocess_image, prediction_model)
    else:
        train_generator, validation_generator = create_generators(args, backbone.preprocess_image, None)

    # print model summary
    print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    train_generator.compute_shapes = make_shapes_callback(model)
    if validation_generator:
        validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        train_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None

    # start training
    return training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator
    )

if __name__ == '__main__':
    main()