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

import numpy as np
import random
import warnings
import keras

from ..utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes,
)
from ..utils.config import parse_anchor_parameters
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb
from ..utils.NMS import nms
from ..utils.compute_overlap import compute_overlap

class Generator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
        self,
        transform_generator = None,
        visual_effect_generator=None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image,
        config=None,
        n_anchors = 9,
        use_adaptive_image_size = False,
        max_overlap = 0.0,
        prediction_model=None,
        use_random_crop = False,
        cut_overlapping_boxes = False,
        ignore_stages_without_bbox = False,
        dataset = ''
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """
        self.transform_generator    = transform_generator
        self.visual_effect_generator = visual_effect_generator
        self.batch_size             = int(batch_size)
        self.dataset                = dataset
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.preprocess_image       = preprocess_image
        self.config                 = config
        self.n_anchors = n_anchors
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
            self.n_anchors = len(anchor_params.ratios) * len(anchor_params.scales)

        self.use_adaptive_image_size = use_adaptive_image_size
        self.max_overlap = max_overlap
        self.prediction_model = prediction_model
        if self.prediction_model is not None:
            self.prediction_model._make_predict_function()
        self.use_random_crop = use_random_crop
        self.cut_overlapping_boxes = cut_overlapping_boxes
        self.ignore_stages_without_bbox  = ignore_stages_without_bbox
        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_raw_prediction(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_raw_prediction method not implemented')


    def load_annotations(self, image_index, box_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group, box_index):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index, box_index) for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] + self.max_overlap < -0.1) |
                (annotations['bboxes'][:, 1] + self.max_overlap < -0.1) |
                (annotations['bboxes'][:, 2] > image.shape[1] + 0.1 + self.max_overlap) |
                (annotations['bboxes'][:, 3] > image.shape[0] + 0.1 + self.max_overlap)
            )[0]

            if self.cut_overlapping_boxes:
                for i in invalid_indices:
                    tmp_box = annotations['bboxes'][i]
                    if tmp_box[0] < 0:
                        tmp_box [0] = 0
                    if tmp_box[1] < 0:
                        tmp_box [1] = 0
                    if tmp_box[2] > image.shape[1]:
                        tmp_box [2] = image.shape[1]
                    if tmp_box[3] > image.shape[0]:
                        tmp_box [3] = image.shape[0]
            else:
                # delete invalid indices
                if len(invalid_indices):
                    warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                        group[index],
                        image.shape,
                        annotations['bboxes'][invalid_indices, :]
                    ))
                    for k in annotations_group[index].keys():
                        annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def load_pseudolabel_raw_prection(self, group):
        return [self.load_raw_prediction(image_index) for image_index in group]


    def random_visual_effect_group_entry(self, image, annotations):
        """ Randomly transforms image and annotation.
        """
        visual_effect = next(self.visual_effect_generator)
        # apply visual effect
        image = visual_effect(image)
        return image, annotations

    def random_visual_effect_group(self, image_group, annotations_group):
        """ Randomly apply visual effect on each image.
        """
        assert(len(image_group) == len(annotations_group))

        if self.visual_effect_generator is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )

        return image_group, annotations_group


    def random_crop_group_entry(self, image, annotations, min_crop_size=0.3):
        """ Randomly transforms image and annotation.
        """

        h, w, c = 1024., 2048., 3.    #Hardcoded for citypersons

        if np.random.rand() < .2:
            # do nothing
            return image, annotations

        while True:
            new_w = random.uniform(min_crop_size * w, w)
            new_h = random.uniform(min_crop_size * h, h)

            if new_h / new_w < 0.25 or new_h / new_w > 1:
                continue

            left = random.uniform(0, w - new_w)
            top = random.uniform(0, h - new_h)

            image = image[int(top):int(top+new_h), int(left):int(left+new_w), :]

            if annotations['bboxes'].shape[0]:
                for i in range(annotations['bboxes'].shape[0]):
                    tmp_box = annotations['bboxes'][i]
                    tmp_box[0] = tmp_box[0] - left
                    tmp_box[1] = tmp_box[1] - top
                    tmp_box[2] = tmp_box[2] - left
                    tmp_box[3] = tmp_box[3] - top

                    annotations['bboxes'][i] = tmp_box

            if annotations['ignore_region'].shape[0]:
                for i in range(annotations['ignore_region'].shape[0]):
                    tmp_box = annotations['ignore_region'][i]
                    tmp_box[0] = tmp_box[0] - left
                    tmp_box[1] = tmp_box[1] - top
                    tmp_box[2] = tmp_box[2] - left
                    tmp_box[3] = tmp_box[3] - top

                    annotations['ignore_region'][i] = tmp_box

            return image, annotations

    def random_crop_group(self, image_group, annotations_group):
        """ Randomly apply visual effect on each image.
        """
        assert(len(image_group) == len(annotations_group))
        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_crop_group_entry(
                image_group[index], annotations_group[index]
            )
        return image_group, annotations_group


    def random_transform_only_img(self, image, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

        return image, transform


    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

            annotations['ignore_region'] = annotations['ignore_region'].copy()
            for index in range(annotations['ignore_region'].shape[0]):
                annotations['ignore_region'][index, :] = transform_aabb(transform, annotations['ignore_region'][index, :])

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """ Randomly transforms each image and its annotations.
        """

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """

        if 'train' in self.set_name and self.use_adaptive_image_size:
            image_min_side = np.arange(608, 1025, 13)
            image_max_side = np.arange(1216, 2049, 26)
            idx = np.random.choice(np.arange(len(image_min_side)))
            return resize_image(image, min_side=image_min_side[idx], max_side=image_max_side[idx])
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image, substract imagenet mean
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale
        annotations['ignore_region'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group


    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes(),
            tmp_image_sizes = max_shape,
            n_anchors = self.n_anchors,
            ignore_stages_without_bbox = self.ignore_stages_without_bbox,
            shapes_callback=self.compute_shapes,
        )

        return list(batches)


    def compute_input_output(self, group, box_index=None):
        """ Compute inputs and target outputs for the network.
        """

        annotations_group = self.load_annotations_group(group, box_index)

        if self.prediction_model is not None:

            image_group = self.load_image_group(group)
            for i in range(len(image_group)):
                image_group[i], im_scale = resize_image(image_group[i], min_side=self.image_min_side, max_side=self.image_max_side)
                image_group[i] = keras.backend.cast_to_floatx(self.preprocess_image(image_group[i]))

                annotations_group[i]['bboxes'] *= im_scale
                annotations_group[i]['ignore_region'] *= im_scale
                boxes, scores, labels = self.prediction_model.predict_on_batch(np.expand_dims(image_group[i], axis=0))[:3]
                annotations_group[i] = self.create_pseudo_labels(boxes, scores, labels, annotations_group[i])

                annotations_group[i]['bboxes'] /= im_scale
                annotations_group[i]['ignore_region'] /= im_scale

        image_group = self.load_image_group(group)

        if self.use_random_crop:
            image_group, annotations_group = self.random_crop_group(image_group, annotations_group)

        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform preprocessing steps (resize of image and annotations), substract ImageNet mean
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets


    def update_model (self, new_model):
        self.model = new_model
        self.model._make_predict_function()


    def create_pseudo_labels(self, boxes, scores, labels, annotations_group):
        scores_argmin = np.argmin(scores[0])
        if np.min(scores[0]) < 0.0 and np.max(scores[0]) > 0.0:
            tmp_scores = scores[0, :scores_argmin]
            tmp_boxes = boxes[0, :scores_argmin]
            tmp_labels = labels[0, :scores_argmin]

            indices = nms(tmp_boxes, tmp_scores, NMS_threshold=0.3, min_certainty=0.5, method=2)

            if indices==():     # No pseudo-labels created
                return annotations_group
            else:
                tmp_boxes = tmp_boxes[indices]
                tmp_labels = tmp_labels[indices]

                gt_boxes = annotations_group['bboxes']
                if np.shape(gt_boxes)[0]>0: # only if there are gt boxes you need to remove detections with gt overlap
                    overlaps = compute_overlap(tmp_boxes.astype(np.float64), gt_boxes.astype(np.float64), iof = np.array([0.0]))
                    new_annotation_mask = np.amax(overlaps, axis=1) < 0.6
                    tmp_boxes = tmp_boxes[new_annotation_mask]
                    tmp_labels = tmp_labels[new_annotation_mask]
                if np.shape(tmp_boxes)[0]>0:
                    annotations_group['bboxes'] = np.concatenate ((annotations_group['bboxes'], tmp_boxes))
                    annotations_group['labels'] = np.concatenate ((annotations_group['labels'], tmp_labels))
        return annotations_group