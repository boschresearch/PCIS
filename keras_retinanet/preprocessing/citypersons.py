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

from ..preprocessing.generator import Generator
from ..utils.image import read_image_bgr

import copy
import numpy as np
import pickle

citypersons_classes = {
    'ignored':-1,  # (fake humans, e.g. people on posters, reflections etc.)
    'person':0,
    'rider':1,
    'sitting_person':2,
    'unusual_person':3,
    'people':4}


class CitypersonsGenerator(Generator):
    """

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
        self,
        set_name,
        classes=citypersons_classes,
        annotation_dir = '',
        image_dir = '',
        **kwargs
    ):
        """ Initialize a Citypersons data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            csv_class_file: Path to the CSV classes file.
        """

        self.set_name             = set_name
        self.classes              = classes
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        if self.set_name == 'train':
            self.image_dir = self.image_dir + 'train/'
            with open(annotation_dir,"rb") as f:
                print("Data: Loading train annotations...", annotation_dir)
                self.filename_label_annotations = pickle.load(f)
            self.image_names =  list(self.filename_label_annotations.keys())

        elif self.set_name == 'val':
            self.image_dir = self.image_dir + 'val/'
            with open('./data/cityPersons_val.pickle', "rb") as f:
                print("Data: Loading val annotations...")
                self.filename_label_annotations = pickle.load(f)
            self.image_names =  list(self.filename_label_annotations.keys())

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(CitypersonsGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        For citypersons the size of images is always 1024, 2048, 3 and the data type uint8.
        """
        return float(2048) / float(1024)

    def load_image_return_filename(self, image_index):
        """ Load an image at the image_index.
        """
        path = self.image_names[image_index]
        path  =self.image_dir + path.split('_')[0] + '/'  + path

        return read_image_bgr(path), path

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        path = self.image_names[image_index]
        path  =self.image_dir + path.split('_')[0] + '/'  + path
        return read_image_bgr(path)

    def load_annotations(self, image_index, box_index=None):
        """ Load annotations for an image_index.
        """
        key = self.image_names[image_index]
        annotations_raw = copy.deepcopy(self.filename_label_annotations[key]['annotations'])
        annotations = {'labels': [], 'bboxes': [], 'ignore_region': []}

        for i, object_annot in enumerate(annotations_raw):
            raw_box = object_annot['pos']
            label = self.name_to_label(object_annot['lbl'])

            box = np.zeros((4,))
            box[0] = raw_box[0]
            box[1] = raw_box[1]
            box[2] = raw_box[0] + raw_box[2]
            box[3] = raw_box[1] + raw_box[3]

            if label == 0.0:
                annotations['bboxes'].append(box)
                annotations['labels'].append(label)
            else:
                annotations['ignore_region'].append(box)

        if annotations['ignore_region']==[]:
            annotations['ignore_region'] = np.empty((0, 4))
        else:
            annotations['ignore_region'] = np.asarray(annotations['ignore_region'])

        if annotations['labels'] ==[]:
            annotations['labels']= np.empty(0,)
            annotations['bboxes'] =  np.empty((0,4))

        else:
            annotations['bboxes'] = np.asarray(annotations['bboxes'])
            annotations['labels'] = np.asarray(annotations['labels'])

        return annotations