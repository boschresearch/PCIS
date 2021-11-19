# -*- coding: utf-8 -*-

"""
// Training Object Detectors if Only Large Objects are Labeled
// Copyright (c) 2019 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pickle
import scipy.io

'''
Create pickles for CityPersons given the downloaded CityPersons annotations (anno_train.mat and anno_val.mat).
'''

##############################################################################################
# MODIFY THE PATHES BELOW DEPENDING ON WHICH .PICKLE FILE YOU WANT TO CREATE
###############################################################################################
# Train
mode = 'train'   # train or val
mat = scipy.io.loadmat('./data/anno_train.mat')['anno_train_aligned'][0]
percentage_of_annotations_to_keep = 25   #25 keeps the largest 25% of annotations
make_remaining_annotations_ignore_regions = False   #Set to true for oracle ignore
save_path = './data/citypersons_largest_' + str(percentage_of_annotations_to_keep) + '_percent.pickle'
###############################################################################################
#Validation
#mode = 'val'
#mat = scipy.io.loadmat('./data/anno_val.mat')['anno_val_aligned'][0]
#save_path = './data/cityPersons_val.pickle'
##############################################################################################

citypersons_classes = {
    'ignored':0,
    'person':1,
    'rider':2,
    'sitting_person':3,
    'unusual_person':4,
    'people':5}

labels = {}
for key, value in citypersons_classes.items():
    labels[value] = key


def generate_pickle_from_mat (mat):
    dict_final = {}
           
    for i in mat:
        tmp_dict={}
        tmp_dict['annotations']=[] 
        dict_final[i['im_name'][0][0][0]]['annotations'] = tmp_dict
        for item in i['bbs']:
            for z in range(len(item)):
                tmp_dict2 = {}
                tmp_dict2['pos'] = np.array(item[0][z][1:5], dtype=float)
                tmp_dict2['lbl'] = labels[item[0][z][0]]
                dict_final[i['im_name'][0][0][0]]['annotations'].append(tmp_dict2)
    return dict_final


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_pickle (annotations, out_path):    
    with open(out_path, 'wb') as handle:
        pickle.dump(annotations, handle)
        
        
def find_large_low_occluded_pedestrians(pickle_file, nb_of_pedestrian_annotations_to_keep = 0):
    list_obj_sizes = []
    img_names = list(pickle_file.keys()) 
    for i in img_names:
        for annot in pickle_file[i]['annotations']:
            if annot['lbl']== 'person':
                size = annot['pos'][3]
                list_obj_sizes.append(size)
            
    list_obj_sizes = sorted(list_obj_sizes)[::-1]
    threshold_size = list_obj_sizes[nb_of_pedestrian_annotations_to_keep]  
    return threshold_size
    
                
def find_reasonable_pedestrians (pickle_file, scale):
    if type(scale) != list:
        min_scale = scale
    else:
        min_scale = scale[0]
        
    img_names = list(pickle_file.keys()) 
    reasonable_predictions_tmp = []
    for i in img_names:
        for annot in pickle_file[i]['annotations']:
            if annot['lbl']== 'person' and annot['pos'][3]>=min_scale:
                reasonable_predictions_tmp.append(1)
            else:
                reasonable_predictions_tmp.append(0)  
    
    return reasonable_predictions_tmp


def keep_subset_annotations(pickle_file, subset_for_inclusion, make_remaining_annotations_ignore_regions):
    img_names = list(pickle_file.keys())
    counter = 0
    nb_removed = 0
    nb_kept = 0
    nb_ignored = 0
    for i in img_names:
        tmp_annotations = []
        for annot in pickle_file[i]['annotations']:
            if subset_for_inclusion[counter]== 1:
                tmp_annotations.append(annot)
                nb_kept += 1
            else:
                if make_remaining_annotations_ignore_regions:
                    nb_ignored += 1
                    annot['lbl'] = 'ignored'
                    tmp_annotations.append(annot)
                else:
                    nb_removed += 1
            counter += 1                   
        pickle_file[i]['annotations']  = tmp_annotations
    return pickle_file
            
   
annotations_pickle = generate_pickle_from_mat (mat)
if mode=='train': #12670 is nb of pedestrians with size > 50
    scale = find_large_low_occluded_pedestrians(annotations_pickle, nb_of_pedestrian_annotations_to_keep = int(12670 * percentage_of_annotations_to_keep /100))
    reasonable_pedestrians = find_reasonable_pedestrians (annotations_pickle, scale=scale)
    annotations_pickle = keep_subset_annotations(annotations_pickle, reasonable_pedestrians, make_remaining_annotations_ignore_regions=make_remaining_annotations_ignore_regions)
save_pickle (annotations_pickle, save_path)