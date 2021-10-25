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

from .coco_for_citypersons import COCO
from .eval_MR_multisetup_citypersons import COCOeval
import os
import json


def write_detections_to_json(detections, json_out_dir):
    result_list = []
    for img_nb in range(len(detections)):
        for cl in detections[img_nb]:
            for det in cl:

                result_list_tmp = {}
                result_list_tmp['image_id'] = img_nb +1
                result_list_tmp['category_id'] = 1
                result_list_tmp['bbox'] = [det[0], det[1], det[2]-det[0], det[3]-det[1]]
                result_list_tmp['score'] = det[4]
                result_list.append(result_list_tmp)

    with open(json_out_dir, 'w') as f:
        json.dump(result_list, f)

def eval_citypersons_miss_rate(detection_json, annType = 'bbox', out_dir = './data/Citypersons_official_eval' ):

    print ('Running demo for *%s* results.'%(annType))

    #initialize COCO ground truth api
    annFile = './data/val_gt.json'
    # initialize COCO detections api
    resFile =  detection_json
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## running evaluation
    res_file = open(out_dir + '/results.txt', 'w')
    for id_setup in range(0,4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        cocoEval.summarize(id_setup,res_file)

    res_file.close()
    print('Evaluation completed. File save to ', out_dir)