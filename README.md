# Training Object Detectors if Only Large Objects are Labeled

This repository contains an implementation of the method called PCIS, i.e.,
“**P**seudo-labels, output **C**onsistency across scales, and an anchor
scale-dependent **I**gnore **S**trategy”. This implementation is the companion
code for the system reported in the paper:  
“Training Object Detectors if Only Large Objects are Labeled” by Daniel
Pototzky et al. (Accepted at BMVC 2021).  
In case of questions, please contact daniel.pototzky@de.bosch.com.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor
monitored in any way.

## Installation

The code was tested using the following modules:

```text
tensorflow-gpu==1.10.0
Keras==2.2.4
cython
h5py
matplotlib
keras-resnet==0.1.0
opencv-python>=3.3.0
pillow
progressbar2
scipy
```

Our code is based on the RetinaNet implementation available at [https://github.com/fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)

## Experiments CityPersons

For preprocessing annotations you need to download the [full annotations](https://bitbucket.org/shanshanzhang/citypersons/src/default/annotations/) and then run `.data/CityPersons_preparation.py`. The output is a pickle file which is used in training.
Furthermore, you need to download the [validation ground truth](https://bitbucket.org/shanshanzhang/citypersons/src/default/evaluation/) (val_gt.json) as used in the official evaluation code and store it in ./data

Train images are expected to be stored in `/image_folder/train/city_name/img_name.png`  
Val images are expected to be stored in `/image_folder/val/city_name/img_name.png`

For running experiments on 25% of annotations, please use the commands below.

### PCIS

`python -u ./keras_retinanet/bin/train.py --use_adaptive_image_size=True --consistency_with_pseudolabel=True --ignore_stages_without_bbox=True --base_annotation_dir=.data/citypersons_largest_25_percent.pickle   --image-min-side=1024 --image-max-side=2048  --lr=3e-5 --image_dir=/path/to/images/  --snapshot-path=./results/exp_name/  citypersons`

### Standard Training

`python -u ./keras_retinanet/bin/train.py --use_adaptive_image_size=True --base_annotation_dir=.data/citypersons_largest_25_percent.pickle --image-min-side=1024 --image-max-side=2048 --lr=3e-5 --image_dir=/path/to/images/  --snapshot-path=./results/exp_name/  citypersons`

### Oracle Ignore

`python -u ./keras_retinanet/bin/train.py --use_adaptive_image_size=True --base_annotation_dir=.data/citypersons_largest_25_percent_with_ignore_regions.pickle   --image-min-side=1024 --image-max-side=2048 --lr=3e-5 --image_dir=/path/to/images/  --snapshot-path=./results/exp_name/  citypersons`

### Data Distillation

First, conduct standard training.

Then, run `python -u ./keras_retinanet/bin/augmentations_predictions.py --image_dir=/path/to/image_folder/ --save_path_predictions=/path/for/saving/predictions/ --base_annotation_dir=citypersons_largest_25_percent.pickle  citypersons /path/to/model.h5`
to create predictions for multiple augmentations of every image.

Afterwards, generate pseudo-labels using Bounding Box Voting

`python ./keras_retinanet/bin/pseudo_labels_bounding_box_voting.py --predictions_dir=/path/of/predictions/ --out_dir=/path/for_saving_pseudolabels/bbox.pickle --base_annotation_dir=/path/to/citypersons_largest_25_percent.pickle citypersons`

Finally, rerun standard training using the combination of ground truth annotations and pseudo-labels.

### Hyperparameters

For 50%, 25%, and 10% of annotations, learning rate should be set to 3e-5.
For 5%, 1% and 0.5% of annotations, learning rate should be set to 1e-5.

## License

PCIS is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
