#WML

Machine learning tools library.

##Dependencies

- Linux or Mac OS
- Python ≥ 3.6
- scipy
- yacs
- OpenCV
- pycocotools
- gcc & g++ ≥ 4.9
- tensorflow ≥ 1.10

##Installation

```
pip install -r requirements.txt
cd tfop
make clean
make -j18
```

##Prepare datasets

###COCO

1) Extract COCO datasets into the following stracture:

```
coco
├── annotations
├── val2017
└── train2017
```

2) Create tf record for object detection, instance segmentation and semantic segmentation 

```
python datasets_tools/create_coco_tf_record.py --data_dir ~/ai/mldata/coco/
```

3) Create tf record for 2D pose estimation

```
python datasets_tools/create_coco_tf_kp_record.py --data_dir ~/ai/mldata/coco
```

###Generate other datasets

The usage is similar to COCO datasets toolkit.

- Pascal VOC: datasets_tools/create_pascal_voc_tf_record.py
- MOT: datasets_tools/create_mot_tf_record.py
- LabelMe: datasets_tools/create_labelme_tf_record.py


##Change the default buildin datasets location

You can set location for buildin datasets by modify the value of object_detection2/data/datasets/buildin.py:dataset_root_path 

##Train

###Train RetinaNet

```
python object_detection_tools/train_net_on_multi_gpus.py --config-file object_detection2/default_configs/coco/RetinaNetN101.yaml --gpus 0 1 2 3 4 5 6 7
```

###Train Mask-RCNN

```
python object_detection_tools/train_net_on_multi_gpus.py --config-file object_detection2/default_configs/coco/Mask-RCNN-FPN-seph.yaml --gpus 0 1 2 3 4 5 6 7
```

###Train Cascade Mask-RCNN

```
python object_detection_tools/train_net_on_multi_gpus.py --config-file object_detection2/default_configs/coco/Cascade-Mask-RCNN-FPN-N.yaml --gpus 0 1 2 3 4 5 6 7
```

###Train OpenPose

```
python object_detection_tools/train_net_on_multi_gpus.py --config-file object_detection2/default_configs/keypoints/OpenPose-coco.yaml --gpus 0 1 2 3 4 5 6 7
```

###Train HRNet 

```
python object_detection_tools/train_net_on_multi_gpus.py --config-file object_detection2/default_configs/keypoints/OpenPose-coco.yaml --gpus 0 1 2 3 4 5 6 7
```

###Train FairMOT

```
python object_detection_tools/train_net_on_multi_gpus.py --config-file object_detection2/default_configs/MOT/FairMOT.yaml --gpus 0 1 2 3 4 5 6 7
```

###Eval

example

```
python object_detection_tools/eval_net.py --config-file object_detection2/default_configs/coco/RetinaNet.yaml --gpus 0
```

##Predict on images

example

```
python object_detection_tools/predict_on_images.py --test_data_dir ../test_imgs --config-file object_detection2/default_configs/coco/RetinaNet.yaml
```

##MOT Track

```
python object_detection_tools/mot_track.py --config-file object_detection2/default_configs/MOT/FairMOT.yaml --gpus 0
```

## License

WML itself is released under the MIT License (refer to the LICENSE file for details).


##Authors

```
    Wang Jie  bluetornado@zju.edu.cn

    Copyright 2017 The WML Authors.  All rights reserved.
```
