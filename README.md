#WML

Machine learning tools library.

##Dependencies
- Linux or Mac OS
- Python ≥ 3.6
- torchvision
- OpenCV
- pycocotools
- gcc & g++ ≥ 4.9
- tensorflow ≥ 1.10
- [wtfop](https://github.com/vghost2008/wtfop) custom tensorflow op.

##Object detection2
Inspired by [detectron2](https://github.com/facebookresearch/detectron2)

Train example
```
python object_detection_tools/train_net_on_multi_gpus.py --config-file object_detection2/default_configs/Base-Mask-RCNN-FPN-C4.yaml
```

Eval example
```
python object_detection_tools/eval_net.py --config-file object_detection2/default_configs/Base-Mask-RCNN-FPN-C4.yaml
```

##Authors

```
    Wang Jie  bluetornado@zju.edu.cn

    Copyright 2017 The WML Authors.  All rights reserved.
```
