#Object detection2 test results

## 结论


##基本配置

- lr=0.02
- warmup steps=1000
- steps = (80000,100000,120000)
- dataset: coco2017

##coco/Mask-RCNN-FPN-C3.yaml 

- batch_size: 4 on 3 gpu
- ROIHeads: Res5ROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 43.1h
- eval time: 0.3724s/img
- ANCHOR SIZES: [[120],[240], [480], [512]] 

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.235
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.363
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.253
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.104
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.396
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.223
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.305
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.309
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.148
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.526
```

###segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.175
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.369
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.158
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.015
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.193
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.384
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.170
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.244
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.247
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.191
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.383

```

##coco/Mask-RCNN-FPN-C3-2.yaml 

- use NonLocal hook
- batch_size: 4 on 3 gpu
- ROIHeads: Res5ROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 42.1h
- eval time: 0.3337s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.274
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.437
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.293
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.019
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.197
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.405
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.256
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.377
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.022
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.332
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.547
```

##segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.220
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.459
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.199
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.030
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.349
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.412
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.329
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.339
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.158
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.348
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.412

```

##coco/Mask-RCNN-FPN-C3-2.yaml 

- use NonLoca hook and FusionBackbone hook
- batch_size: 4 on 3 gpu
- ROIHeads: Res5ROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 43.1h
- eval time: 0.3724s/img
- ANCHOR SIZES: [[120],[240], [480], [512]] 

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.144
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.236
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.152
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.045
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.241
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.165
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.224
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.226
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.071
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.391
```

##segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.247
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.094
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.007
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.132
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.306
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.137
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.193
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.195
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.049
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.130
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.305
```

##coco/Mask-RCNN-FPN-C3-3.yaml 

- use FusionBackbone hook
- batch_size: 4 on 3 gpu
- ROIHeads: Res5ROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 43.1h
- eval time: 0.3724s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 

###bbox

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.108
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.195
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.106
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.052
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.167
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.217
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.334
```

##segm

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.099
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.221
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.079
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.208
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.297
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.145
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.225
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.108
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.296
```

##coco/Mask-RCNN-FPN-3.yaml 

- batch_size: 4 on 3 gpu
- ROIHeads: StandROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 35.9h
- eval time: 0.1702s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.267
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.444
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.284
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.194
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.385
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.249
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.376
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.391
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.346
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.540

```

###segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.252
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.475
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.244
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.473
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.237
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.375
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.184
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.473

```
