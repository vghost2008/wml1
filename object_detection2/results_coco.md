#Object detection2 test results

## 结论

- 在batch size较小时EvoNormS比BN更有优势
- 在batch size特别小时BN可能不收敛，但EvoNormS会收敛的更好
- GIOU有稳定的提升
- Stitch有稳定的提升

##基本配置

- lr=0.02 (部分配置可能因为学习率过大而batch size过小没有收敛)
- warmup steps=1000
- steps = (80000,100000,120000)

##coco/RetinaNet.yaml BN

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- FPN normal:GN
- Head normal: BN
- iterator: 120k
- train time: 29.5h
- eval time: 0.1313s/img

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.494
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.275
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.046
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.221
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.366
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.257
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.410
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.408
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.557
```

##coco/RetinaNet.yaml BN

- batch_size: 4 on 1 gpu
- backbone resnet50, FrozenBN
- FPN normal:GN
- Head normal: BN
- iterator: 120k
- train time: 21.2h
- eval time: 0.1349s/img

Unconvergence

##coco/RetinaNet.yaml BN

- batch_size: 2 on 2 gpu
- backbone resnet50, FrozenBN
- FPN normal: BN
- Head normal: BN
- iterator: 120k
- train time: 19.88h
- eval time: 0.1319s/img

Unconvergence

##coco/RetinaNet.yaml GN

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- FPN normal:GN
- Head normal: GN
- iterator: 120k
- train time: 37h
- eval time: 0.1365s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.274
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.494
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.276
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.046
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.218
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.368
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.411
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.439
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.113
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.401
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.561

```

##coco/RetinaNet.yaml EvoNormS

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 36.5h
- eval time: 0.1375s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.278
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.501
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.280
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.218
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.376
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.261
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.413
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.560
```

##coco/RetinaNet.yaml EvoNormS

- batch_size: 4 on 1 gpu
- backbone resnet50, FrozenBN
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 23.5h
- eval time: 0.1392

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.149
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.294
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.134
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.015
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.106
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.205
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.190
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.313
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.330
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.038
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.270
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.448

```

##coco/RetinaNet.yaml EvoNormS

- batch_size: 2 on 2 gpu
- backbone resnet50, FrozenBN
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 16.3h
- eval time: 0.1394s/img

Unconvergence

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 46.1h
- eval time: 0.1364s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.319
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.502
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.339
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.246
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.429
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.469
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.497
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.634
```

##coco/RetinaNet.yaml EvoNormS + GIOU + Stitch

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 66.1h
- eval time: 0.1319/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.323
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.506
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.343
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.060
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.252
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.433
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.474
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.456
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638
```

##coco/RetinaNet.yaml TwoWayFPN+EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- TWFPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 32.1h
- eval time: 0.1365s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.318
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.503
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.337
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.056
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.242
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.292
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.466
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.495
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.131
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
```

##coco/RetinaNet.yaml PConv+EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- PConv normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 27.7h
- eval time: 0.1272s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.287
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.486
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.297
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.217
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.390
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.273
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.433
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.460
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593

```

##coco/RetinaNet.yaml MobileNetV3 + EvoNormS + GIOU

- back bone: MobileNetV3
- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 32.3h
- eval time: 0.1448s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.220
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.120
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.006
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.055
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.181
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.173
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.284
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.019
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.183
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.441
```

##coco/EfficientDet-D0.yaml EvoNormS

- No pretrain weights
- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- BIFPN normal: swish
- Head normal: EvoNormS
- iterator: 120k
- train time: 31.4h
- eval time: 0.1407s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.096
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.197
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.083
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.014
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.072
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.124
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.155
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.260
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.271
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.041
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.218
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.352
```

##coco/EfficientDet-D0.yaml EvoNormS

- resnet-50 backbone
- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- BIFPN normal: swish
- Head normal: EvoNormS
- iterator: 120k
- train time: 32.0h
- eval time: 0.1250s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.235
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.395
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.242
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.176
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.319
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.251
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.409
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.108
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549

```


##coco/RetinaNet.yaml SN + GIOUOutputs

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- FPN normal: SN
- Head normal: SN
- iterator: 120k
- train time: 30.7h
- eval time: 0.1262s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.306
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.492
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.323
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.050
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.414
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.287
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.481
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.115
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616
```

##coco/RetinaNet.yaml SN

- batch_size: 4 on 1 gpu
- backbone resnet50, FrozenBN
- FPN normal: SN
- Head normal: SN
- iterator: 120k
- train time: 19.9h
- eval time: 0.1253

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.188
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.354
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.181
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.025
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.142
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.254
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.215
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.354
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.491
```
