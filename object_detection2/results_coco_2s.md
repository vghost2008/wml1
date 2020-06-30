#Object detection2 test results

## 结论

- box head使用EnvNormS0性能远远好于BN(在box head为conv的情况下)


##基本配置

- lr=0.02
- warmup steps=1000
- steps = (80000,100000,120000)
- dataset: coco2017

##结果汇总
###coco/Mask-RCNN-FPN-C3.yaml 

- batch_size: 4 on 3 gpu
- ROIHeads: Res5ROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 43.1h
- eval time: 0.3724s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 

####bbox
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|base line(1)|0.235|0.363|0.253|0.000|0.104|0.396|0.223|0.305|0.309|0.000|0.148|0.526|
|use nolocal|0.274|0.437|0.293|0.019|0.197|0.405|0.256|0.377|0.388|0.022|0.332|0.547|
|use FuseBackbone|0.307|0.484|0.329|0.027|0.234|0.448|0.274|0.407|0.419|0.031|0.374|0.581|



####segm
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|base line(1)|0.175|0.369|0.158|0.015|0.193|0.384|0.170|0.244|0.247|0.059|0.191|0.383|
|use nolocal|0.220|0.459|0.199|0.030|0.349|0.412|0.209|0.329|0.339|0.158|0.348|0.412|
|use Fusebackbone|0.241|0.497|0.222|0.033|0.368|0.428|0.218|0.343|0.354|0.168|0.367|0.428|

```
- 1) ANCHOR SIZES: [[120],[240], [480], [512]] 
```

###coco/Mask-RCNN-FPN-3.yaml 

- batch_size: 4 on 3 gpu
- ROIHeads: StandROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 35.9h
- eval time: 0.1702s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 

####bbox
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|base line|0.267|0.444|0.284|0.029|0.194|0.385|0.249|0.376|0.391|0.034|0.346|0.540|
|cosine+EvoNormS0 mask head|0.285|0.470|0.304|0.028|0.213|0.413|0.258|0.386|0.400|0.030|0.358|0.555|
|P3output+cosine|0.283|0.471|0.299|0.031|0.210|0.412|0.258|0.386|0.401|0.035|0.354|0.556|
|1)|0.280|0.468|0.299|0.035|0.214|0.403|0.254|0.385|0.400|0.039|0.359|0.550|
|1) x2|0.285|0.474|0.305|0.032|0.220|0.408|0.255|0.386|0.402|0.036|0.366|0.548|
|2) |0.293|0.486|0.313|0.035|0.219|0.423|0.261|0.394|0.408|0.038|0.370|0.562|
|seph+BN|0.070|0.129|0.068|0.000|0.030|0.109|0.114|0.165|0.168|0.000|0.078|0.267|
|seph+cosine+EnvNormS0 box head|0.296|0.471|0.322|0.030|0.220|0.429|0.267|0.400|0.414|0.032|0.367|0.575|
|sephv2|0.133|0.236|0.133|0.001|0.078|0.203|0.166|0.244|0.250|0.001|0.157|0.382|
|sephv2+cosine+EvoNormS head|0.300|0.479|0.328|0.036|0.223|0.437|0.270|0.406|0.421|0.039|0.376|0.581|





####segm
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|base line|0.252|0.475|0.244|0.029|0.388|0.473|0.237|0.375|0.388|0.184|0.388|0.473|
|cosine+EvoNormS0 mask head|0.262|0.492|0.255|0.030|0.389|0.478|0.239|0.374|0.388|0.182|0.388|0.478|
|P3output+cosine|0.262|0.494|0.252|0.032|0.388|0.482|0.241|0.378|0.391|0.183|0.388|0.482|
|1)|0.260|0.496|0.249|0.034|0.387|0.477|0.237|0.377|0.390|0.193|0.386|0.477|
|1) x2|0.263|0.499|0.251|0.035|0.388|0.476|0.237|0.376|0.389|0.184|0.388|0.476|
|2) |0.269|0.507|0.256|0.034|0.394|0.485|0.241|0.380|0.394|0.187|0.394|0.486|
|seph + BN|0.077|0.159|0.066|0.006|0.184|0.289|0.144|0.212|0.216|0.093|0.183|0.288|
|seph + cosine+EvoNormS0 box head|0.270|0.496|0.267|0.031|0.396|0.493|0.247|0.384|0.398|0.177|0.396|0.493|
|sephv2|0.136|0.272|0.122|0.013|0.246|0.361|0.176|0.269|0.275|0.116|0.245|0.361|
|sephv2+cosine+EvoNormS head|0.274|0.501|0.269|0.031|0.401|0.496|0.248|0.388|0.402|0.183|0.400|0.496|


```
- 1) use nonlocal and fusebackbone+cosine
- 2) use cosine + BalanceBackboneHook + Mask Head EvoNormS0
```

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
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.307
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.484
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.329
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.027
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.234
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.448
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.274
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.407
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.374
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.581
```

##segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.241
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.497
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.222
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.033
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.368
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.218
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.343
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.354
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.168
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.367
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
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

##coco/Mask-RCNN-FPN-3.yaml 

- batch_size: 4 on 3 gpu
- ROIHeads: StandROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: EnvNormS0
- iterator: 120k
- train time: 34.3h
- eval time: 0.1519/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 
- lr decay: cosine

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.285
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.304
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.213
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.386
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.030
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.555
```

###segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.262
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.492
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.255
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.030
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.478
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.374
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.478

```

##coco/Mask-RCNN-FPN-3.yaml

- batch_size: 4 on 3 gpu
- ROIHeads: StandROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0 + BalanceBackboneHook
- iterator: 120k
- train time: 35.2h
- eval time: 0.1685s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 
- lr decay type: cosine

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.293
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.486
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.313
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.219
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.423
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.261
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.394
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.038
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.370
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.562
```

###segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.269
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.507
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.256
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.394
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.241
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.380
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.187
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.394
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.486
```


##coco/Mask-RCNN-FPN-3.yaml + only P3 output

- batch_size: 4 on 3 gpu
- ROIHeads: StandROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- iterator: 120k
- train time: 35.1h
- eval time: 0.1459s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 
- lr decay type: cosine

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.283
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.471
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.299
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.210
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.412
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.386
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.556

```

###segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.262
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.494
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.032
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.241
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.378
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.391
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.183
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
```

##coco/Mask-RCNN-FPN-3.yaml + only F3 output

```
使用Nonlocal与FusionBackbone好像没有什么用处
```

- use NonLocal hook and FusionBackbone hook
- batch_size: 4 on 3 gpu
- ROIHeads: StandROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- iterator: 120k
- train time: 38.3h
- eval time: 0.1547/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 
- lr decay type: cosine

###bbox

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.299
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.550
```

###segm

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.249
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.237
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.477
```

iterator: 240k

###bbox

```
```

###segm

```
```

##coco/Mask-RCNN-FPN-seph.yaml

- batch_size: 4 on 3 gpu
- ROIHeads: StandardROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 37.1h
- eval time: 0.2397s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.070
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.129
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.068
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.030
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.109
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.114
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.165
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.168
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.078
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.267
```

##segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.077
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.159
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.066
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.006
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.184
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.289
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.144
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.212
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.216
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.093
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.183
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.288
```

##coco/Mask-RCNN-FPN-seph.yaml

- batch_size: 4 on 3 gpu
- ROIHeads: StandardROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- BoxHead: EnvNormS0
- MaskHead:EnvNormS0
- iterator: 120k
- train time: 41.5h
- eval time: 0.2679s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 
- lr decay type: cosine

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.471
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.322
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.030
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.220
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.429
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.267
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.032
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.367
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575

```

##segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.270
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.496
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.267
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.396
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.247
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.384
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.398
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.177
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.396
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
```

##coco/Mask-RCNN-FPN-sephv2.yaml

- batch_size: 4 on 3 gpu
- ROIHeads: StandardROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: BN
- iterator: 120k
- train time: 37.9h
- eval time: 0.2397s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.133
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.236
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.133
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.078
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.203
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.166
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.244
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.250
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.157
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.382

```

##segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.136
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.272
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.122
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.013
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.246
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.361
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.176
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.269
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.275
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.245
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.361
```

##coco/Mask-RCNN-FPN-sephv2.yaml

- batch_size: 4 on 3 gpu
- ROIHeads: StandardROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: EnvNormS0
- iterator: 120k
- train time: 34.8h
- eval time: 0.2229s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.479
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.328
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.223
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.437
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.406
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.376
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.581
```

##segm

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.274
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.501
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.269
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.401
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.496
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.248
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.388
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.402
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.183
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.496

```
