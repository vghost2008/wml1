#Object detection2 test results

## 结论

- 在batch size较小时EvoNormS比BN更有优势
- 在batch size特别小时BN可能不收敛，但EvoNormS会收敛的更好
- GIOU有稳定的提升
- Stitch有稳定的提升
- cosine lr decay 有明显提升
- dropblock有稳定轻微提升
- WAA+Stitch有明显稳定提升
- BalanceBackboneHook有明显提升
- 仅将retinanet每一层anchor的顺序改变一下性能就有明显的下降
- 仅将retinanet使用等间隔的anchor size性能出现明显下降

##基本配置

- dataset: coco2014
- GPU: NVIDIA 1080TI
- lr=0.02 (部分配置可能因为学习率过大而batch size过小没有收敛)
- warmup steps=1000
- steps = (80000,100000,120000)
- input size for train: (512,544,576,608,640)
- input size for eval: 576

##结果汇总

###bbox
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|RetinaNet+BN+FPN GN|0.273|0.494|0.275|0.046|0.221|0.366|0.257|0.410|0.440|0.124|0.408|0.557|
|RetinaNet+GN+FPN GN|0.274|0.494|0.276|0.046|0.218|0.368|0.258|0.411|0.439|0.113|0.401|0.561|
|1)-GIOUOutput|0.278|0.501|0.280|0.048|0.218|0.376|0.261|0.413|0.442|0.119|0.406|0.560|
|1) |0.319|0.502|0.339|0.057|0.246|0.429|0.297|0.469|0.497|0.136|0.453|0.630|
|1)+Stitch|0.323|0.506|0.343|0.060|0.252|0.433|0.299|0.474|0.502|0.143|0.456|0.638|
|1)+cosine|0.333|0.524|0.354|0.058|0.260|0.447|0.302|0.473|0.502|0.130|0.460|0.640|
|1)+cosine+ATSSMatcher2|0.328|0.533|0.345|0.073|0.253|0.438|0.298|0.466|0.496|0.176|0.455|0.620|
|1)+AA+cosine|0.337|0.527|0.360|0.059|0.268|0.453|0.303|0.479|0.507|0.137|0.469|0.646|
|1)+FPN DB+cosine|0.335|0.525|0.358|0.060|0.262|0.454|0.302|0.476|0.504|0.142|0.463|0.641|
|1)+FPN DB+cosine+unordered anchor ratio|0.326|0.518|0.347|0.057|0.253|0.439|0.297|0.468|0.495|0.124|0.454|0.633|
|1)+FPN DB+cosine+等间隔anchor size|0.299|0.464|0.317|0.029|0.198|0.438|0.280|0.432|0.448|0.065|0.352|0.633|
|1)+FPN DB+cosine+等比例搜索anchor|0.333|0.528|0.353|0.071|0.241|0.458|0.300|0.475|0.505|0.177|0.433|0.648|
|1)+FPN DB+cosine+BalanceBackboneHook|0.341|0.533|0.362|0.065|0.270|0.457|0.304|0.481|0.509|0.146|0.466|0.647|
|1)+FPN DB+cosine+BalanceBackboneHook+GIOUOutputs|0.342|0.534|0.363|0.065|0.266|0.463|0.307|0.483|0.513|0.152|0.467|0.655|
|1)+FPN DB+cosine+BalanceBackboneHook+GIOUOutputsV2|0.342+bs=3x3|0.527|0.365|0.053|0.269|0.468|0.304|0.480|0.509|0.122|0.470|0.656|
|1)+FPN DB+cosine+BalanceBackboneHook+ATSS2+1anchor+bs=3x3|0.295|0.480|0.307|0.060|0.206|0.415|0.285|0.431|0.461|0.151|0.369|0.613|
|1)+FPN DB+cosine+BalanceBackboneHook+ATSS3+bs=3x3|0.323|0.526|0.338|0.072|0.253|0.433|0.297|0.462|0.493|0.169|0.447|0.624|
|1)+FPN DB+cosine+BalanceBackboneHook+ATSS4+bs=3x3|0.329|0.533|0.346|0.072|0.257|0.440|0.299|0.465|0.495|0.161|0.451|0.625|
|1)+FPN DB+cosine+BalanceBackboneHook+ATSS5+bs=3x3|0.324|0.512|0.341|0.055|0.253|0.446|0.297|0.449|0.473|0.096|0.407|0.632|
|1)+FPN DB+cosine+BalanceBackboneHook+ATSS3+9anchor+bs=3x3|0.315|0.504|0.329|0.058|0.247|0.430|0.294|0.447|0.471|0.107|0.400|0.630|
|1)+FPN DB+cosine+BalanceBackboneHook+ATSS3+3anchor+bs=3x3|0.322|0.524|0.335|0.074|0.253|0.430|0.297|0.463|0.495|0.162|0.446|0.625|
|1)+FPN DB,WSum+cosine|0.334|0.526|0.356|0.061|0.262|0.450|0.302|0.475|0.504|0.147|0.462|0.643|
|2)|0.339|0.531|0.362|0.063|0.270|0.455|0.304|0.481|0.510|0.145|0.473|0.648|
|EfficientDet-D0|0.329|0.558|0.347|0.049|0.259|0.433|0.284|0.443|0.470|0.117|0.419|0.588|
|FCOS|0.332|0.529|0.354|0.081|0.268|0.451|0.291|0.455|0.476|0.141|0.444|0.608|


```
1) RetinaNet+EvoNormS0+FPN EvoNormS0+GIOUOutput
2) 1+AA+Stitch+cosine
```

###CenterNet

|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|max pool|0.127|0.241|0.124|0.001|0.041|0.202|0.148|0.210|0.215|0.002|0.097|0.350|
|max poolv2|0.124|0.235|0.121|0.001|0.043|0.198|0.147|0.206|0.210|0.002|0.095|0.343| 
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

## coco/RetinaNet.yaml BN

- batch_size: 2 on 2 gpu
- backbone resnet50, FrozenBN
- FPN normal: BN
- Head normal: BN
- iterator: 120k
- train time: 19.88h
- eval time: 0.1319s/img

Unconvergence

## coco/RetinaNet.yaml GN

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

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 33.6h
- eval time: 0.1364s/img
- lr decay: cosine

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.524
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.354
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.058
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.260
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.447
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.473
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.130
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.460
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
```

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS + dropblock
- Head normal: EvoNormS
- iterator: 120k
- train time: 31.3h
- eval time: 0.1340s/img
- lr decay: cosine

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.060
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.142
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.641
```

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS + dropblock
- Head normal: EvoNormS
- iterator: 120k
- train time: 31.3h
- eval time: 0.1340s/img
- lr decay: cosine
- ASPECT_RATIOS: [[0.5,1,2],[1,2,0.5],[2,0.5,1],[2,1,0.5],[1,0.5,2]]

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.326
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.518
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.439
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
```

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS + dropblock
- Head normal: EvoNormS
- iterator: 120k
- train time: 31.1h
- eval time: 0.1332s/img
- lr decay: cosine
- ANCHOR SIZES: [[32.0, 66.29, 100.57], [134.86, 169.14, 203.43], [237.71, 272.0, 306.29], [340.57, 374.86, 409.14], [443.43, 477.71, 512.0]] 

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.299
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.464
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.317
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.432
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.065
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.352
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633

```

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS + dropblock
- Head normal: EvoNormS
- iterator: 120k
- train time: 32.5h
- eval time: 0.1383s/img
- lr decay: cosine
- ANCHOR SIZES: [[10.57, 21.97, 37.15], [56.87, 81.08, 109.32], [141.3, 176.59, 214.98], [256.63, 302.07, 351.56], [406.23, 470.85, 550.6]]
- ANCHOR ASPECT RATIO:  [[0.29, 0.77, 2.14]]
- 搜索的等比例anchor

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.528
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.353
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.241
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.458
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.300
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.177
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.433
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648
```

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS + dropblock + BalanceBackboneHook
- Head normal: EvoNormS
- iterator: 120k
- train time: 30.5h
- eval time: 0.1352/img
- lr decay: cosine

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.341
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.362
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.065
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.270
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.457
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.481
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.509
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.466
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.647
```

##coco/RetinaNet.yaml EvoNormS + GIOU

```
FPN中使用wsum好像没有太多用处
```

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS + dropblock
- FPN: wsum fusion
- Head normal: EvoNormS
- iterator: 120k
- train time: 35.9h
- eval time: 0.1389/img
- lr decay: cosine

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.334
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.061
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.147
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.643
```

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 54.5h
- eval time: 0.1419s/img
- DA: WAA+Stitch
- lr decay: cosine

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.339
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.362
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.063
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.270
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.455
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.481
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.510
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.473
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648
```

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 59.5h
- eval time: 0.1335/img
- DA: WAA
- lr decay: cosine

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.527
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.360
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.268
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.453
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.479
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.507
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.137
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.469
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.646

```

##coco/RetinaNet.yaml EvoNormS + GIOU

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- FPN normal: EvoNormS
- Head normal: EvoNormS
- iterator: 120k
- train time: 32.2h
- eval time: 0.1364s/img
- DA: AA

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.308
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.493
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.327
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.052
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.234
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.418
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.292
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.459
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.487
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.625
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

### inputsize=640

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.323
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.513
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.346
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.238
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.425
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.478
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.436
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630
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

- batch_size: 4 on 3 gpu
- backbone: FrozenBN
- Loss: GIOU loss
- BIFPN normal: swish
- Head normal: EvoNormS
- iterator: 120k
- lr decay: consin
- train time: 39.2h
- eval time: 0.1328/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.558
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.347
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.049
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.259
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.433
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.284
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.443
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.470
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.419
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.588
```

##coco/EfficientDet-D0.yaml+resnet50+EvoNormS

- resnet-50 backbone
- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- Loss: GIOU loss
- BIFPN: channel=96, repeat=3
- BIFPN normal: swish
- Head normal: EvoNormS
- Head conv num: 3
- iterator: 120k
- lr decay: consin
- train time: 28.8h
- eval time: 0.1497s/img

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.294
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.508
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.307
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.051
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.228
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.380
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.266
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.428
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
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

##coco/FCOS2.yaml

- batch_size: 4 on 3 gpu
- backbone resnet50, FrozenBN
- FPN normal: EvoNormS0
- Head normal: EvoNormS0
- iterator: 120k
- train time: -
- eval time: 0.0753

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.332
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.529
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.354
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.081
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.268
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.451
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.291
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.476
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.141
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.444
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.608

```
