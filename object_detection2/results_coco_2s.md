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
|sephv2+cosine+EvoNormS head+pred iou1|0.306|0.478|0.329|0.031|0.225|0.447|0.271|0.406|0.419|0.034|0.369|0.584|
|sephv10(shared head)|0.304|0.486|0.328|0.036|0.226|0.440|0.269|0.400|0.414|0.045|0.371|0.566|
|sephv9(retinanet rpn)|0.300|0.474|0.323|0.039|0.221|0.427|0.265|0.386|0.396|0.059|0.340|0.537|
|3) |0.306|0.479|0.324|0.055|0.219|0.433|0.279|0.415|0.430|0.092|0.393|0.559|
|4) |0.314|0.502|0.332|0.039|0.228|0.456|0.276|0.406|0.421|0.056|0.375|0.571|


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
|sephv2+cosine+EvoNormS head+pred iou1|0.281|0.506|0.282|0.031|0.396|0.497|0.250|0.388|0.400|0.178|0.395|0.498|



```
- 1) use nonlocal and fusebackbone+cosine
- 2) use cosine + BalanceBackboneHook + Mask Head EvoNormS0
- 3) sephv2+cosine+EvoNormS0+ASTTMatcher+class loss linear scalear+new rcn sample+p2 output
- 4) sephv2+cosine+EvoNormS0+ASTTMatcher3+new rcn sample

###bbox

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.306
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.479
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.324
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.219
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.433
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.279
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.415
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.430
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.092
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.559

```

##coco/Mask-RCNN-FPN-sephv10.yaml

- batch_size: 4 on 3 gpu
- ROIHeads: StandardROIHeads
- backbone resnet50, FrozenBN
- FPN normal: EnvNormS0
- Head normal: EnvNormS0
- iterator: 120k
- train time: 30.0h
- eval time: 0.1687s/img
- ANCHOR SIZES: [[64],[128], [256], [512]] 
- new rcn sample

###bbox
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.304
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.486
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.328
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.226
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.440
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.269
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.371
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.566

```
