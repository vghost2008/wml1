#Object detection2 test results

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
- train time: 
- eval time: 

```
```
