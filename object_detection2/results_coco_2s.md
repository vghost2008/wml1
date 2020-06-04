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
- train time: 29.5h
- eval time: 0.1313s/img

```
```
