##Results
```
- geo_rcnn0) ClsNonLocalROIHeadsHook+FusionBackboneHookV2+SeparateFastRCNNConvFCHeadV2
- geo_rcnn1) NonLocalBackboneHookV2+FusionBackboneHookV2+ClsNonLocalROIHeadsHookV2+SeparateFastRCNNConvFCHeadV2
- geo_rcnn2)
- geo_rcnn3) ClsNonLocalROIHeadsHookV2+FusionBackboneHookV2+SeparateFastRCNNConvFCHeadV2
- geo_rcnn4) NonLocalBackboneHookV2+FusionBackboneHookV2+ClsNonLocalROIHeadsHook+SeparateFastRCNNConvFCHeadV2(1) ClsNonLocalROIHeadsHookV2->ClsNonLocalROIHeadsHook)
- geo_rcnn5) NonLocalBackboneHookV2+FusionBackboneHookV2+ClsNonLocalROIHeadsHookV2+SeparateFastRCNNConvFCHeadV2+KeepRatioPooler
- geo_rcnn6) NonLocalBackboneHookV2+FusionBackboneHookV2+ClsNonLocalROIHeadsHookV2+SeparateFastRCNNConvFCHeadV2+F1
- geo_rcnn7) NonLocalBackboneHookV2+FusionBackboneHookV2+ClsNonLocalROIHeadsHookV2+SeparateFastRCNNConvFCHeadV2+F1+Pool size=9
- geo_rcnn8) NonLocalBackboneHookV2+FusionBackboneHookV2+ClsNonLocalROIHeadsHookV2+SeparateFastRCNNConvFCHeadV2+3x (1) 3x)
- geo_rcnn9) NonLocalBackboneHookV2+FusionBackboneHookV2+ClsNonLocalROIHeadsHookV2+SeparateFastRCNNConvFCHeadV2+F1+Pool size=11
- geo_rcnn10) NonLocalBackboneHookV2+FusionBackboneHookV2+ClsNonLocalROIHeadsHookV2+SeparateFastRCNNConvFCHeadV2+F1+Pool size=11 + KeepRatioPooler
```
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|geo_rcnn0|0.880|0.947|0.947|-1.000|0.793|0.883|0.809|0.907|0.907|-1.000|0.821|0.911|
|geo_rcnn0|0.697|0.743|0.743|-1.000|0.618|0.701|0.690|0.772|0.772|-1.000|0.687|0.775|
|geo_rcnn0 big|0.660|0.746|0.746|-1.000|0.624|0.723|0.533|0.744|0.744|-1.000|0.716|0.790|
|geo_rcnn1|0.504|0.538|0.538|-1.000|0.474|0.505|0.577|0.647|0.647|-1.000|0.583|0.650|
|geo_rcnn1|0.770|0.822|0.822|-1.000|0.581|0.777|0.733|0.822|0.822|-1.000|0.667|0.827|
|geo_rcnn1 big|0.739|0.835|0.835|-1.000|0.685|0.819|0.568|0.796|0.796|-1.000|0.758|0.857|
|geo_rcnn2|0.671|0.713|0.713|-1.000|0.613|0.677|0.681|0.760|0.760|-1.000|0.704|0.764|
|geo_rcnn3|0.852|0.913|0.913|-1.000|0.653|0.859|0.789|0.885|0.885|-1.000|0.722|0.891|
|geo_rcnn3|0.873|0.937|0.937|-1.000|0.768|0.879|0.807|0.903|0.903|-1.000|0.812|0.907|
|geo_rcnn3 big|0.838|0.945|0.945|-1.000|0.813|0.882|0.619|0.875|0.875|-1.000|0.852|0.912|
|geo_rcnn4|0.486|0.512|0.512|-1.000|0.538|0.488|0.572|0.641|0.641|-1.000|0.643|0.642|
|geo_rcnn5|0.477|0.552|0.552|-1.000|0.527|0.476|0.550|0.620|0.620|-1.000|0.606|0.621|
|geo_rcnn6|0.926|0.959|0.959|-1.000|0.836|0.930|0.841|0.944|0.944|-1.000|0.865|0.946|
|geo_rcnn7|0.491|0.503|0.503|-1.000|0.559|0.490|0.576|0.650|0.650|-1.000|0.661|0.649|
|geo_rcnn8|0.948|0.984|0.984|-1.000|0.886|0.952|0.858|0.966|0.966|-1.000|0.904|0.968|
|geo_rcnn8(big)|0.900|0.990|0.990|-1.000|0.889|0.920|0.650|0.926|0.926|-1.000|0.916|0.946|
|geo_rcnn9|0.493|0.508|0.508|-1.000|0.558|0.492|0.577|0.649|0.649|-1.000|0.655|0.649|
|geo_rcnn10|0.837|0.915|0.915|-1.000|0.714|0.840|0.780|0.873|0.873|-1.000|0.765|0.876|

