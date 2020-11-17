#Object detection2 test results

## 结论

- box head使用EnvNormS0性能远远好于BN(在box head为conv的情况下)


##基本配置

- lr=0.02
- warmup steps=1000
- steps = (80000,100000,120000)
- dataset: coco2017

##结果汇总

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
|MaskRCNNFPNCOCODemon-sephNv15|0.385|0.582|0.417|0.156|0.383|0.497|0.317|0.486|0.507|0.290|0.515|0.603|
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
|4) + cls loss weight (sephv14)|0.315|0.503|0.334|0.047|0.230|0.458|0.280|0.407|0.420|0.062|0.374|0.572|
|4) + cls loss weight + roi nr=512(sephv14)|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|
|4) + cls loss weight + roi nr=512 + 1 anchor(sephv14_1)|0.325|0.522|0.350|0.067|0.313|0.486|0.281|0.407|0.418|0.108|0.440|0.587|
|5) + cls loss weight + roi nr=512 + 1 anchor(sephv14_2)|0.326|0.519|0.349|0.071|0.315|0.488|0.279|0.407|0.418|0.107|0.440|0.588|
|4) + cls loss weight +RPN evo normal(sephv17)|0.314|0.501|0.337|0.050|0.231|0.453|0.279|0.405|0.417|0.063|0.371|0.569|
|4) + cls loss weight + no balance sample(sephv16)|0.311|0.495|0.332|0.045|0.229|0.453|0.275|0.401|0.411|0.056|0.355|0.569|
|4) + cls loss weight + no neg balance sample(sephv18)|0.314|0.506|0.334|0.047|0.231|0.451|0.277|0.407|0.420|0.061|0.374|0.570|
|4) + cls loss weight + no pos balance sample(sephv18_1)|0.315|0.494|0.341|0.040|0.227|0.460|0.277|0.402|0.413|0.053|0.356|0.571|
|4) + cls loss weight + multiscalepooler(sephv19)|0.310|0.497|0.329|0.047|0.226|0.450|0.273|0.401|0.414|0.063|0.367|0.564|
|4)+GIOUoutputs|0.301|0.498|0.319|0.042|0.215|0.441|0.266|0.390|0.403|0.055|0.352|0.555|
|4)+GIOUoutputs+reg loss x4 (sephv15)|0.316|0.501|0.335|0.047|0.229|0.460|0.278|0.408|0.421|0.059|0.365|0.577|
|4)+GIOUoutputs+reg loss x8 (sephv15_1)|0.322|0.498|0.343|0.049|0.233|0.469|0.287|0.420|0.434|0.066|0.383|0.594|
|4)+GIOUoutputs+reg loss x12 (sephv15_2)|0.317|0.488|0.339|0.046|0.224|0.464|0.285|0.418|0.432|0.063|0.379|0.591|
|4)+GIOUoutputs+no cls loss weight|0.295|0.493|0.308|0.042|0.207|0.437|0.263|0.389|0.403|0.055|0.347|0.559|
|4)+GIOUoutputs+reg loss x8 + low box reg threshold(sephv21)|0.299|0.466|0.321|0.041|0.207|0.437|0.275|0.398|0.409|0.059|0.352|0.562|
|4) + cls loss weight+p3 output (sephv22)|0.313|0.500|0.331|0.043|0.236|0.455|0.277|0.405|0.418|0.058|0.377|0.571|
|Cascade|0.338|0.501|0.361|0.051|0.252|0.485|0.294|0.434|0.449|0.069|0.404|0.613|
|gev1|0.327|0.515|0.349|0.066|0.314|0.491|0.283|0.413|0.424|0.106|0.449|0.595|


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

##Effect of input size
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4)+GIOUoutputs+reg loss x8 (sephv15_1)|0.322|0.498|0.343|0.049|0.233|0.469|0.287|0.420|0.434|0.066|0.383|0.594|
|MaskRCNNFPNCOCODemon-sephNv15|0.385|0.582|0.417|0.156|0.383|0.497|0.317|0.486|0.507|0.290|0.515|0.603|

##Effect of roi nr
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight (sephv14)|0.315|0.503|0.334|0.047|0.230|0.458|0.280|0.407|0.420|0.062|0.374|0.572|-|
|4) + cls loss weight + roi nr=512(sephv14)|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|1080ti 0.3069|
|Cascade+roi nr=128|0.338|0.501|0.361|0.051|0.252|0.485|0.294|0.434|0.449|0.069|0.404|0.613|-|
|Cascade+roi nr=512|0.357|0.519|0.387|0.079|0.337|0.535|0.303|0.441|0.454|0.129|0.470|0.633|titan 0.1136|

##Effect of neck
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight + roi nr=512(sephv14)|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|1080ti 0.3069|
|4) + cls loss weight + roi nr=512(bifpn v1)|0.330|0.511|0.354|0.077|0.315|0.485|0.283|0.409|0.421|0.120|0.439|0.580|1080ti 0.2742|
|4) + cls loss weight + roi nr=512(wtwfpn v1)|0.326|0.514|0.353|0.066|0.312|0.492|0.280|0.407|0.419|0.101|0.438|0.588|titan 0.2636|
|4) + cls loss weight + roi nr=512 + wsum (wtwfpn v1_1)|0.326|0.513|0.351|0.060|0.311|0.493|0.282|0.406|0.418|0.094|0.438|0.589|titan 0.0946|
|4) + cls loss weight + roi nr=512 + wsum (wtwfpn v1_1)+NRPNT|0.328|0.518|0.353|0.062|0.313|0.497|0.285|0.414|0.427|0.102|0.451|0.597|titan 0.3328|
|7) + cls loss weight + roi nr=512(sephv14_10) + 3 anchor|0.336|0.539|0.359|0.086|0.324|0.486|0.284|0.425|0.440|0.146|0.463|0.590|-|
|7) + cls loss weight + roi nr=512(sephv14_21) + FPN channel=384+3 anchor|0.337|0.538|0.368|0.095|0.325|0.492|0.289|0.429|0.444|0.154|0.466|0.597|titan 0.1488|
|7) + cls loss weight + roi nr=512(bifpnv2) + 3 anchor|0.351|0.551|0.382|0.113|0.339|0.498|0.293|0.444|0.461|0.190|0.476|0.602|titan 0.1725|
|7) + cls loss weight + roi nr=512(bifpnv2_1) + 3 anchor+FusionBackboneHook+OneHeadNonLocalROIHeadsHook+NonLocalBackboneHook|0.354|0.555|0.379|0.119|0.346|0.500|0.294|0.445|0.462|0.198|0.481|0.604|titan 0.1717|
|7) + cls loss weight + roi nr=512(sephv14_32) + 3 anchor+OneHeadNonLocalROIHeadsHook+NonLocalBackboneHook+FusionBackboneHook|0.347|0.551|0.376|0.099|0.339|0.498|0.292|0.439|0.457|0.177|0.482|0.602|titan 0.1848|




##Effect of neck fusion
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight + roi nr=512(sephv14)(sum)|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|1080ti 0.3069|
|4) + cls loss weight + roi nr=512(v14_8)(mix_fusion)|0.330|0.520|0.355|0.071|0.318|0.493|0.284|0.417|0.430|0.118|0.455|0.597|1080ti 0.3738|
|4) + cls loss weight + roi nr=512(wtwfpn v1)|0.326|0.514|0.353|0.066|0.312|0.492|0.280|0.407|0.419|0.101|0.438|0.588|titan 0.2636|
|4) + cls loss weight + roi nr=512 + wsum (wtwfpn v1_1)|0.326|0.513|0.351|0.060|0.311|0.493|0.282|0.406|0.418|0.094|0.438|0.589|titan 0.0946|

##Effect of neck hook
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight + roi nr=512(sephv14)|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|1080ti 0.3069|
|balance hook|0.330|0.517|0.356|0.065|0.320|0.494|0.282|0.410|0.422|0.104|0.446|0.589|titan 0.2829|
|balance hookV2|0.329|0.518|0.352|0.069|0.315|0.488|0.283|0.412|0.424|0.111|0.445|0.589|titan 0.3068|
|4) + cls loss weight + roi nr=512(bifpn v1)|0.330|0.511|0.354|0.077|0.315|0.485|0.283|0.409|0.421|0.120|0.439|0.580|1080ti 0.2742|
|4) + cls loss weight + roi nr=512 + balance hook(bifpn v1_1)|0.331|0.511|0.356|0.082|0.320|0.483|0.282|0.413|0.425|0.136|0.445|0.581|1080ti 0.2723|
|7) + cls loss weight + roi nr=512(sephv14_10) + 3 anchor|0.336|0.539|0.359|0.086|0.324|0.486|0.284|0.425|0.440|0.146|0.463|0.590|-|
|7) + cls loss weight + roi nr=512(sephv14_13) + 3 anchor + Nonlocal hook|0.340|0.503|0.365|0.072|0.326|0.505|0.294|0.436|0.450|0.133|0.478|0.615|titan 0.1286|
|7) + cls loss weight + roi nr=512(sephv14_26) + 3 anchor+SEBackboneHook|0.337|0.543|0.365|0.093|0.327|0.493|0.286|0.427|0.442|0.158|0.463|0.596|


##Effect of giououtputs
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) |0.314|0.502|0.332|0.039|0.228|0.456|0.276|0.406|0.421|0.056|0.375|0.571|
|4)+GIOUoutputs|0.301|0.498|0.319|0.042|0.215|0.441|0.266|0.390|0.403|0.055|0.352|0.555|
|4)+GIOUoutputs+reg loss x4 (sephv15)|0.316|0.501|0.335|0.047|0.229|0.460|0.278|0.408|0.421|0.059|0.365|0.577|
|4)+GIOUoutputs+reg loss x8 (sephv15_1)|0.322|0.498|0.343|0.049|0.233|0.469|0.287|0.420|0.434|0.066|0.383|0.594|
|4)+GIOUoutputs+reg loss x12 (sephv15_2)|0.317|0.488|0.339|0.046|0.224|0.464|0.285|0.418|0.432|0.063|0.379|0.591|

##Effect of cls loss weight
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) |0.314|0.502|0.332|0.039|0.228|0.456|0.276|0.406|0.421|0.056|0.375|0.571|
|4) + cls loss weight (sephv14)|0.315|0.503|0.334|0.047|0.230|0.458|0.280|0.407|0.420|0.062|0.374|0.572|
|4)+GIOUoutputs+no cls loss weight|0.295|0.493|0.308|0.042|0.207|0.437|0.263|0.389|0.403|0.055|0.347|0.559|
|4)+GIOUoutputs|0.301|0.498|0.319|0.042|0.215|0.441|0.266|0.390|0.403|0.055|0.352|0.555|

##Effect of roi bboxes sample
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight (sephv14)|0.315|0.503|0.334|0.047|0.230|0.458|0.280|0.407|0.420|0.062|0.374|0.572|
|4) + cls loss weight + no balance sample(sephv16)|0.311|0.495|0.332|0.045|0.229|0.453|0.275|0.401|0.411|0.056|0.355|0.569|
|4) + cls loss weight + no neg balance sample(sephv18)|0.314|0.506|0.334|0.047|0.231|0.451|0.277|0.407|0.420|0.061|0.374|0.570|
|4) + cls loss weight + no pos balance sample(sephv18_1)|0.315|0.494|0.341|0.040|0.227|0.460|0.277|0.402|0.413|0.053|0.356|0.571|

##Effect of matcher
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|sephv2+cosine+EvoNormS head|0.300|0.479|0.328|0.036|0.223|0.437|0.270|0.406|0.421|0.039|0.376|0.581|
|4) + cls loss weight + no balance sample(sephv16)|0.311|0.495|0.332|0.045|0.229|0.453|0.275|0.401|0.411|0.056|0.355|0.569|
|4) + cls loss weight + roi nr=512(sephv14) + 3 anchor|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|
|4) + cls loss weight + roi nr=512 + 1 anchor(sephv14_1)|0.325|0.522|0.350|0.067|0.313|0.486|0.281|0.407|0.418|0.108|0.440|0.587|
|5) + cls loss weight + roi nr=512 + 1 anchor(sephv14_2)|0.326|0.519|0.349|0.071|0.315|0.488|0.279|0.407|0.418|0.107|0.440|0.588|
|9) + cls loss weight + roi nr=512(sephv14_29) + 3 anchor|0.323|0.515|0.352|0.063|0.314|0.482|0.278|0.407|0.419|0.099|0.446|0.586|
|7) + cls loss weight + roi nr=512(sephv14_10) + 3 anchor|0.336|0.539|0.359|0.086|0.324|0.486|0.284|0.425|0.440|0.146|0.463|0.590|
|7) + cls loss weight + roi nr=512(sephv14_9) + 1 anchor|0.331|0.538|0.358|0.089|0.322|0.479|0.283|0.425|0.441|0.159|0.461|0.592|
|8) + cls loss weight + roi nr=512(sephv14_11) + 1 anchor|0.326|0.523|0.354|0.065|0.316|0.486|0.280|0.410|0.423|0.103|0.451|0.591|


##Effect of RCNN output layers number.
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight (sephv14)|0.315|0.503|0.334|0.047|0.230|0.458|0.280|0.407|0.420|0.062|0.374|0.572|
|4) + cls loss weight+p3 output (sephv22)|0.313|0.500|0.331|0.043|0.236|0.455|0.277|0.405|0.418|0.058|0.377|0.571|

##Effect of head
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight + roi nr=512(sephv14)|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|1080ti 0.3069|
|4) + cls loss weight + roi nr=512 + sephv5(sephv25)|0.321|0.511|0.342|0.059|0.308|0.484|0.279|0.404|0.415|0.098|0.437|0.583|1080ti 0.0998|
|4) + cls loss weight + roi nr=512 + sephv6(sephv25_1)|0.329|0.516|0.354|0.062|0.315|0.493|0.283|0.410|0.422|0.099|0.443|0.592|1080ti 0.2807|
|7) + cls loss weight + roi nr=512 + sephv7(sephv25_2)|0.335|0.539|0.362|0.086|0.323|0.487|0.287|0.427|0.442|0.154|0.460|0.594|
|4) + cls loss weight + roi nr=512 + nonlocal head hook (sephv14_7)|0.328|0.514|0.352|0.065|0.312|0.492|0.283|0.415|0.427|0.107|0.455|0.594|titan 0.1693|
|4) + cls loss weight + roi nr=512 + nonlocal head hook(no gamma scale) (sephv14_7)|0.328|0.516|0.355|0.066|0.314|0.491|0.283|0.413|0.426|0.109|0.451|0.591|
|7) + cls loss weight + roi nr=512(sephv14_10) + 3 anchor|0.336|0.539|0.359|0.086|0.324|0.486|0.284|0.425|0.440|0.146|0.463|0.590|-|
|7) + cls loss weight + roi nr=512(sephv14_14) + 3 anchor+OneHeadNonLocalROIHeadsHook|0.341|0.542|0.372|0.088|0.326|0.497|0.289|0.435|0.451|0.154|0.471|0.608|-|
|7) + cls loss weight + roi nr=512(sephv14_15) + 3 anchor+ClsNonLocalROIHeadsHook|0.337|0.542|0.364|0.094|0.326|0.488|0.286|0.432|0.448|0.166|0.471|0.597|titan 0.1538|
|7) + cls loss weight + roi nr=512(sephv14_16) + 3 anchor+BoxNonLocalROIHeadsHook|0.338|0.540|0.367|0.091|0.324|0.491|0.289|0.434|0.451|0.164|0.472|0.604|titan 0.1534|
|7) + cls loss weight + roi nr=512(sephv14_17) + 3 anchor+BoxNonLocalROIHeadsHookV2|0.340|0.543|0.371|0.088|0.330|0.494|0.289|0.433|0.449|0.153|0.476|0.602|titan 0.1743|
|7) + cls loss weight + roi nr=512(sephv14_18) + 3 anchor+OneHeadNonLocalROIHeadsHookV3|0.336|0.543|0.365|0.085|0.325|0.490|0.287|0.432|0.448|0.160|0.469|0.603|titan 0.1734|
|7) + cls loss weight + roi nr=512(sephv14_19) + 3 anchor+OneHeadNonLocalROIHeadsHook+NonLocalBackboneHook|0.342|0.550|0.371|0.090|0.331|0.496|0.288|0.435|0.453|0.160|0.477|0.604|titan 0.1745|
|7) + cls loss weight + roi nr=128(sephv14_20_1) + 3 anchor|0.315|0.509|0.335|0.069|0.298|0.465|0.278|0.416|0.433|0.134|0.454|0.586|1080ti 0.1467|
|7) + cls loss weight + roi nr=512(sephv14_20) + 3 anchor|0.335|0.535|0.362|0.075|0.327|0.490|0.284|0.422|0.436|0.131|0.457|0.593|titan 0.1165|
|7) + cls loss weight + roi nr=512(sephv14_24) + 3 anchor+SEROIHeadsHook|0.334|0.536|0.362|0.085|0.323|0.482|0.285|0.424|0.440|0.155|0.463|0.587|1080ti 0.1509|
|7) + cls loss weight + roi nr=512(sephv14_25) + 3 anchor+OneHeadSEROIHeadsHook|0.336|0.537|0.367|0.089|0.327|0.490|0.287|0.427|0.442|0.147|0.464|0.597|1080ti 0.1510|
|7) + cls loss weight + roi nr=512(sephv14_27) + 3 anchor+SEROIHeadsHook+SEBackboneHook|0.337|0.547|0.359|0.088|0.326|0.489|0.287|0.437|0.457|0.169|0.478|0.610|titan 0.1288|
|7) + cls loss weight + roi nr=512(sephv14_22) + 3 anchor+box conv dim=1024|0.339|0.537|0.368|0.083|0.329|0.495|0.290|0.429|0.443|0.146|0.465|0.599|titan 0.3034|
|7) + cls loss weight + roi nr=512(sephv14_23) + 3 anchor+box conv dim=1024+FPN channels=384|0.343|0.543|0.372|0.090|0.331|0.497|0.291|0.432|0.447|0.152|0.468|0.601|titan 0.3136|
|7) + cls loss weight + roi nr=512(sephv14_28) + 3 anchor+OneHeadCBAMROIHeadsHook+SEBackboneHook|0.338|0.544|0.367|0.086|0.326|0.489|0.288|0.438|0.456|0.159|0.484|0.606|titan 0.1345|
|7) + cls loss weight + roi nr=512(sephv14_31) + 3 anchor+box conv dim=1024+FPN channels=384+OneHeadNonLocalROIHeadsHook+NonLocalBackboneHook|0.345|0.547|0.374|0.098|0.335|0.496|0.289|0.435|0.450|0.166|0.473|0.598|titan 0.3218|
|7) + cls loss weight + roi nr=512(sephv14_32) + 3 anchor+OneHeadNonLocalROIHeadsHook+NonLocalBackboneHook+FusionBackboneHook|0.347|0.551|0.376|0.099|0.339|0.498|0.292|0.439|0.457|0.177|0.482|0.602|titan 0.1848|




##Effect of RCNN box transform
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight (sephv14)|0.315|0.503|0.334|0.047|0.230|0.458|0.280|0.407|0.420|0.062|0.374|0.572|
|4) + cls loss weight + offset encode(sephv24)|0.317|0.499|0.336|0.065|0.296|0.482|0.279|0.410|0.423|0.116|0.441|0.594|

##Effect of pred iou and centerness
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|sephv2+cosine+EvoNormS head|0.300|0.479|0.328|0.036|0.223|0.437|0.270|0.406|0.421|0.039|0.376|0.581|-|
|sephv2+cosine+EvoNormS head+pred iou1|0.306|0.478|0.329|0.031|0.225|0.447|0.271|0.406|0.419|0.034|0.369|0.584|-|
|4) + cls loss weight + roi nr=512(sephv14) + 3 anchor|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|1080ti 0.3069|
|4) + cls loss weight + roi nr=512(sephv14_6) + 3 anchor+pred centerness|0.332|0.512|0.361|0.065|0.315|0.500|0.284|0.409|0.420|0.099|0.439|0.595|titan 0.0976|
|4) + cls loss weight + roi nr=512(sephv14_6) + 3 anchor+pred centerness + NRPNT|0.334|0.514|0.363|0.064|0.316|0.501|0.284|0.414|0.426|0.104|0.446|0.600|titan 0.1551|

##Effect of rpn threshold
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight + roi nr=512(sephv14_6) + 3 anchor+pred centerness|0.332|0.512|0.361|0.065|0.315|0.500|0.284|0.409|0.420|0.099|0.439|0.595|titan 0.0976|
|4) + cls loss weight + roi nr=512(sephv14_6) + 3 anchor+pred centerness + NRPNT|0.334|0.514|0.363|0.064|0.316|0.501|0.284|0.414|0.426|0.104|0.446|0.600|titan 0.1551|
|4) + cls loss weight + roi nr=512 + wsum (wtwfpn v1_1)|0.326|0.513|0.351|0.060|0.311|0.493|0.282|0.406|0.418|0.094|0.438|0.589|titan 0.0946|
|4) + cls loss weight + roi nr=512 + wsum (wtwfpn v1_1)+NRPNT|0.328|0.518|0.353|0.062|0.313|0.497|0.285|0.414|0.427|0.102|0.451|0.597|titan 0.3328|
|4) + cls loss weight + roi nr=512(sephv14) + 3 anchor|0.331|0.518|0.359|0.067|0.314|0.493|0.284|0.412|0.423|0.117|0.443|0.588|1080ti 0.3069|
|4) + cls loss weight + roi nr=512(sephv14) + 3 anchor+NRPNT|0.332|0.519|0.360|0.066|0.317|0.495|0.285|0.417|0.429|0.121|0.454|0.593|titan 0.2044|
|4) + cls loss weight + roi nr=512(wtwfpn v1)|0.326|0.514|0.353|0.066|0.312|0.492|0.280|0.407|0.419|0.101|0.438|0.588|titan 0.2636|
|4) + cls loss weight + roi nr=512(wtwfpn v1)+NRPNT|0.328|0.517|0.354|0.066|0.315|0.494|0.282|0.413|0.426|0.107|0.450|0.593|titan 0.2027|
|4) + cls loss weight + roi nr=512 + sephv5(sephv25)|0.321|0.511|0.342|0.059|0.308|0.484|0.279|0.404|0.415|0.098|0.437|0.583|1080ti 0.0998|
|4) + cls loss weight + roi nr=512 + sephv5(sephv25)+NRPNT|0.323|0.513|0.344|0.062|0.309|0.486|0.281|0.409|0.422|0.105|0.448|0.587|1080ti 0.3940|


##Effect of rpn hook
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight + roi nr=512(sephv14) + 3 anchor+NRPNT|0.332|0.519|0.360|0.066|0.317|0.495|0.285|0.417|0.429|0.121|0.454|0.593|titan 0.2044|
|4) + cls loss weight + roi nr=512 + rpn balance hook (sephv26) + 3 anchor+NRPNT|0.327|0.515|0.349|0.066|0.314|0.491|0.283|0.413|0.424|0.106|0.449|0.595|1080ti 0.3572|

##Effect of rpn 
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4) + cls loss weight + roi nr=512(sephv14) + 3 anchor+NRPNT|0.332|0.519|0.360|0.066|0.317|0.495|0.285|0.417|0.429|0.121|0.454|0.593|titan 0.2044|
|4) + cls loss weight + roi nr=512 + RPN evo normal (sephv14_12) + 3 anchor+NRPNT|0.325|0.512|0.348|0.068|0.311|0.487|0.281|0.407|0.418|0.112|0.439|0.589|1080ti 0.2895|


##Cascade rcnn
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Cascade+roi nr=128|0.338|0.501|0.361|0.051|0.252|0.485|0.294|0.434|0.449|0.069|0.404|0.613|-|
|Cascade+roi nr=512|0.357|0.519|0.387|0.079|0.337|0.535|0.303|0.441|0.454|0.129|0.470|0.633|titan 0.1136|
|Cascade+roi nr=512+min iou threshold=0.4|0.342|0.503|0.366|0.076|0.319|0.515|0.297|0.436|0.449|0.140|0.461|0.623|titan 0.039|

##IOU predict
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|pearsonr|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|roi nr=128 + iou version 0 + fc x2 iou1|0.315|0.502|0.336|0.065|0.292|0.476|0.278|0.408|0.421|0.116|0.441|0.585|0.640|
|roi nr=128 + iou version 0 + conv x4 iou2|0.314|0.502|0.335|0.061|0.294|0.474|0.277|0.405|0.419|0.107|0.437|0.584|0.640|
|roi nr=128 + iou version 4 + conv x4 iou4|0.314|0.502|0.336|0.063|0.294|0.472|0.278|0.407|0.419|0.106|0.439|0.585|0.683|
|roi nr=128 + iou version 4 + conv x4 + Multi pool+IouNonLocal iou5|0.315|0.500|0.338|0.060|0.294|0.471|0.279|0.405|0.418|0.113|0.439|0.581|0.656|
|roi nr=128 + iou version 4 + conv x4 IouNonLocal|0.315 iou6|0.502|0.339|0.065|0.296|0.473|0.277|0.408|0.420|0.109|0.441|0.587|0.677|
|roi nr=128 + iou version 4 + conv x4 iou7|0.314|0.499|0.335|0.063|0.295|0.473|0.278|0.405|0.418|0.110|0.441|0.583|0.742|


##Effect or lr
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|7) + cls loss weight + roi nr=512(sephv14_10) + 3 anchor + lr=0.02|0.336|0.539|0.359|0.086|0.324|0.486|0.284|0.425|0.440|0.146|0.463|0.590|
|7) + cls loss weight + roi nr=512(sephv14_30) + 3 anchor + lr=0.01|0.336|0.540|0.367|0.091|0.327|0.489|0.284|0.421|0.435|0.146|0.453|0.588|
|7) + cls loss weight + roi nr=512(sephv14_30) + 3 anchor + lr=0.001|0.307|0.526|0.326|0.074|0.310|0.449|0.263|0.392|0.405|0.114|0.434|0.558|



##Effect of Mask
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|7) + cls loss weight + roi nr=512(sephv14_19) + 3 anchor+OneHeadNonLocalROIHeadsHook+NonLocalBackboneHook|0.342|0.550|0.371|0.090|0.331|0.496|0.288|0.435|0.453|0.160|0.477|0.604|titan 0.1745|
|7) + cls loss weight + roi nr=512(sephv30_1) + 3 anchor+OneHeadNonLocalROIHeadsHook+NonLocalBackboneHook+mask on|0.349|0.550|0.379|0.098|0.338|0.502|0.293|0.444|0.461|0.176|0.487|0.611|titan 0.1647|



#Mask
|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|time cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|7) + cls loss weight + roi nr=512(sephv30_1) + 3 anchor+OneHeadNonLocalROIHeadsHook+NonLocalBackboneHook+mask on|0.311|0.571|0.307|0.083|0.460|0.502|0.262|0.413|0.428|0.272|0.460|0.502|titan 0.1647|


```
- 1) use nonlocal and fusebackbone+cosine
- 2) use cosine + BalanceBackboneHook + Mask Head EvoNormS0
- 3) sephv2+cosine+EvoNormS0+ASTTMatcher+class loss linear scalear+new rcn sample+p2 output
- 4) sephv2+cosine+EvoNormS0+ASTTMatcher3+new rcn sample
- 5) sephv2+cosine+EvoNormS0+ASTTMatcher4+new rcn sample
- 6) NRPNT:表示RPN不设置threshold, 默认rpn threshold=0.005
- 7) sephv2+cosine+EvoNormS0+ASTTMatcher+new rcn sample
- 8) sephv2+cosine+EvoNormS0+DynamicMatcher+new rcn sample
- 9) sephv2+cosine+EvoNormS0+Matcher+new rcn sample
```
