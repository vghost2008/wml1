import wml_utils as wmlu
from object_detection2.snpe_toolkit.snpe_engine import SNPEEngine
import numpy as np

dlc_path = wmlu.home_dir("0day/test.dlc")
#snpe = SNPEEngine(dlc_path,output_layers=["shared_head/l2_normalize"])
snpe = SNPEEngine(dlc_path,
                          output_names=['shared_head/ct_regr/Conv_1/BiasAdd','shared_head/heat_ct/Conv_1/BiasAdd',
                                        'shared_head/hw_regr/Conv_1/BiasAdd','shared_head/l2_normalize'],
                          output_layers=["shared_head/l2_normalize/Square", "shared_head/hw_regr/Conv_1/Conv2D",
                                         "shared_head/ct_regr/Conv_1/Conv2D","shared_head/heat_ct/Conv_1/Conv2D"],
                          output_shapes=[[1,135,240,2],[1,135,240,1],[1,135,240,2],[1,135,240,64]])
input = np.random.rand(1,540,960,3)
res = snpe.forward(input.astype(np.float32))
for x in res:
    print(x.shape)

