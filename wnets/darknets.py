import tensorflow as tf
from collections import OrderedDict

slim = tf.contrib.slim

class ResidualBlock(object):
    def __init__(self,chnls,inner_chnls=None,normalizer_fn=None,normalizer_params=None,activation_fn=None):
        if inner_chnls is None:
            inner_chnls = chnls
        self.inner_chnls = inner_chnls
        self.chnls = chnls
        self.normalizer_fn = normalizer_fn
        self.normalizer_params = normalizer_params
        self.activation_fn = activation_fn

    def forward(self,x,scope=None):
        with tf.variable_scope(scope,default_name="unit"):
            out = slim.conv2d(x,self.inner_chnls,
                              1,1,
                              activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.normalizer_params)
            out = slim.conv2d(out,self.chnls,
                              3,1,
                              activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.normalizer_params)
            out = out+x
            #if self.activation_fn is not None:
                #out = self.activation_fn(out)
            return out

class CSPFirst(object):
    def __init__(self,in_chnls,out_chnls,normalizer_fn=None,normalizer_params=None,activation_fn=None):
        self.in_chnls = in_chnls
        self.out_chnls = out_chnls
        self.normalizer_fn = normalizer_fn
        self.normalizer_params = normalizer_params
        self.activation_fn = activation_fn
        self.block = ResidualBlock(out_chnls,out_chnls//2,
                                   normalizer_fn=self.normalizer_fn,
                                   normalizer_params=self.normalizer_params,
                                   activation_fn=self.activation_fn
                                   )

    def forward(self,x,scope=None):
        with tf.variable_scope(scope,"Block"):
            x = slim.conv2d(x,self.out_chnls,
                              3,2,
                              activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.normalizer_params)
            x = self.block.forward(x,scope="unit_0")
            return x

class CSPStem(object):
    def __init__(self,in_chnls,out_chnls,num_block,normalizer_fn=None,normalizer_params=None,activation_fn=None):
        self.in_chnls = in_chnls
        self.out_chnls = out_chnls
        self.normalizer_fn = normalizer_fn
        self.normalizer_params = normalizer_params
        self.activation_fn = activation_fn
        self.block = ResidualBlock(out_chnls//2,
                                   normalizer_fn=self.normalizer_fn,
                                   normalizer_params=self.normalizer_params,
                                   activation_fn=self.activation_fn
                                   )
        self.num_block = num_block

    def forward(self,x,scope=None):
        with tf.variable_scope(scope,default_name="Block"):
            x = slim.conv2d(x,self.out_chnls,
                            3,2,
                            activation_fn=self.activation_fn,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.normalizer_params)
            out_0 = slim.conv2d(x,self.out_chnls//2,
                                1,1,
                                activation_fn=self.activation_fn,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=self.normalizer_params)
            out_1 = slim.conv2d(x,self.out_chnls//2,
                                1,1,
                                activation_fn=self.activation_fn,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=self.normalizer_params)
            for i in range(self.num_block):
                out_1 = self.block.forward(out_1,scope=f"unit_{i}")
            out = tf.concat([out_0,out_1],axis=-1)
            out = slim.conv2d(out,self.out_chnls,
                              1,1,
                              activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.normalizer_params)
            return out

class CSPDarkNet(object):
    def __init__(self,chnls=[32,64, 128, 256, 512, 1024],num_blocks=[None,2,8,8,4],
                 num_classes=None,normalizer_fn=None,normalizer_params=None,activation_fn=None):
        self.chnls = chnls
        self.num_blocks = num_blocks
        self.normalizer_fn = normalizer_fn
        self.normalizer_params = normalizer_params
        self.activation_fn = activation_fn
        self.neck = CSPFirst(self.chnls[0],self.chnls[1],
                             normalizer_fn=self.normalizer_fn,
                             normalizer_params=self.normalizer_params,
                             activation_fn=self.activation_fn)
        self.num_classes = num_classes


    def forward(self,x,scope=None):
        endpoints = OrderedDict()
        with tf.variable_scope(scope,"CSPDarkNet"):
            x = slim.conv2d(x, self.chnls[0],
                            3, 1,
                            activation_fn=self.activation_fn,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.normalizer_params)
            endpoints["C0"] = x
            x = self.neck.forward(x,scope=f"Block0")
            endpoints["C1"] = x
            for i in range(1,len(self.chnls)-1):
                stem = CSPStem(self.chnls[i],self.chnls[i+1],self.num_blocks[i],
                               normalizer_fn=self.normalizer_fn,
                               normalizer_params=self.normalizer_params,
                               activation_fn=self.activation_fn)
                x = stem.forward(x,scope=f"Block{i}")
                endpoints[f"C{i+1}"] = x

            if self.num_classes is not None:
                x = tf.reduce_mean(x,axis=[1,2],keepdims=False)
                x = slim.fully_connected(x,self.num_classes)

            return x,endpoints


