import os
import numpy as np
import wml_utils as wmlu
import cv2
import tensorflow as tf

class SNPEEngine(object):
    def __init__(self,model_path,input_size=None,output_names=None,output_shapes=None,output_layers=None,output_suffix=":0.raw"):
        self.output_names = output_names
        self.output_shapes = output_shapes
        self.input_size = input_size
        self.tmp_path = "/tmp/snpe"
        self.bin = os.path.join(os.environ['SNPE_ROOT'],"bin/x86_64-linux-clang/snpe-net-run")
        self.model_path = model_path
        self.output_layers = output_layers
        self.output_suffix = output_suffix
        if self.output_names is not None and self.output_shapes is None:
            self.output_shapes = [None]*len(self.output_names)

    def tf_forward(self,inputs):
        nr_outputs = 1 if self.output_names is None else len(self.output_names)
        output = tf.py_func(self.forward,[inputs],Tout=[tf.float32]*nr_outputs)
        if self.output_shapes is not None:
            res = []
            for x,s in zip(output,self.output_shapes):
                x = tf.reshape(x,s)
                res.append(x)
            return res
        else:
            return output

    def forward(self,inputs):
        if self.tmp_path.startswith("/tmp"):
            wmlu.create_empty_dir(self.tmp_path,remove_if_exists=True,yes_to_all=True)
        print("inputs shape:",inputs.shape)
        raw_path = os.path.join(self.tmp_path,"input.raw")
        input_list = os.path.join(self.tmp_path,"file_list.txt")
        output_dir = os.path.join(self.tmp_path,"output")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(input_list, "w") as f:
            if self.output_layers is not None:
                v = f"#{self.output_layers[0]}"
                for x in self.output_layers[1:]:
                    v += f" {x}"
                v += "\n"
                f.write(v)
            f.write(raw_path)

        self.to_snpe_raw(inputs,raw_path)

        cmd = "{}  --container {} --input_list {} --output_dir {}".format(self.bin, self.model_path,
                                                                          input_list,
                                                                          output_dir)
        print(f"CMD:{cmd}")
        print(f"All output files.")
        os.system(cmd)
        all_files = wmlu.recurse_get_filepath_in_dir(output_dir,suffix=".raw")
        wmlu.show_list(all_files)
        print("-------------------------------")
        res_data = []
        output_dir = "/home/wj/0day/output" #for DEBUG
        if self.output_names is not None:
            for name,shape in zip(self.output_names,self.output_shapes):
                path = os.path.join(output_dir,"Result_0",name+self.output_suffix)
                if not os.path.exists(path):
                    print(f"{path} not exits")
                    res_data.append(None)
                else:
                    print(f"Read from {path}")
                    td = np.fromfile(path,dtype=np.float32)
                    if shape is not None:
                        td = np.reshape(td,shape)
                    res_data.append(td)
        else:
            path = all_files[0]
            print(f"Use ")
            td = np.fromfile(path, dtype=np.float32)
            res_data.append(td)

        return res_data

    def to_snpe_raw(self,data, raw_filepath):
        img_array = np.array(data)  # read it
        if self.input_size is not None:
            print(f"Resize data to {self.input_size}")
            img_array = cv2.resize(img_array, [self.input_size[1],self.input_size[0]])
        # save
        fid = open(raw_filepath, 'wb')
        img_array.tofile(fid)
