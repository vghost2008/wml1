#TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_INC = $(shell /usr/local/anaconda3/bin/python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_INC = $(shell /home/wj/anaconda3_cpu/bin/python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_INC = "/home/vghost/.local/lib/python3.6/site-packages/tensorflow/include"
#TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_LIB = $(shell /usr/local/anaconda3/bin/python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#TF_LIB = $(shell /home/wj/anaconda3_cpu/bin/python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#TF_LIB = "/home/vghost/.local/lib/python3.6/site-packages/tensorflow"
#USE_CUDA := "YES"
#all: build
#all: clean build
#g++ -std=c++14 -shared -g -o $@ $^ -I $(TF_INC) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -L${TF_LIB} -ltensorflow_framework 
#CXXFLAGS = -I ${TF_INC} -fPIC -L${TF_LIB} -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI
CXX11_ABI=1
CXXFLAGS = -std=c++14 -Wno-ignored-attributes -Wno-unused-result -shared -g -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI} -fPIC -L${TF_LIB} -ltensorflow_framework `pkg-config --cflags opencv` `pkg-config --libs opencv`
#CUDAFLAGS = -std=c++14 -g -shared -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA --expt-relaxed-constexpr -Xcompiler="-fPIC"
#CUDAFLAGS = -std=c++14 -g -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA --expt-relaxed-constexpr -Xcompiler="-fPIC" -arch=sm_30
CUDAFLAGS = -std=c++14 -g -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI} -DGOOGLE_CUDA --expt-relaxed-constexpr -Xcompiler="-fPIC" -gencode arch=compute_60,code=compute_60

build: wtfop.so

#wtfop.so: wtfop.cc bboxes.cc wtoolkit.cc nlp.cc dentalfilm.cc medical.cc
ifdef USE_CUDA
targets = mask.o wtfop.o bboxes.o wtoolkit.o nlp.o medical.o gn.o anchor_bboxes.o bboxes_encode_decode.o nms.o bboxes_encode_decode_gpu.o center_net_encode_decode.o opencvadapter.o matcher.o matcher_gpu.o
CXXFLAGS += -DGOOGLE_CUDA -DEIGEN_USE_GPU -DEIGEN_USE_THREADS
CXXFLAGS += -I/usr/local/cuda-9.0/include 
else
targets = mask.o wtfop.o bboxes.o wtoolkit.o nlp.o medical.o gn.o anchor_bboxes.o bboxes_encode_decode.o nms.o center_net_encode_decode.o opencvadapter.o matcher.o
endif

bboxes_encode_decode_gpu.o:bboxes_encode_decode_gpu.cu
	nvcc -c $^ -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 ${CUDAFLAGS}

matcher_gpu.o:matcher_gpu.cu
	nvcc -c $^ -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 ${CUDAFLAGS}

wtfop.so:${targets}
	echo $CXXFLAGS
	echo $^
	g++ -std=c++14 -Wno-ignored-attributes -Wno-unused-result -shared -g -o $@ $^ -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI} -fPIC -L${TF_LIB} -ltensorflow_framework  `pkg-config --cflags opencv` `pkg-config --libs opencv`

clean:
	rm -f *.o *.so *.pyc *.npy
