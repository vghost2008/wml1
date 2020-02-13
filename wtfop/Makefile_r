TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_INC = "/home/vghost/.local/lib/python3.6/site-packages/tensorflow/include"
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#TF_LIB = "/home/vghost/.local/lib/python3.6/site-packages/tensorflow"
all: build
#all: clean build
#g++ -std=c++14 -shared -g -o $@ $^ -I $(TF_INC) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -L${TF_LIB} -ltensorflow_framework 
#CXXFLAGS = -I ${TF_INC} -fPIC -L${TF_LIB} -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI
CXXFLAGS = -std=c++14 -Wno-ignored-attributes -Wno-unused-result -shared -g -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -L${TF_LIB} -ltensorflow_framework 

build: wtfop.so

#wtfop.so: wtfop.cc bboxes.cc wtoolkit.cc nlp.cc dentalfilm.cc medical.cc
wtfop.so: wtfop.o bboxes.o wtoolkit.o nlp.o dentalfilm.o medical.o
	echo $^
	g++ -std=c++14 -Wno-ignored-attributes -Wno-unused-result -shared -g -o $@ $^ -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -L${TF_LIB} -ltensorflow_framework 

clean:
	rm -f *.o *.so *.pyc *.npy
