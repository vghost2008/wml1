TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC
nvcc -std=c++11 -c -o deform_conv.cu.o deform_conv.cu.cc -I /usr/local -I /usr/local/cuda-9.0/include/ -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-9.0/lib64/ --expt-relaxed-constexpr
