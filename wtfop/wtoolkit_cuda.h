#pragma once
#ifdef GOOGLE_CUDA
#include <iostream>
#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define CHECK_OK(call) \
{ \
	const cudaError_t error = call; \
	if(error != cudaSuccess) { \
		printf("Error:%s:%d coda: %d, reson: %s\n",__FILE__,__LINE__,error,cudaGetErrorString(error)); \
	} \
}
#define CHECK_CUDA_ERRORS(a) do { \
if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
} \
} while(0)

template<typename GPUTensor,typename CPUTensor>
bool assign_cpu_to_gpu(GPUTensor& lhv,const CPUTensor& rhv)
{
    if(lhv.size() != rhv.size()) {
        std::cout<<"Elements size dosen't match ("<<lhv.size()<<" vs "<<rhv.size()<<")."<<std::endl;
        return false;
    }
    if(sizeof(typename GPUTensor::Scalar) != sizeof(typename CPUTensor::Scalar)) {
        std::cout<<"Scalar size dosen't match ("<<sizeof(typename GPUTensor::Scalar)<<" vs "<<sizeof(typename CPUTensor::Scalar)<<")."<<std::endl;
        return false;
    }
    const auto bytes_size = sizeof(typename CPUTensor::Scalar)*lhv.size();
    CHECK_OK(cudaMemcpy(lhv.data(),rhv.data(),bytes_size,cudaMemcpyHostToDevice));
    return true;
}

template<typename CPUTensor,typename GPUTensor>
bool assign_gpu_to_cpu(CPUTensor& lhv,const GPUTensor& rhv)
{
    if(lhv.size() != rhv.size()) {
        lhv = CPUTensor(rhv.dimensions());
    }
    if(sizeof(typename GPUTensor::Scalar) != sizeof(typename CPUTensor::Scalar)) {
        std::cout<<"Scalar size dosen't match ("<<sizeof(typename GPUTensor::Scalar)<<" vs "<<sizeof(typename CPUTensor::Scalar)<<")."<<std::endl;
        return false;
    }
    const auto bytes_size = sizeof(typename CPUTensor::Scalar)*lhv.size();
    CHECK_OK(cudaMemcpy(lhv.data(),rhv.data(),bytes_size,cudaMemcpyDeviceToHost));
    return true;
}
template<typename GPUTensor0,typename GPUTensor1>
bool assign_gpu_to_gpu(GPUTensor0& lhv,const GPUTensor1& rhv)
{
    if(lhv.size() != rhv.size()) {
        std::cout<<"Elements size dosen't match ("<<lhv.size()<<" vs "<<rhv.size()<<")."<<std::endl;
        return false;
    }
    if(sizeof(typename GPUTensor0::Scalar) != sizeof(typename GPUTensor1::Scalar)) {
        std::cout<<"Scalar size dosen't match ("<<sizeof(typename GPUTensor0::Scalar)<<" vs "<<sizeof(typename GPUTensor1::Scalar)<<")."<<std::endl;
        return false;
    }
    const auto bytes_size = sizeof(typename GPUTensor0::Scalar)*lhv.size();
    CHECK_OK(cudaMemcpy(lhv.data(),rhv.data(),bytes_size,cudaMemcpyDeviceToDevice));
    return true;
}
template<typename Device0,typename Device1,typename Tensor0,typename Tensor1>
struct assign_tensor_imp
{
};
template<typename Tensor0,typename Tensor1>
struct assign_tensor_imp<GPUDevice,CPUDevice,Tensor0,Tensor1>
{
static bool apply(Tensor0& lhv,const Tensor1& rhv)
{
    return assign_cpu_to_gpu(lhv,rhv);
}
};
template<typename Tensor0,typename Tensor1>
struct assign_tensor_imp<CPUDevice,GPUDevice,Tensor0,Tensor1>
{
static bool apply(Tensor0& lhv,const Tensor1& rhv)
{
    return assign_gpu_to_cpu(lhv,rhv);
}
};
template<typename Tensor0,typename Tensor1>
struct assign_tensor_imp<CPUDevice,CPUDevice,Tensor0,Tensor1>{
   static bool apply(Tensor0& lhv,const Tensor1& rhv)
{
    lhv = rhv;
    return true;
}
};
template<typename Tensor0,typename Tensor1>
struct assign_tensor_imp<GPUDevice,GPUDevice,Tensor0,Tensor1> {
    static bool apply(Tensor0& lhv,const Tensor1& rhv)
{
    return assign_gpu_to_gpu(lhv,rhv);
}
};
template<typename Device0,typename Device1,typename Tensor0,typename Tensor1>
bool assign_tensor(Tensor0& lhv,const Tensor1& rhv)
{
    return assign_tensor_imp<Device0,Device1,Tensor0,Tensor1>::apply(lhv,rhv);

}
template<typename Device0,typename Device1,typename Tensor0,typename Tensor1>
bool assign_tensor_chip0(Tensor0& lhv,const Tensor1& rhv,int offset)
{
    auto lhv_data = chip_data(lhv,offset);
    auto bytes_size = rhv.size()*sizeof(typename Tensor1::Scalar);
    CHECK_OK(cudaMemcpy(lhv_data,rhv.data(),bytes_size,cudaMemcpyDeviceToDevice));
    return true;
}
template<typename T>
struct MemBuddy
{
    ~MemBuddy() {
        free_hmem();
        free_dmem();
    }
    void free_hmem()
    {
        if(nullptr != h_data_) {
            delete[] h_data_;
            h_data_ = nullptr;
        }
    }
    void free_dmem()
    {
        if(nullptr != d_data_) {
            cudaFree(d_data_);
            d_data_ = nullptr;
        }
    }
    void init_hmem(size_t size) {
        free_hmem();
        h_data_ = new T[size];
        if((nullptr != d_data_) && (size_ != size)) {
            std::cout<<"Memory size dosen't match."<<std::endl;
        }
        size_ = size;
    }
    void init_dmem(size_t size) {
        free_dmem();
        CHECK_OK(cudaMalloc((T**)&d_data_,sizeof(T)*size));
        if((nullptr != h_data_) && (size_ != size)) {
            std::cout<<"Memory size dosen't match."<<std::endl;
        }
        size_ = size;
    }
    void device_to_host()
    {
        if((d_data_ == nullptr) || (0 == size_))
            return;
        if(nullptr == h_data_)
            init_hmem(size_);
        CHECK_OK(cudaMemcpy(h_data_,d_data_,size_*sizeof(T),cudaMemcpyDeviceToHost));
    }
    void host_to_device()
    {
        if((h_data_ == nullptr) || (0 == size_))
            return;
        if(nullptr == d_data_)
            init_dmem(size_);
        CHECK_OK(cudaMemcpy(d_data_,h_data_,size_*sizeof(T),cudaMemcpyHostToDevice));
    }
    T* h_data_ = nullptr;
    T* d_data_ = nullptr;
    size_t size_ = 0;
};
template<typename T>
struct CudaDelete{
    void operator()(T* p)const{
        if(nullptr != p) 
            cudaFree(p);
    }
};
template<typename T>
using cuda_unique_ptr = std::unique_ptr<T,CudaDelete<T>>;
template<typename T>
 cuda_unique_ptr<T> make_cuda_unique(size_t size=1)
{
    T* p = nullptr;
    CHECK_OK(cudaMalloc((T**)&p,sizeof(T)*size));
    return cuda_unique_ptr<T>(p,CudaDelete<T>());
}
template<typename T>
 cuda_unique_ptr<T> make_cuda_unique(const T* hdata,size_t size=1)
{
    T* p = nullptr;
    CHECK_OK(cudaMalloc((T**)&p,sizeof(T)*size));
    CHECK_OK(cudaMemcpy(p,hdata,sizeof(T)*size,cudaMemcpyHostToDevice));
    return cuda_unique_ptr<T>(p,CudaDelete<T>());
}
template<typename T>
 cuda_unique_ptr<T> make_cuda_unique(unsigned char v,size_t size=1)
{
    T* p = nullptr;
    CHECK_OK(cudaMalloc((T**)&p,sizeof(T)*size));
    CHECK_OK(cudaMemset(p,v,sizeof(T)*size));
    return cuda_unique_ptr<T>(p,CudaDelete<T>());
}
template<typename T>
void show_cuda_data(const T* data,size_t size, int col_nr=20,const std::string& name="cuda_data")
{
    T* h_data = new T[size];
    CHECK_OK(cudaMemcpy(h_data,data,sizeof(T)*size,cudaMemcpyDeviceToHost));
    std::cout<<name<<std::endl;
    for(auto i=0; i<size;) {
        for(auto j=0; (j<col_nr)&&(i<size); ++j,++i) {
            std::cout<<h_data[i]<<" ";
        }
        std::cout<<std::endl;
    }
    delete[] h_data;
}
template<typename T>
void show_host_data(const T* h_data,size_t size, int col_nr=20,const std::string& name="cuda_data")
{
    std::cout<<name<<std::endl;
    for(auto i=0; i<size;) {
        for(auto j=0; (j<col_nr)&&(i<size); ++j,++i) {
            std::cout<<h_data[i]<<" ";
        }
        std::cout<<std::endl;
    }
}
template<typename T>
__device__ void d_show_cuda_data(const T* data,size_t size, int col_nr=20,const char* name="cuda_data")
{
    printf("%s\n",name);
    for(auto i=0; i<size;) {
        for(auto j=0; (j<col_nr)&&(i<size); ++j,++i) {
            printf("%.2f  ",float(data[i]));
        }
        printf("\n");
    }
}
#endif
