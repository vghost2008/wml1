#pragma once
#include<string>
#include <chrono>
#include <sstream>
#include <iostream>
 
inline void default_log_func(const std::string& v)
 {
     std::cout<<v<<std::endl;
 }
 class WTimeThis
{
    public:
        WTimeThis(const std::string& name,std::function<void(const std::string&)> func=default_log_func,bool autolog=true)
            :name_(name),func_(func),t_(std::chrono::steady_clock::now()),autolog_(autolog){}
        ~WTimeThis() {
            if(autolog_)
                log();
        }
        inline int time_duration()const {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_).count();
        }
        inline void log()const {
            std::stringstream ss;
            ss<<name_<<":"<<time_duration()<<" milliseconds.";
            func_(ss.str());
        }
    private:
        const std::string name_;
        std::function<void(const std::string&)> func_;
        const std::chrono::steady_clock::time_point t_;
        bool autolog_ = false;
};
template<typename T>
size_t dims_prod(const T& v,int execlude_dim=0)
{
    size_t res = 1;
    for(auto i=0; i<v.dimensions().size(); ++i)
        if(i!=execlude_dim)
            res *= v.dimension(i);
   return res;
}
template<typename T>
typename T::Scalar* chip_data(const T& v,int offset)
{
    return v.data()+offset*dims_prod(v,0);
}
//#define _TT_

#ifdef _TT_
#define TIME_THIS() WTimeThis tt__(std::string(__func__)+":"+std::to_string(__LINE__)+":"+__FILE__)
#define TIME_THISV1(x) WTimeThis tt__(x)
#else
#define TIME_THIS() 
#define TIME_THISV1(x) 
#endif
#define CTIME_THIS() WTimeThis tt__(std::string(__func__)+":"+std::to_string(__LINE__)+":"+__FILE__)
#define CTIME_THISV1(x) WTimeThis tt__(x)

