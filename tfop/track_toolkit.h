#pragma once
#include <algorithm>

namespace MOT
{
    template<typename T>
       T l2_normalize(const T& v) {
           return v/v.norm();
       }
    template<typename Countainer,typename Indexs>
        void inplace_gather(Countainer& data,Indexs idxs) {
            Countainer res;
            for(auto i:idxs)
                res.emplace_back(std::move(data[i]));
            std::swap(data,res);
        }
    template<typename Countainer,typename Indexs,typename Pred>
        void inplace_gather(Countainer& data,Indexs idxs,Pred fn) {
            Countainer res;
            for(auto i:idxs) {
                if(fn(data[i]))
                    res.emplace_back(std::move(data[i]));
            }
            std::swap(data,res);
        }
    template<typename Countainer,typename Indexs>
        Countainer gather(Countainer& data,Indexs idxs) {
            Countainer res;
            for(auto i:idxs)
                res.emplace_back(data[i]);
            return res;
        }
    template<typename Countainer,typename Indexs,typename Pred>
        Countainer gather(Countainer& data,Indexs idxs,Pred fn) {
            Countainer res;
            for(auto i:idxs) {
                if(fn(data[i]))
                    res.emplace_back(data[i]);
            }
            return res;
        }
    template<typename T>
        T square(const T& v) {
            T res = v.cwiseProduct(v);
            return res;
        }
}
