#pragma once

#include <iostream>

#include "Array.cuh"

template<typename T>
class CudaArray {
private:
    T *Data;
    uint64_t Size;
public:
    CudaArray(T *Data, uint64_t Size)
    {
        this->Data = Data;
        this->Size = Size;
    }

    CudaArray(uint64_t Size)
    {
        cudaMallocManaged(&Data, sizeof(T) * Size);
        this->Size = Size;
    }

    static CudaArray<T> From(Array<T> *Array_)
    {
        T *CudaData;
        uint64_t DataSize = sizeof(T) * Array_->Num();
        cudaMallocManaged(&CudaData, DataSize);
        cudaMemcpy(CudaData, Array_->begin(), DataSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
        CudaArray<T> CudaArray(CudaData, Array_->Num());

        return CudaArray;
    }

    static void Free(CudaArray<T> *CudaArray_)
    {
        if(CudaArray_) {
            if(CudaArray_->Data) {
                cudaFree(CudaArray_->Data);
                CudaArray_->Data = nullptr;
            }
            CudaArray_->Size = 0;
        }
    }

    __host__ __device__ uint64_t Num()
    {
        return Size;
    }

    __host__ __device__ T Get(uint64_t Index)
    {
        if(Index >= Size) {
            return Data[0];
            printf("Index %llu out of bounds!\n", Index);
        }
        return Data[Index];
    }

    __host__ __device__ T *GetRef(uint64_t Index)
    {
        if(Index >= Size) {
            return &Data[0];
            printf("Index %llu out of bounds!\n", Index);
        }
        return &Data[Index];
    }

    __host__ __device__ void Set(uint64_t Index, T Item)
    {
        if(Index >= Size) return;
        Data[Index] = Item;
    }

    __host__ __device__ T *begin()
    {
        return Data;
    }

    __host__ __device__ T *end()
    {
        return Data + Size;
    }
};