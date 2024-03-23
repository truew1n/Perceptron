#include "Array.cuh"

template<typename T>
Array<T>::Array(uint64_t InitSize)
{
    Data = (T *) malloc(sizeof(T) * InitSize);
    if(!Data) {
        std::cerr << "Error: Memory Allocation failed in Array(uint64_t Size)!\n" << std::endl;
    }
    Size = InitSize;
    CudaLock = false;
}


template<typename T>
void Array<T>::Add(T Item)
{
    if(CudaLock) return;


}

template<typename T>
T Array<T>::Get(uint64_t Index)
{

}

template<typename T>
Array<T> Array<T>::ToCudaArray()
{
    T *CudaData;
    cudaMalloc(&CudaData, sizeof(T) * Array_->Size);
    Array<T> CudaArray = *Array_;

    return nullptr;
}

template<typename T>
static void Array<T>::CudaArrayFree(Array<T> *Array_)
{

}