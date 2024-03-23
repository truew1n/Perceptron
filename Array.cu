#include "Array.cuh"

template<typename T>
Array<T>::Array(uint64_t InitSize)
{
    Data = (T *) malloc(sizeof(T) * InitSize);
    if(!Data) {
        std::cerr << "Error: Memory Allocation failed in Array(uint64_t Size)!\n" << std::endl;
    }
    Size = InitSize;
    CurrentMode = Mode::NORMAL;
}


template<typename T>
void Array<T>::Add(T Item)
{
    if(CurrentMode != Mode::NORMAL) return;


}

template<typename T>
T Array<T>::Get(uint64_t Index)
{

}

template<typename T>
static Array<T> Array<T>::ToCudaArray(Array<T> *Array_)
{
    T *CudaData;
    int64_t DataSize = sizeof(T) * Array_->Size
    cudaMalloc(&CudaData, DataSize);
    cudaMemcpy(CudaData, Array_->Data, DataSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
    Array<T> CudaArray = *Array_;
    CudaArray.Data = CudaData;
    CudaArray.CurrentMode = Mode::CUDA;

    return CudaArray;
}

template<typename T>
static void Array<T>::CudaArrayFree(Array<T> *Array_)
{
    if(Array_->CurrentMode == Mode::CUDA) return;
    
}