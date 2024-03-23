#pragma once

#include <iostream>

template<typename T>
class Array {
private:
    enum Mode : int8_t {
        NORMAL,
        CUDA
    };

    T *Data;
    uint64_t Size;
    Mode CurrentMode; 
public:
    Array() : Data(nullptr), Size(0), CudaLock(false), CurrentMode(Mode::NORMAL) {}
    Array(uint64_t Size);
    
    void Add(T Item);
    T Get(uint64_t Index);
    static Array<T> ToCudaArray(Array<T> *Array_);
    static void CudaArrayFree(Array<T> *Array);
};