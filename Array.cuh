#pragma once

#include <iostream>

template<typename T>
class Array {
private:
    T *Data;
    uint64_t Size;
    bool CudaLock;
public:
    Array() : Data(nullptr), Size(0), CudaLock(false) {}
    Array(uint64_t Size);
    
    void Add(T Item);
    T Get(uint64_t Index);
    Array<T> ToCudaArray();
    static void CudaArrayFree(Array<T> *Array);
};