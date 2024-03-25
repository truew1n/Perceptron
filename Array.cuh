#pragma once

#include <iostream>

template<typename T>
class Array {
private:
    T *Data;
    uint64_t Size;
public:
    Array() : Data(nullptr), Size(0) {}
    Array(uint64_t InitSize)
    {
        Data = (T *) calloc(InitSize, sizeof(T));
        if(!Data) {
            std::cerr << "Error: Memory Allocation failed!\n" << std::endl;
        }
        Size = InitSize;
    }
    
    uint64_t Num()
    {
        return Size;
    }

    void Add(T Item)
    {
        T *NewData = (T *) realloc(Data, sizeof(T) * (Size + 1));
        if (!NewData) {
            std::cerr << "Error: Memory Reallocation failed!\n" << std::endl;
            return;
        }
        NewData[Size] = Item;
        Data = NewData;
        Size++;
    }
    

    T Get(uint64_t Index)
    {
        if(Index >= Size) {
            std::cerr << "Error: Index " << Index << " out of bounds for Size: " << Size << std::endl;
        }
        return Data[Index];
    }

    void Set(uint64_t Index, T Item)
    {
        if(Index >= Size) {
            std::cerr << "Error: Index " << Index << " out of bounds for Size: " << Size << std::endl;
        }
        Data[Index] = Item;
    }

    T *GetRef(uint64_t Index)
    {
        if(Index >= Size) {
            std::cerr << "Error: Index out of bounds for Size: " << Size << std::endl;
        }
        return &Data[Index];
    }

    static void Free(Array<T> *Array_)
    {
        if(Array_) {
            if(Array_->Data) {
                free(Array_->Data);
                Array_->Data = nullptr;
            }
            Array_->Size = 0;
        }
    }

    static Array<T> Detach(Array<T> *Array_)
    {
        Array<T> Result = Array<T>();
        if(Array_) {
            Result.Data = Array_->Data;
            Result.Size = Array_->Size;
            Array_->Data = nullptr;
            Array_->Size = 0;
        }
        return Result; 
    }

    T *begin()
    {
        return Data;
    }

    T *end()
    {
        return Data + Size;
    }
};