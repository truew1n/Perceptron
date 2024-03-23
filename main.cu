#include <iostream>
#include <math.h>

#include "Array.cuh"

__device__ float Sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// __global__ void PropagateForward(float *output, float **input, float *weights, float bias, int32_t input_width, int32_t input_height)
// {
    
// }


int main(void)
{
    uint32_t amount_of_threads = 256;
    uint32_t amount_of_blocks = (250 / amount_of_threads) + ((250 % amount_of_threads) > 0);
    
    Array<float> *floats = new Array<float>();
    
    return 0;
}