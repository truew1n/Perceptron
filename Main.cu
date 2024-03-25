#include <iostream>
#include <math.h>

#include "Array.cuh"
#include "CudaArray.cuh"

#include "Parser.cuh"

typedef struct CudaRecord {
    CudaArray<float> Values;
    float ExpectedValue;
} CudaRecord;

__device__ float Sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float Randf()
{
    return ((float) rand()) / RAND_MAX;
}

__global__ void PropagateAllForward(CudaArray<float> Output, CudaArray<CudaRecord> CudaTrainset, CudaArray<float> CudaWeights, float Bias)
{
    uint64_t Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index >= CudaTrainset.Num()) return;

    float WeightedSum = 0.0f;
    CudaRecord CurrentRecord = CudaTrainset.Get(Index);
    CudaArray<float> Current = CurrentRecord.Values;
    for(uint64_t i = 0; i < Current.Num(); ++i) {
        WeightedSum += Current.Get(i) * CudaWeights.Get(i);
    }
    WeightedSum -= Bias;

    Output.Set(Index, Sigmoidf(WeightedSum));
}

__device__ float Squaref(float Value)
{
    return Value * Value;
}

__global__ void CalculateCost(CudaArray<float> Output, CudaArray<CudaRecord> CudaTrainset)
{
    uint64_t Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index >= CudaTrainset.Num()) return;
    
    float NewOutput = Squaref(CudaTrainset.Get(Index).ExpectedValue - Output.Get(Index));
    Output.Set(Index, NewOutput);
}

__global__ void UpdateWeights(CudaArray<float> Output, CudaArray<float> CudaWeights, CudaArray<CudaRecord> CudaTrainset, float LearningRate)
{
    uint64_t Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index >= CudaWeights.Num()) return;

    for(uint64_t i = 0; i < CudaTrainset.Num(); ++i) {
        CudaRecord CurrentRecord = CudaTrainset.Get(i);
        float Error = CurrentRecord.ExpectedValue - Output.Get(i);
        float RateDiff = LearningRate * Error * CurrentRecord.Values.Get(Index);
        CudaWeights.Set(Index, CudaWeights.Get(Index) + RateDiff);
    }
}

uint64_t CalcAOB(uint64_t Value, uint64_t AOT)
{
    return (Value / AOT) + ((Value % AOT) > 0);
}

float MatchLabel(Array<char> CharArray)
{
    if(!strcmp(CharArray.begin(), "Iris-versicolor")) {
        return 0.0f;
    } else if(!strcmp(CharArray.begin(), "Iris-virginica")) {
        return 1.0f;
    }
    return 0.0f;
}

CudaArray<CudaRecord> TrainsetConvert(trainset_t Trainset)
{
    CudaArray<CudaRecord> CudaTrainset = CudaArray<CudaRecord>(Trainset.Records.Num());

    for(uint64_t i = 0; i < Trainset.Records.Num(); ++i) {
        CudaRecord CurrentRecord = {0};
        CurrentRecord.Values = CudaArray<float>::From(&Trainset.Records.Get(i).Values);
        CurrentRecord.ExpectedValue = MatchLabel(Trainset.Records.Get(i).Classname);
        CudaTrainset.Set(i, CurrentRecord);
    }
    return CudaTrainset;
}

int main(void)
{
    trainset_t Trainset = ParseTrainsetFile("iris/perceptron.data");
    trainset_t TestTrainset = ParseTrainsetFile("iris/perceptron.test.data");

    CudaArray<CudaRecord> CudaTrainset = TrainsetConvert(Trainset);

    uint64_t AOT = 64;

    srand(72);
    CudaArray<float> CudaWeights = CudaArray<float>(CudaTrainset.Get(0).Values.Num());
    for(uint64_t i = 0; i < CudaWeights.Num(); ++i) {
        CudaWeights.Set(i, Randf());
    }

    float Bias = Randf();
    float LearningRate = 0.01f;

    CudaArray<float> Output = CudaArray<float>(CudaTrainset.Num());

    float SumedCost = 0.0f;
    for(int i = 0; i < 4000; ++i) {
        PropagateAllForward<<<CalcAOB(CudaTrainset.Num(), AOT), AOT>>>(Output, CudaTrainset, CudaWeights, Bias);
        cudaDeviceSynchronize();

        UpdateWeights<<<1, CudaWeights.Num()>>>(Output, CudaWeights, CudaTrainset, LearningRate);
        cudaDeviceSynchronize();

        // Updating Bias
        for(uint64_t i = 0; i < CudaTrainset.Num(); ++i) {
            Bias -= LearningRate*(CudaTrainset.Get(i).ExpectedValue - Output.Get(i));
        }
    }
    CalculateCost<<<CalcAOB(CudaTrainset.Num(), AOT), AOT>>>(Output, CudaTrainset);
    cudaDeviceSynchronize();

    SumedCost = 0.0f;
    for(float Item : Output) {
        SumedCost += Item;
    }
    SumedCost /= CudaTrainset.Num();
    std::cout << "Cost = " << SumedCost << std::endl;
    
    CudaArray<float>::Free(&CudaWeights);
    CudaArray<float>::Free(&Output);

    for(uint64_t i = 0; i < CudaTrainset.Num(); ++i) {
        CudaArray<float>::Free(&CudaTrainset.Get(i).Values);
    }
    CudaArray<CudaRecord>::Free(&CudaTrainset);
    TrainsetDealloc(&Trainset);
    TrainsetDealloc(&TestTrainset);

    std::cout << "Everything worked fine!" << std::endl;
    return 0;
}