#include <iostream>
#include <math.h>

#include "Array.cuh"
#include "CudaArray.cuh"

#include "Parser.cuh"

typedef struct CudaRecord {
    CudaArray<float> Values;
    float ExpectedValue;
} CudaRecord;

__host__ __device__ float Sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float Randf()
{
    return ((float) rand()) / RAND_MAX;
}

__host__ __device__ float Absf(float x)
{
    return x < 0.0f ? -x : x;
}

__host__ __device__ float Distf (float x0, float x1)
{
    return Absf(x0 - x1);
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

__global__ void TestAll(CudaArray<float> Output, CudaArray<CudaRecord> CudaTrainset, CudaArray<float> CudaWeights, float Bias)
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

    float Y = Sigmoidf(WeightedSum);

    float D0Y = Distf(0.0, Y);
    float D1Y = Distf(1.0, Y);
    if(D0Y < D1Y) {
        Y = 0.0f;
    } else {
        Y = 1.0f;
    }
    Output.Set(Index, CudaTrainset.Get(Index).ExpectedValue == Y);
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

__global__ void UpdateWeights(CudaArray<float> Output, CudaArray<float> CudaWeights, float *Bias, CudaArray<CudaRecord> CudaTrainset, float LearningRate)
{
    uint64_t Index = blockDim.x * blockIdx.x + threadIdx.x;
    if(Index >= CudaWeights.Num()) return;

    for(uint64_t i = 0; i < CudaTrainset.Num(); ++i) {
        CudaRecord CurrentRecord = CudaTrainset.Get(i);
        float Error = CurrentRecord.ExpectedValue - Output.Get(i);
        float RateDiff = LearningRate * Error * CurrentRecord.Values.Get(Index);
        CudaWeights.Set(Index, CudaWeights.Get(Index) + RateDiff);
    }

    if(Index == 0) {
        float totalError = 0.0f;
        for(int i = 0; i < CudaTrainset.Num(); ++i) {
            totalError += CudaTrainset.Get(i).ExpectedValue - Output.Get(i);
        }
        *Bias -= LearningRate * totalError;
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

std::string MatchValue(float Value)
{
    if(Value == 0.0f) {
        return "Iris-versicolor";
    } else if(Value == 1.0f) {
        return "Iris-virginica";
    }
    return "None";
}

CudaArray<CudaRecord> TrainsetConvert(trainset_t Trainset)
{
    CudaArray<CudaRecord> CudaTrainset = CudaArray<CudaRecord>(Trainset.Records.Num());

    for(uint64_t i = 0; i < Trainset.Records.Num(); ++i) {
        CudaRecord CurrentRecord = {0};
        CurrentRecord.Values = CudaArray<float>::From(&Trainset.Records.GetRef(i)->Values);
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

    uint64_t AOT = 256;

    srand(72);
    CudaArray<float> CudaWeights = CudaArray<float>(CudaTrainset.Get(0).Values.Num());
    for(uint64_t i = 0; i < CudaWeights.Num(); ++i) {
        CudaWeights.Set(i, Randf());
    }

    float *Bias;
    cudaMallocManaged(&Bias, sizeof(float));
    *Bias = Randf();
    float LearningRate = 0.01f;

    CudaArray<float> Output = CudaArray<float>(CudaTrainset.Num());

    float SumedCost = 0.0f;
    for(int i = 0; i < 4000; ++i) {
        PropagateAllForward<<<CalcAOB(CudaTrainset.Num(), AOT), AOT>>>(Output, CudaTrainset, CudaWeights, *Bias);
        cudaDeviceSynchronize();

        UpdateWeights<<<1, CudaWeights.Num()>>>(Output, CudaWeights, Bias, CudaTrainset, LearningRate);
        cudaDeviceSynchronize();
    }
    CalculateCost<<<CalcAOB(CudaTrainset.Num(), AOT), AOT>>>(Output, CudaTrainset);
    cudaDeviceSynchronize();

    SumedCost = 0.0f;
    for(float Item : Output) {
        SumedCost += Item;
    }
    SumedCost /= CudaTrainset.Num();
    std::cout << "Cost = " << SumedCost << std::endl;

    CudaArray<float>::Free(&Output);
    

    CudaArray<CudaRecord> CudaTestTrainset = TrainsetConvert(TestTrainset);
    CudaArray<float> TestOutput = CudaArray<float>(CudaTestTrainset.Num());

    TestAll<<<CalcAOB(CudaTestTrainset.Num(), AOT), AOT>>>(TestOutput, CudaTestTrainset, CudaWeights, *Bias);
    cudaDeviceSynchronize();

    float Accuracy = 0.0f;
    for(float Item : TestOutput) {
        Accuracy += Item;
    }
    std::cout << "Accuracy = " << (Accuracy / TestOutput.Num())*100 << "% (" << Accuracy << "/" << TestOutput.Num() << ")" << std::endl;

    
    CudaArray<float>::Free(&TestOutput);
    for(uint64_t i = 0; i < CudaTrainset.Num(); ++i) {
        CudaArray<float>::Free(&CudaTrainset.GetRef(i)->Values);
    }
    CudaArray<CudaRecord>::Free(&CudaTrainset);
    TrainsetDealloc(&Trainset);
    TrainsetDealloc(&TestTrainset);

    std::string InputBuffer = "";
    while(true) {
        std::cout << ">: ";
        std::cin >> InputBuffer;
        if(InputBuffer == "exit") {
            break;
        } else {
            Array<float> InputVector = ParseInputLine((char *) InputBuffer.c_str());
            if(InputVector.Num() != CudaWeights.Num()) continue;

            float WeightedSum = 0.0f;
            for(uint64_t i = 0; i < InputVector.Num(); ++i) {
                WeightedSum += CudaWeights.Get(i) * InputVector.Get(i);
            }
            WeightedSum -= *Bias;

            float Y = Sigmoidf(WeightedSum);

            float D0Y = Distf(0.0, Y);
            float D1Y = Distf(1.0, Y);
            if(D0Y < D1Y) {
                Y = 0.0f;
            } else {
                Y = 1.0f;
            }
            std::cout << "Classification: " << MatchValue(Y) << std::endl;
        }
    }
    CudaArray<float>::Free(&CudaWeights);

    std::cout << "Return reached!" << std::endl;
    return 0;
}