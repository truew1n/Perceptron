#pragma once

#include <iostream>
#include <string.h>

#include "Array.cuh"

typedef struct record_t {
    Array<float> Values;
    Array<char> Classname;
} record_t;

typedef struct trainset_t {
    Array<record_t> Records;
} trainset_t;

void TrainsetDealloc(trainset_t *trainset)
{
    for(record_t Record : trainset->Records) {
        Array<float>::Free(&Record.Values);
        Array<char>::Free(&Record.Classname);
    }
    Array<record_t>::Free(&trainset->Records);
}

Array<float> ParseInputLine(char *Str)
{
    Array<float> Result = Array<float>();

    Array<char> StringFloat = Array<char>();

    char c = 0;
    do {
        c = *Str++;
        switch(c) {
            case '\0':
            case ',': {
                StringFloat.Add('\0');

                // Converting string float to float
                float Item = atof(StringFloat.begin());
                
                Result.Add(Item);
                Array<char>::Free(&StringFloat);
                break;
            }
            default: {
                StringFloat.Add(c);
                break;
            }
        }
        
    } while(c != '\0');
    return Result;
}

trainset_t ParseTrainsetFile(const char *Filepath)
{
    FILE *File = fopen(Filepath, "rb");

    trainset_t trainset = {0};

    record_t record = {0};

    Array<float> FloatArray = Array<float>();

    Array<char> CharArray = Array<char>();

    char c = 0;
    do {
        c = fgetc(File);
        switch(c) {
            case ',': {
                CharArray.Add('\0');

                float Item = atof(CharArray.begin());
                
                FloatArray.Add(Item);
                Array<char>::Free(&CharArray);
                break;
            }
            case '\r': {
                size_t cpos = ftell(File);
                if(fgetc(File) != '\n') {
                    fseek(File, cpos, SEEK_SET);
                    break;
                }
                __fallthrough;
            }
            case EOF: {
                if(!CharArray.Num()) return trainset;
                __fallthrough;
            }
            case '\n': {
                CharArray.Add('\0');
                
                record.Values = Array<float>::Detach(&FloatArray);
                record.Classname = Array<char>::Detach(&CharArray);
                
                trainset.Records.Add(record);
                break;
            }
            default: {
                CharArray.Add(c);
                break;
            }
        }
    } while(c != EOF);

    fclose(File);
    return trainset;
}