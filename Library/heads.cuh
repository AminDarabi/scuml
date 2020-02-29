#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <stdlib.h>

#include <curand.h>
#include "cublas_v2.h"


//threads per block
#define TPB 128

// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess) 
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }

//scuml


// //Network
// #define Network std::vector<float *>
// Network Network_new(std::vector<int>& arch);
// //float Network_BP(Network net, std::vector<int>& arch, float * X, float * Y, int nums);
// float * Network_FF(Network net, float * X, std::vector<int>& arch, int nums);
// void Network_Del(Network & net);
// void Network_Product(Network & Ra, Network & Rb, Network & A, Network & B, std::vector<int>& arch, float mut);
// __global__ void Rand_select(float * child_A, float * child_B, float * prnt_A, float * prnt_B, float * vecs, float MUT, int sz);
// void Network_ass(Network & A, Network & B, std::vector<int>& arch);
// void Network_FF_fast(std::vector<float *>& A, Network & net, std::vector<int>& arch, int nums);


//utility
std::vector<int> GET_ARCHITECTURE(std::string architecture);
__global__ void rand_scal(float * vec, int sz);
__global__ void fill_ones(float * vec, int sz);
__global__ void active_func(float * vec, int sz);
void rand_gen(float * vec, int sz);
__global__ void Rand_select(float * child_A, float * child_B, float * prnt_A, float * prnt_B, float * vecs, float MUT, int sz, int RW);


//Matrix
class Matrix {

private:

    float * Device;
    float * Host;

    int row;
    int column;

    int first_column;

    bool Is_Host_synced;
    bool Is_Device_synced;
    
public:

    Matrix(int ROWS, int COLUMNS);

    void Delete();
    void Sync();
    bool assign(Matrix & mat);

    void fill_column_one(int column_index);

    void set_first(int FIRST_COLUMN);
    bool MultiplyNN(Matrix & A, Matrix & B, cublasHandle_t handle);
    bool MultiplyTN(Matrix & A, Matrix & B, cublasHandle_t handle);
    bool MultiplyNT(Matrix & A, Matrix & B, cublasHandle_t handle);

    bool MultiplyTN(Matrix & A, Matrix & B, float Alph, cublasHandle_t handle);

    bool DMultiplyNN(Matrix & A, Matrix & B);
    bool DMultiplyNN(Matrix & A);
    
    bool MinusNN(Matrix & A, Matrix & B, cublasHandle_t handle);
    bool PlusNN(float Alph, Matrix & A, float Bet, Matrix & B, cublasHandle_t handle);

    friend std::ostream &operator<<(std::ostream &output, Matrix &mat);
    friend std::istream &operator>>(std::istream &input, Matrix &mat);
    
    friend class Network;
    friend std::vector<std::string> READ_CSV_DATA(std::string CSVdata, int features, int classes, Matrix & X, Matrix & Y);
    friend float cal_err(Matrix & y, Matrix & Y, cublasHandle_t handle);
    friend bool Matrix_Product(Matrix & Ra, Matrix & Rb, Matrix & A, Matrix & B, Matrix & vec, float muate_rate);
    friend void TEST(std::string CSVdata, std::string net);
    
};

//Network
class Network {

public:

    std::vector<Matrix> weights;
    std::vector<Matrix> Dw;
    std::vector<int>    architecture;

public:

    Network(std::vector<int>& arch);
    //~Network();
    void Delete();
    bool assign(Network & net);

    bool feed_forward(std::vector<Matrix> & A, cublasHandle_t handle);
    bool Backpropagation(Matrix &Y, std::vector<Matrix> & A, std::vector<Matrix> & Errors, float lambda, float BetM, cublasHandle_t handle);

    bool BackPs(Matrix &Y, std::vector<Matrix> & A, std::vector<Matrix> & Errors, float lambda, float BetM, int nums, cublasHandle_t handle);

    friend bool Network_Product(Network & Ra, Network & Rb, Network & A, Network & B, std::vector<Matrix> & Pchance, float muate_rate);
    friend void TRAIN(std::string CSVdata, std::string arch, int pops, int gens, float Bplambda, float BpBetM, int BpNums, float MUT);
    friend void TEST(std::string CSVdata, std::string net);

};