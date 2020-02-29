#pragma once


__global__ void Dmal(float * A, float * B, float * mat, int sz) {

    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < sz)   mat[ind] = A[ind] * B[ind];
    
    return ;

}
Matrix::Matrix(int ROWS, int COLUMNS) {

    int SZ = ROWS * COLUMNS;
    
    Host = new float[SZ];
    cudaMalloc((void**) &Device, SZ * sizeof(float));

    row = ROWS;
    column = COLUMNS;

    first_column = 0;
    
    rand_gen(Device, SZ);

    Is_Device_synced = true;
    Is_Host_synced   = false;

}
void Matrix::Delete() {

    delete[] Host;
    cudaFree(Device);

}
void Matrix::set_first(int FIRST_COLUMN) {

    first_column = FIRST_COLUMN;
    return ;

}
void Matrix::Sync() {

    if (!Is_Device_synced)
        cudaMemcpy(Device, Host, row * column * sizeof(float), cudaMemcpyHostToDevice);
    else
        cudaMemcpy(Host, Device, row * column * sizeof(float), cudaMemcpyDeviceToHost);
    
    Is_Device_synced = Is_Host_synced = true;
    return ;

}
bool Matrix::MultiplyNN(Matrix & A, Matrix & B, cublasHandle_t handle) {

    if (!A.Is_Device_synced)    A.Sync();
    if (!B.Is_Device_synced)    B.Sync();  

    float alpha = 1;
    float beta  = 0;

    int CLM = B.column - B.first_column;

    if ((column - first_column) != CLM)         return false;
    if ((A.column - A.first_column) != B.row)   return false;
    if (row != A.row)                           return false;


    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row, CLM, B.row, &alpha, A.Device + (A.first_column * A.row), A.row, B.Device + (B.first_column * B.row), B.row, &beta, Device + (first_column * row), row);
    
    Is_Device_synced = true;
    Is_Host_synced   = false;

    return true;

}
bool Matrix::MultiplyTN(Matrix & A, Matrix & B, cublasHandle_t handle) {

    if (!A.Is_Device_synced)    A.Sync();
    if (!B.Is_Device_synced)    B.Sync();  

    float alpha = 1;
    float beta  = 0;

    int MM = A.column - A.first_column;
    int NN = B.column - B.first_column;
    int KK = B.row;

    if ((column - first_column) != NN)          return false;
    if (row != MM)                              return false;
    if (A.row != KK)                            return false;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, MM, NN, KK, &alpha, A.Device + (A.first_column * A.row), A.row, B.Device + (B.first_column * B.row), B.row, &beta, Device + (first_column * row), row);
    
    Is_Device_synced = true;
    Is_Host_synced   = false;

    return true;

}
bool Matrix::MultiplyTN(Matrix & A, Matrix & B, float Alph, cublasHandle_t handle) {

    if (!A.Is_Device_synced)    A.Sync();
    if (!B.Is_Device_synced)    B.Sync();  

    float alpha = Alph;
    float beta  = 0;

    int MM = A.column - A.first_column;
    int NN = B.column - B.first_column;
    int KK = B.row;

    if ((column - first_column) != NN)          return false;
    if (row != MM)                              return false;
    if (A.row != KK)                            return false;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, MM, NN, KK, &alpha, A.Device + (A.first_column * A.row), A.row, B.Device + (B.first_column * B.row), B.row, &beta, Device + (first_column * row), row);
    
    Is_Device_synced = true;
    Is_Host_synced   = false;

    return true;

}
bool Matrix::MultiplyNT(Matrix & A, Matrix & B, cublasHandle_t handle) {

    if (!A.Is_Device_synced)    A.Sync();
    if (!B.Is_Device_synced)    B.Sync();  

    float alpha = 1;
    float beta  = 0;

    int MM = A.row;
    int NN = B.row;
    int KK = B.column - B.first_column;

    if ((column - first_column) != NN)          return false;
    if (row != MM)                              return false;
    if ((A.column - A.first_column) != KK)      return false;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, MM, NN, KK, &alpha, A.Device + (A.first_column * A.row), A.row, B.Device + (B.first_column * B.row), B.row, &beta, Device + (first_column * row), row);
    
    Is_Device_synced = true;
    Is_Host_synced   = false;

    return true;

}
bool Matrix::DMultiplyNN(Matrix & A, Matrix & B) {

    if (!A.Is_Device_synced)    A.Sync();
    if (!B.Is_Device_synced)    B.Sync();  

    int CLM = B.column - B.first_column;
    int ROW = B.row;
    

    if ((column - first_column) != CLM)         return false;
    if ((A.column - A.first_column) != CLM)     return false;
    
    if (A.row != ROW)   return false;
    if (row != ROW)     return false;

    int sz = ROW * CLM;
    Dmal<<<(sz + TPB - 1) / TPB, TPB>>>(A.Device + (A.first_column * A.row), B.Device + (B.first_column * B.row), Device + (first_column * row), sz);
    
    Is_Device_synced = true;
    Is_Host_synced   = false;

    return true;

}
bool Matrix::DMultiplyNN(Matrix & A) {

    if (!A.Is_Device_synced)    A.Sync();
    if (!Is_Device_synced)      Sync();

    int CLM = A.column - A.first_column;
    int ROW = A.row;
    

    if ((column - first_column) != CLM)         return false;
    if (row != ROW)     return false;

    int sz = ROW * CLM;
    Dmal<<<(sz + TPB - 1) / TPB, TPB>>>(A.Device + (A.first_column * A.row), Device + (first_column * row), Device + (first_column * row), sz);
    
    Is_Device_synced = true;
    Is_Host_synced   = false;

    return true;

}
bool Matrix::MinusNN(Matrix & A, Matrix & B, cublasHandle_t handle) {

    if (!A.Is_Device_synced)    A.Sync();
    if (!B.Is_Device_synced)    B.Sync();  

    float alpha = 1;
    float beta  = -1;

    int CLM = B.column - B.first_column;
    int ROW = B.row;
    
    if ((column - first_column) != CLM)         return false;
    if ((A.column - A.first_column) != CLM)     return false;
    
    if (A.row != ROW)   return false;
    if (row != ROW)     return false;

    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, ROW, CLM, &alpha, A.Device + (A.first_column * A.row), A.row, &beta, B.Device + (B.first_column * B.row), B.row, Device + (first_column * row), row);
    Is_Device_synced = true;
    Is_Host_synced   = false;

    return true;

}
bool Matrix::PlusNN(float Alph, Matrix & A, float Bet, Matrix & B, cublasHandle_t handle) {

    if (!A.Is_Device_synced)    A.Sync();
    if (!B.Is_Device_synced)    B.Sync();  

    float alpha = Alph;
    float beta  = Bet;

    int CLM = B.column - B.first_column;
    int ROW = B.row;
    
    if ((column - first_column) != CLM)         return false;
    if ((A.column - A.first_column) != CLM)     return false;
    
    if (A.row != ROW)   return false;
    if (row != ROW)     return false;

    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, ROW, CLM, &alpha, A.Device + (A.first_column * A.row), A.row, &beta, B.Device + (B.first_column * B.row), B.row, Device + (first_column * row), row);
    Is_Device_synced = true;
    Is_Host_synced   = false;

    return true;

}
std::ostream &operator<<(std::ostream &output, Matrix & mat) {

    if (!mat.Is_Host_synced) mat.Sync();

    for (int i = 0; i < mat.row; i++, output << '\n')
        for (int j = 0; j < mat.column; j++) {

            if (j)  output << ' ';
            output << mat.Host[i + j * mat.row];

        }

    return output;
    
}
std::istream &operator>>(std::istream &input, Matrix &mat) { 

    for (int i = 0; i < mat.row; i++)
        for (int j = 0; j < mat.column; j++)
            input >> mat.Host[i + j * mat.row];
    
    mat.Is_Host_synced = true;
    mat.Is_Device_synced = false;

    return input;            

}
void Matrix::fill_column_one(int column_index) {
    fill_ones<<<(row + TPB - 1) / TPB, TPB>>>(Device + column_index * row, row);
}
bool Matrix::assign(Matrix & mat) {

    if (!Is_Device_synced)      Sync();
    if (!mat.Is_Device_synced)  mat.Sync();

    if (mat.row != row) return false;
    if (mat.column != column) return false;
    
    int S = row * column;
    cudaMemcpy(Device, mat.Device, S * sizeof(float), cudaMemcpyDeviceToDevice);

    Is_Host_synced = false;
    first_column = mat.first_column;

    return true;

}
bool Matrix_Product(Matrix & Ra, Matrix & Rb, Matrix & A, Matrix & B, Matrix & vec, float muate_rate) {

    if (Ra.row != Rb.row || Ra.row != A.row || Ra.row != B.row) return false;
    if (Ra.column != Rb.column || Ra.column != A.column)        return false;
    if (Ra.column != B.column || Ra.column != vec.column)       return false;
    if (Ra.row != vec.row)                                      return false;

    if (!A.Is_Device_synced) A.Sync();
    if (!B.Is_Device_synced) B.Sync();

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, rand() + time(NULL));

    int S = A.row * A.column;
    curandGenerateUniform(gen, vec.Device, S);
    Rand_select<<<(S + TPB - 1) / TPB, TPB>>>(Ra.Device, Rb.Device, A.Device, B.Device, vec.Device, muate_rate, S, A.row);

    curandDestroyGenerator(gen);

    return true;

}