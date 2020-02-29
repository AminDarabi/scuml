#pragma once


__global__ void rand_scal(float * vec, int sz) {

    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < sz)
        vec[ind] = (2 * vec[ind]) - 1;
    
    return ;

}
void rand_gen(float * vec, int sz) {

    curandGenerator_t gen;

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, rand() + time(NULL));
   
    curandGenerateUniform(gen, vec, sz);
    rand_scal<<<(sz + TPB - 1) / TPB, TPB>>> (vec, sz);
    
	curandDestroyGenerator(gen);

}
std::vector<int> GET_ARCHITECTURE(std::string architecture) {

    std::vector<int> ret;

    for (int i = 0; i < architecture.size(); i++)
        if (architecture[i] == ':')
            architecture[i] = ' ';

    std::stringstream SS; int ch;
    SS << architecture;
    while (SS >> ch)    ret.push_back(ch);

    return ret;

}
__global__ void fill_ones(float * vec, int sz) {

    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ind < sz)
        vec[ind] = 1;
    
    return ;

}
__global__ void active_func(float * vec, int sz) {

    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ind < sz)
        vec[ind] = tanhf(vec[ind]);
    
    return ;

}
//it destroys y
float cal_err(Matrix & y, Matrix & Y, cublasHandle_t handle) {

    if (!y.Is_Device_synced) y.Sync();
    if (!Y.Is_Device_synced) Y.Sync();

    int sz = y.row * y.column;
    float alpha = -1, ret = 0;
    cublasSaxpy(handle, sz, &alpha, Y.Device, 1, y.Device, 1);
    cublasSnrm2(handle, sz, y.Device, 1, &ret);

    y.Is_Host_synced = false;

    return ret;

}
__global__ void Rand_select(float * child_A, float * child_B, float * prnt_A, float * prnt_B, float * vecs, float MUT, int sz, int RW) {

    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < sz) {
        int g = ind / RW;
        g *= RW;
        if(vecs[g] < 0.5) {
            if (!(vecs[ind] < MUT))                                 child_A[ind] = prnt_A[ind];
            else    child_A[ind] = (vecs[ind] * 2) / MUT - 1;
            if (!(vecs[ind] < 2 * MUT && vecs[ind] > MUT))          child_B[ind] = prnt_B[ind];
            else    child_B[ind] = ((vecs[ind] - MUT) * 2) / MUT - 1;
        } else {
            if (!(vecs[ind] < MUT))                                 child_A[ind] = prnt_B[ind];
            else    child_A[ind] = (vecs[ind] * 2) / MUT - 1;
            if (!(vecs[ind] < 2 * MUT && vecs[ind] > MUT))          child_B[ind] = prnt_A[ind];
            else    child_B[ind] = ((vecs[ind] - MUT) * 2) / MUT - 1;
        }
    }
    
    return ;

}