#pragma once

Network::Network(std::vector<int>& arch) {
    
    architecture.push_back(arch[0]);

    for (int i = 1; i < arch.size(); i++) {
        architecture.push_back(arch[i]);
        Matrix mat(arch[i - 1] + 1, arch[i]), Dmat(arch[i - 1] + 1, arch[i]);
        weights.push_back(mat);
        Dw.push_back(Dmat);
    }

}
bool Network::feed_forward(std::vector<Matrix> &A, cublasHandle_t handle) {

    for (int i = 1; i < architecture.size(); i++) {

        if (i != (architecture.size() - 1)) A[i].set_first(1);
        else A[i].set_first(0);
        A[i - 1].set_first(0);
        weights[i - 1].set_first(0);
        
        //std::cerr << A[i - 1].row << '*' << A[i - 1].column << " and " << weights[i - 1].row << '*' << weights[i - 1].column << ' ';
        //std::cerr << A[i].row << '*' << A[i].column << '\n';
        if (A[i].MultiplyNN(A[i - 1], weights[i - 1], handle) == false) return false;

        int sz = A[i].row * (A[i].column - A[i].first_column);
        active_func<<<(sz + TPB - 1) / TPB, TPB>>>(A[i].Device + (A[i].first_column * A[i].row), sz);

    }

    return true;

}
bool Derivatives(Matrix &Error, Matrix &A) {

    if (!Error.Is_Device_synced)    Error.Sync();
    if (!A.Is_Device_synced)    Error.Sync();
    
    if (A.row != Error.row) return false;
    if (A.column - A.first_column != Error.column - Error.first_column) return false;
    
    int sz = A.row * (A.column - A.first_column);
    dirv_active_func<<<(sz + TPB - 1) / TPB, TPB>>>(Error.Device + (Error.row * Error.first_column), A.Device + (A.first_column * A.row), sz);
    Error.Is_Host_synced = false;

    return true;

}
bool Network::Backpropagation(Matrix &Y, std::vector<Matrix> & A, std::vector<Matrix> & Errors, float lambda, float BetM, cublasHandle_t handle) {
    
    if (feed_forward(A, handle) == false) return false;
    
    int L = Errors.size() - 1;

    A[L].set_first(0);
    Y.set_first(0);
    Errors[L].set_first(0);
    if (Errors[L].MinusNN(A[L], Y, handle) == false) return false;
    //std::cerr<< "\n EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE \n" << Errors[L] << "\n\n DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD \n\n";
    if (Derivatives(Errors[L], Y) == false) return false;
    //std::cerr<< Errors[L] << "\n FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF \n";
    for (int i = L - 1; i > 0; i --) {

        if (i < (L - 1)) Errors[i + 1].set_first(1);
        Errors[i].set_first(0);
        weights[i].set_first(0);
        A[i].set_first(0);
        
        if (Errors[i].MultiplyNT(Errors[i + 1], weights[i], handle) == false) return false;
        if (Derivatives(Errors[i], A[i]) == false) return false;
    
    }

    for(int i = 0; i < weights.size(); i++) {

        Errors[i + 1].set_first(1);
        Dw[i].set_first(0);
        A[i].set_first(0);
        if ((i + 1) == weights.size())  Errors[i + 1].set_first(0);
        
        if (Dw[i].MultiplyTN(A[i], Errors[i + 1], handle) == false) return false;
        if (weights[i].PlusNN(1, weights[i], (-1 * BetM) / A[i].row, Dw[i], handle) == false) return false;
        
        Dw[i].set_first(1);
        weights[i].set_first(1);
        if (weights[i].PlusNN(1 - lambda, weights[i], 0, Dw[i], handle) == false) return false;

    }

    return true;

}
bool Network::BackPs(Matrix &Y, std::vector<Matrix> & A, std::vector<Matrix> & Errors, float lambda, float BetM, int nums, cublasHandle_t handle) {

    for (int i = 0; i < nums; i++)
        if (Backpropagation(Y, A, Errors, lambda, BetM, handle) == false)
            return false;

    return true;

}
bool Network::assign(Network & net) {

    if (net.architecture.size() != architecture.size()) return false;

    for (int i = 1; i < architecture.size(); i++)
        if (!weights[i - 1].assign(net.weights[i - 1]))
            return false;

    return true;

}
void Network::Delete() {

    for (int i = 1; i < architecture.size(); i++) {weights[i - 1].Delete(); Dw[i - 1].Delete();}
    architecture.clear();
    return ;

}
bool Network_Product(Network & Ra, Network & Rb, Network & A, Network & B, std::vector<Matrix> & Pchance, float muate_rate) {

    for (int i = 1; i < A.architecture.size(); i++)
        if (!Matrix_Product(Ra.weights[i - 1], Rb.weights[i - 1], A.weights[i - 1], B.weights[i - 1], Pchance[i - 1], muate_rate))
            return false;
        
    return true;

}
