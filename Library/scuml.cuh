#pragma once

#include "heads.cuh"
#include "Matrix.cuh"
#include "utility.cuh"
#include "Network.cuh"


void TRAIN(std::string CSVdata, std::string arch, int pops, int gens, float Bplambda, float BpBetM, int BpNums, float MUT) {

    std::vector<int> architecture = GET_ARCHITECTURE(arch);
    
    int features = architecture[0];
    int classes  = architecture.back();

    std::vector<std::string> IND;
    
    Matrix X(1, 1), Y(1, 1);
    
    IND = READ_CSV_DATA(CSVdata, features, classes, X, Y);

    std::vector<Matrix> A;  A.push_back(X);
    for (int i = 1; i < architecture.size(); i++) {
        int g = 1;
        if ((i + 1) == architecture.size()) g = 0;
        Matrix mat(IND.size(), architecture[i] + g);
        mat.fill_column_one(0);
        A.push_back(mat);
    }

    std::vector<Matrix> Err;
    for (int i = 0; i < architecture.size(); i++) {
        
        if (i == 0) {Matrix mat(0, 0); Err.push_back(mat);}
        else if ((i + 1) == architecture.size()) {Matrix mat(IND.size(), architecture[i]); Err.push_back(mat);}
        else {Matrix mat(IND.size(), architecture[i] + 1); Err.push_back(mat);}

    }
    std::vector<Matrix> Pchance;
    for (int i = 1; i < architecture.size(); i++){
        Matrix mat(architecture[i - 1] + 1, architecture[i]); Pchance.push_back(mat);}

    cublasHandle_t handle;
    cublasCreate(&handle);


    int CRG = gens / 64; // to show percentage


    std::vector<Network> Population;
    std::vector<Network> Nw_Population;
    for (int i = 0; i < pops; i++) {
        
        Network N1(architecture), N2(architecture);

        Population.push_back(N1);
        Nw_Population.push_back(N2);
    
    }

    
    Network BEST(architecture);
    BEST.feed_forward(A, handle);
    float BEST_err = cal_err(A.back(), Y, handle);

    
    while (gens --) {
        
        //evlauate
        std::vector<float> errors(pops);
        float max_err, min_err, Mu_err;

        for (int i = 0; i < pops; i++) {

            Population[i].BackPs(Y, A, Err, Bplambda, BpBetM, BpNums, handle);
            errors[i] = cal_err(A.back(), Y, handle);

            if (i == 0) max_err = min_err = Mu_err = errors[i];
            else {

                if (errors[i] > max_err)    max_err = errors[i];
                if (errors[i] < min_err)    min_err = errors[i];
                Mu_err += errors[i];

            }

            if (errors[i] < BEST_err) {
                BEST_err = errors[i];
                BEST.assign(Population[i]);
            }

        }

        float Ms_err = 1024 / (max_err - min_err);
        Mu_err /= pops;

        if (gens % CRG == 0)
        std::cerr << gens / CRG << '\t' << Mu_err << '\t' << max_err << '\t' << min_err << '\t' << BEST_err << '\n';

        std::vector<float> vals(pops);
        for (int i = 0; i < pops; i++)
            vals[i] = 758 - Ms_err * (errors[i] - Mu_err);


        //parent selecting and reproduction
        for (int i = 0; i < pops / 2; i++) {

            int a = rand() % (pops * 768);
            int b = rand() % (pops * 768);

            int A = 0, B = 0;
            for (; A < (pops - 1); A++) {a -= vals[A]; if (a <= 0)  break ;}
            for (; B < (pops - 1); B++) {b -= vals[B]; if (b <= 0)  break ;}
            //if (gens == 1) std::cerr << i << ' ' << Network_Product(Nw_Population[i * 2], Nw_Population[i * 2 + 1], Population[A], Population[B], Pchance, MUT) << ": " << A << ' ' << B <<'\n';
            Network_Product(Nw_Population[i * 2], Nw_Population[i * 2 + 1], Population[A], Population[B], Pchance, MUT);

        }

        Population.swap(Nw_Population);        

    }

    for (int i = 0; i < pops; i++) {
        Nw_Population[i].Delete();
        Population[i].Delete();
    }
    Nw_Population.clear();
    Population.clear();

    for (int i = 0; i < A.size(); i++)  A[i].Delete();
    A.clear();
    for (int i = 0; i < Err.size(); i++)  Err[i].Delete();
    Err.clear();
    for (int i = 0; i < Pchance.size(); i++)  Pchance[i].Delete();
    Pchance.clear();

    
    std::ofstream fout("out.net");
    fout << arch << "\n\n";

    for (int i = 0; i < BEST.weights.size(); i++)
        fout << BEST.weights[i] << '\n';
    
    cublasDestroy(handle);
    return ;

}
void TEST(std::string CSVdata, std::string net) {

    std::ifstream Netin(net);
    
    std::string arch;
    Netin >> arch;

    std::vector<int> architecture = GET_ARCHITECTURE(arch);
    
    Matrix X(1, 1), ych(1, 1);
    std::vector<std::string> indexes = READ_CSV_DATA(CSVdata, architecture[0], 0, X, ych);

    Network Fnet(architecture);

    for (int i = 0; i < Fnet.weights.size(); i++)
        Netin >> Fnet.weights[i];

    std::vector<Matrix> A;  A.push_back(X);
    for (int i = 1; i < architecture.size(); i++) {
        int g = 1;
        if ((i + 1) == architecture.size()) g = 0;
        Matrix mat(indexes.size(), architecture[i] + g);
        mat.fill_column_one(0);
        A.push_back(mat);
    }
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    Fnet.feed_forward(A, handle);

    CSVdata.pop_back();
    CSVdata.pop_back();
    CSVdata.pop_back();
    CSVdata.pop_back();
    
    CSVdata += "_out.csv";
    std::ofstream fout(CSVdata);

    A.back().Sync();
    for (int i = 0; i < indexes.size(); i++) {

        fout << indexes[i];

        for (int j = 0; j < architecture.back(); j++)
            fout << ',' << A.back().Host[j * indexes.size() + i];

        fout << '\n';

    }

    for (int i = 0; i < A.size(); i++) A[i].Delete();
    A.clear();
    cublasDestroy(handle);

    return ;

}

std::vector<std::string> READ_CSV_DATA(std::string CSVdata, int features, int classes, Matrix  & X, Matrix & Y) {

    srand (time(NULL));

    int nums = 0;

    std::vector<std::string> ret;

    std::ifstream fin(CSVdata);
    std::string ch;
    std::getline(fin, ch);

    std::map<int, std::map<int, float> > XX;
    std::map<int, std::map<int, float> > YY;
    
    while (std::getline(fin, ch)) {

        for (int i = 0; i < ch.size(); i++)
            if (ch[i] == ',')
                ch[i] = ' ';
        
        std::stringstream SS;
        SS << ch;
        SS >> ch;
        
        ret.push_back(ch);
        XX[nums][0] = 1;
        for (int i = 1; i <= features; i++) {

            float chh; SS >> chh;
            XX[nums][i] = chh;

        }

        for (int i = 0; i < classes; i++) {

            float chh; SS >> chh;
            YY[nums][i] = chh;

        }

        nums ++;

    }


    X.Delete();
    Y.Delete();
    Matrix Matx(nums, (features + 1)), Maty(nums, classes);
    X = Matx;
    Y = Maty;

    float * XXX = new float[nums * (features + 1)];
    float * YYY = new float[nums * classes];

    for (int i = 0; i < nums; i++)
        for(int j = 0; j < (features + 1); j++)
            XXX[j * nums + i] = XX[i][j];

    for (int i = 0; i < nums; i++)
        for(int j = 0; j < classes; j++)
            YYY[j * nums + i] = YY[i][j];

    cudaMemcpy(X.Device, XXX, nums * (features + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.Device, YYY, nums * classes * sizeof(float), cudaMemcpyHostToDevice);

    X.Is_Host_synced = Y.Is_Host_synced = false;
    X.Is_Device_synced = Y.Is_Device_synced = true;
    
    XX.clear();
    YY.clear();
    delete[] XXX;
    delete[] YYY;

    return ret;

}