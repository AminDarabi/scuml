#include <iostream>
#include <string>
#include <vector>

#include "Library/scuml.cuh"

using namespace std;


bool EqlCstr(char * a, char * b) {

    for (int i = 0; a[i] != 0 || b[i] != 0; i++)
        if (a[i] != b[i])
            return false;
    
    return true;

}

int main(int arg, char ** args) {

    vector<string> ARGS;
    for (int i = 0; i < arg; i++) {string S(args[i]); ARGS.push_back(S);}

    //void TRAIN(std::string CSVdata, std::string arch, int pops, int gens, float Bplambda, float BpBetM, int BpNums, float MUT)
    if (arg == 4 && ARGS[1] == "train") TRAIN(ARGS[3], ARGS[2], 16, 128, 0.0001, 0.16, 16, 0.0005);
    else if (arg == 4 && ARGS[1] == "test")  TEST(ARGS[3], ARGS[2]);
    else if (ARGS[1] == "train") {

        int pops = 16; int gens = 128;
        float Bplambda = 0.0001; float BpBetM = 0.16; int BpNums = 16;
        float MUT = 0.0005;
    
        int i = 4;
        
        if(i < ARGS.size()) {stringstream SS; SS << ARGS[i++];  SS >> pops;}
        if(i < ARGS.size()) {stringstream SS; SS << ARGS[i++];  SS >> gens;}

        if(i < ARGS.size()) {stringstream SS; SS << ARGS[i++];  SS >> Bplambda;}
        if(i < ARGS.size()) {stringstream SS; SS << ARGS[i++];  SS >> BpBetM;}
        if(i < ARGS.size()) {stringstream SS; SS << ARGS[i++];  SS >> BpNums;}

        if(i < ARGS.size()) {stringstream SS; SS << ARGS[i++];  SS >> MUT;}
        
        TRAIN(ARGS[3], ARGS[2], pops, gens, Bplambda, BpBetM, BpNums, MUT);

    }


    return 0;

}