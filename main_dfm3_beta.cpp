#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <pthread.h>
#include "polaris.h"
#include "common.h"
#include "fmweight.h"
#include "fcweight.h"
#include "preprocess.h"

using namespace std;

PolarisContext* ctxt;   // Polaris context object
// FPGA memory pointers of FC
float* fc_L0i_fpga = NULL;      float* fc_L0w_fpga = NULL;      float* fc_L0b_fpga = NULL;
float* fc_L1i_fpga = NULL;      float* fc_L1w_fpga = NULL;      float* fc_L1b_fpga = NULL;
float* fc_L2i_fpga = NULL;      float* fc_L2w_fpga = NULL;      float* fc_L2b_fpga = NULL;
float* fc_L3i_fpga = NULL;      float* fc_L3w_fpga = NULL;      float* fc_L3b_fpga = NULL;
float* fc_L3O_fpga = NULL;

// Global Variables
Preprocess p;       // 预处理类对象
// FC Buffers
float fc_dati_buf[FRACTION_SIZE * FC_DATA_WIDTH];
float fc_res_buf[FRACTION_SIZE];
// FM Buffers
float dense_res_buf[FRACTION_SIZE * FACTOR_SIZE];
float sparse_res_buf[FRACTION_SIZE];



// Functions
void data_gen(ifstream& infile);            // Prepare data for FPGA
void* gemm_hw(void* param);                 // Use FPGA to compute.
void deep_fm_malloc();                      // FPGA memory allocation
void deep_fm_init();                        // FPGA memory initialization
void deep_fm_free();                        // Free FPGA memory
void fm_dense_sw();                         // Use software to compute fm result of dense data.
void fm_sparse_sw();                        // Use software to compute fm result of sparse data.
void write_result(ofstream& ofs);           // Write results into output file.



int main(int argc, char* argv[])
{
    clock_t start = clock();		// Timing begins.

    if (argc != 2)      // Check arguments.
    {
        cerr << "Argument error. 2 arguments are needed.\n";
        exit(1);
    }

    ctxt = polaris_create_context(0);   // Create Polaris context object.

    deep_fm_malloc();       // FPGA memory allocation
    deep_fm_init();         // FPGA memory initialization

    ifstream infile(argv[1], ios::in | ios::binary);     // Open input data file.
    if (!infile)
    {
        cerr << "Failed to open files." << endl;
        exit(1);
    }

    ofstream outfile("predict.csv", ios::out | ios::binary);    // Create output data file.
    if (!outfile)
    {
        cerr << "Failed to create output files." << endl;
        exit(1);
    }

    pthread_t gh_tid[CORE_NUM];
    pthread_attr_t tattr[CORE_NUM];

    int param[CORE_NUM];
    for (int i = 0; i < CORE_NUM; i++) param[i] = i;

    int round = 0;
    while (FRACTION_SIZE * round < FULL_SIZE)
    {
        data_gen(infile);
        cout << "data_gen " << 1 + round << " completed.\n";
        for (int i = 0; i < CORE_NUM; i++)
        {
            pthread_create(&gh_tid[i], &tattr[i], gemm_hw, &param[i]);
            cout << "Thread " << 1 + i << " created.\n";
        }
        for (int i = 0; i < CORE_NUM; i++)
        {
            pthread_join(gh_tid[i], NULL);
            cout << "Thread " << 1 + i << " completed.\n";
        }
        write_result(outfile);  // Write results into output file.
        cout << "Round " << 1 + round++ << " finished.\n";
    }

    infile.close();     // Close input file.
    outfile.close();    // Close output file.
    deep_fm_free();     // Free FPGA memory.

    clock_t end = clock();			// Timing ends.
    float time = (float)(end - start) / CLOCKS_PER_SEC;

    cout << "Total time: " << time << "s" << endl;
    return 0;
}



// Prepare data for FPGA
void data_gen(ifstream& infile)
{
    p.process(infile, FRACTION_SIZE);

    //*************************************************************************
    // Compute FM Result
    fm_sparse_sw();     // Compute fm result of sparse data.
    fm_dense_sw();      // Compute fm result of dense data.

    //*************************************************************************
    // FC Data Preparation
    for (int db_cnt = 0; db_cnt < FRACTION_SIZE; db_cnt ++)
    {
        int offset = FC_DATA_WIDTH * db_cnt,
            offset2 = CONTINOUS_SIZE * db_cnt,
            offset3 = CATEGORIAL_SIZE * db_cnt;
        memcpy(&fc_dati_buf[offset], p.dense_data_buf + offset2, CONTINOUS_SIZE * sizeof(float));

        for (int i = 0; i < CATEGORIAL_SIZE; i++)       // Embedding
            for (int j = 0; j < FACTOR_SIZE; j++)
            fc_dati_buf[offset + CONTINOUS_SIZE + i * FACTOR_SIZE + j] =
                SparseFactors[p.sparse_data_buf[offset3 + i]][j];
    }   

    // Copy FC input data from CPU memory to FPGA memory.
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L0i_fpga, fc_dati_buf,
                        FRACTION_SIZE * FC_DATA_WIDTH * sizeof(float));
}

// Use FPGA to compute.
void* gemm_hw(void* param)
{
    int sid = *(int*)param;
    printf("sid == %d\n", sid);

    // FC Layer 0
    polaris_gemm(ctxt, CORE_DAT_SIZE, FC_SIZE, FC_DATA_WIDTH,
                    &fc_L0i_fpga[CORE_DAT_SIZE * FC_DATA_WIDTH * sid],
                    fc_L0w_fpga,
                    &fc_L1i_fpga[CORE_DAT_SIZE * FC_SIZE * sid],
                    fc_L0b_fpga);
    polaris_activation(ctxt, POLARIS_RELU, CORE_DAT_SIZE * FC_SIZE,
                    1, fc_L1i_fpga, 0, fc_L1i_fpga);
    
    // FC Layer 1
    polaris_gemm(ctxt, CORE_DAT_SIZE, FC_SIZE, FC_SIZE,
                    &fc_L1i_fpga[CORE_DAT_SIZE * FC_SIZE * sid],
                    fc_L1w_fpga,
                    &fc_L2i_fpga[CORE_DAT_SIZE * FC_SIZE * sid],
                    fc_L1b_fpga);
    polaris_activation(ctxt, POLARIS_RELU, CORE_DAT_SIZE * FC_SIZE,
                    1, fc_L2i_fpga, 0, fc_L2i_fpga);
    
    // FC Layer 2
    polaris_gemm(ctxt, CORE_DAT_SIZE, FC_SIZE, FC_SIZE,
                    &fc_L2i_fpga[CORE_DAT_SIZE * FC_SIZE * sid],
                    fc_L2w_fpga,
                    &fc_L3i_fpga[CORE_DAT_SIZE * FC_SIZE * sid],
                    fc_L2b_fpga);
    polaris_activation(ctxt, POLARIS_RELU, CORE_DAT_SIZE * FC_SIZE,
                    1, fc_L3i_fpga, 0, fc_L3i_fpga);
    
    // FC Layer 3
    polaris_gemm(ctxt, CORE_DAT_SIZE, 1, FC_SIZE,
                    &fc_L3i_fpga[CORE_DAT_SIZE * FC_SIZE * sid],
                    fc_L3w_fpga,
                    &fc_L3O_fpga[CORE_DAT_SIZE * sid],
                    fc_L3b_fpga);

    return NULL;
}

// FPGA memory allocation
inline void deep_fm_malloc()
{
    // FPGA memory allocation for FC
    polaris_malloc(ctxt, FRACTION_SIZE * FC_DATA_WIDTH * sizeof(float), (void**)&fc_L0i_fpga);
    polaris_malloc(ctxt, FRACTION_SIZE * FC_SIZE * sizeof(float), (void**)&fc_L1i_fpga);
    polaris_malloc(ctxt, FRACTION_SIZE * FC_SIZE * sizeof(float), (void**)&fc_L2i_fpga);
    polaris_malloc(ctxt, FRACTION_SIZE * FC_SIZE * sizeof(float), (void**)&fc_L3i_fpga);
    polaris_malloc(ctxt, FRACTION_SIZE * sizeof(float), (void**)&fc_L3O_fpga);

    polaris_malloc(ctxt, FC_SIZE * FC_DATA_WIDTH * sizeof(float), (void**)&fc_L0w_fpga);
    polaris_malloc(ctxt, FC_SIZE * FC_SIZE * sizeof(float), (void**)&fc_L1w_fpga);
    polaris_malloc(ctxt, FC_SIZE * FC_SIZE * sizeof(float), (void**)&fc_L2w_fpga);
    polaris_malloc(ctxt, FC_SIZE * sizeof(float), (void**)&fc_L3w_fpga);

    polaris_malloc(ctxt, FC_SIZE * sizeof(float), (void**)&fc_L0b_fpga);
    polaris_malloc(ctxt, FC_SIZE * sizeof(float), (void**)&fc_L1b_fpga);
    polaris_malloc(ctxt, FC_SIZE * sizeof(float), (void**)&fc_L2b_fpga);
    polaris_malloc(ctxt, sizeof(float), (void**)&fc_L3b_fpga);
}

// FPGA memory initialization
inline void deep_fm_init()
{
    // FPGA memory initialization for FC
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L0w_fpga, FC_L0_W,
                        FC_SIZE * FC_DATA_WIDTH * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L1w_fpga, FC_L1_W,
                        FC_SIZE * FC_SIZE * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L2w_fpga, FC_L2_W,
                        FC_SIZE * FC_SIZE * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L3w_fpga, FC_L3_W,
                        FC_SIZE * sizeof(float));

    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L0b_fpga, FC_L0_B, FC_SIZE * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L1b_fpga, FC_L1_B, FC_SIZE * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L2b_fpga, FC_L2_B, FC_SIZE * sizeof(float));    
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, fc_L3b_fpga, &FC_L3_B, sizeof(float));
}

// Free FPGA memory
inline void deep_fm_free()
{
    // Free FC's FPGA memory.
    polaris_free(ctxt, fc_L0i_fpga);
    polaris_free(ctxt, fc_L1i_fpga);
    polaris_free(ctxt, fc_L2i_fpga);
    polaris_free(ctxt, fc_L3i_fpga);

    polaris_free(ctxt, fc_L0w_fpga);
    polaris_free(ctxt, fc_L1w_fpga);
    polaris_free(ctxt, fc_L2w_fpga);
    polaris_free(ctxt, fc_L3w_fpga);

    polaris_free(ctxt, fc_L0b_fpga);
    polaris_free(ctxt, fc_L1b_fpga);
    polaris_free(ctxt, fc_L2b_fpga);
    polaris_free(ctxt, fc_L3b_fpga);

    polaris_free(ctxt, fc_L3O_fpga);
}

// Use software to compute fm result of dense data.
inline void fm_dense_sw()
{
    for (int k = 0; k < FRACTION_SIZE; k++)
    {
        float Vsum = 0;
        int offset = k * CONTINOUS_SIZE;
        for (int f = 0; f < FACTOR_SIZE; f++)
        {
            float V0sum = 0, V1sum = 0;
            for (int i = 0; i < CONTINOUS_SIZE; i++)
            {
                V0sum += p.dense_data_buf[offset + i] * D_Factors[f][i];
                V1sum += p.dense_data_s_buf[offset + i] * D_Factors2[f][i];
            }
            Vsum += V0sum * V0sum - V1sum;
        }

        float layer0sum = 0;
        for (int i = 0; i < CONTINOUS_SIZE; i++)        // 线性求和
            layer0sum += p.dense_data_buf[offset + i] * layer0_w[i];

        dense_res_buf[k] = 	layer0sum + 0.5 * Vsum + layer0_bias;
    }
}

// Use software to compute fm result of sparse data.
inline void fm_sparse_sw()
{
    for (int k = 0; k < FRACTION_SIZE; k++)
    {
        float Vsum = 0;
        int offset = k * CATEGORIAL_SIZE;
        for (int f = 0; f < FACTOR_SIZE; f++)
        {
            float V0sum = 0, V1sum = 0;
            for (int i = 0; i < CATEGORIAL_SIZE; i++)
            {
                V0sum += SparseFactors[p.sparse_data_buf[offset + i]][f];
                V1sum += SparseFactors2[p.sparse_data_buf[offset + i]][f];
            }
            Vsum += V0sum * V0sum - V1sum;
        }

        float layer1sum = 0;
        for (int i = 0; i < CATEGORIAL_SIZE; i++)       // 线性求和
            layer1sum += layer1_w[p.sparse_data_buf[offset + i]];

        sparse_res_buf[k] = layer1sum + 0.5 * Vsum + layer1_bias;
    }
}

// Write results into output file.
void write_result(ofstream& ofs)
{
    float dfm_res;
    for (int i = 0; i < FRACTION_SIZE; i++)
    {
        dfm_res = dense_res_buf[i] * layer2_w[0]            // FM: Dense result
                    + sparse_res_buf[i] * layer2_w[1]       // FM: Sparse result
                    + fc_res_buf[i];                        // FC: 4-FC outputs

        if (i < 10)
            cout << "Dense res: " << dense_res_buf[i] << "\t"
                << "Sparse res: " << sparse_res_buf[i] << "\t"
                << "FC res: " << fc_res_buf[i] << endl;

        ofs << p.data_id[i] << "," << 1 / (1 + exp(-dfm_res)) << endl;    // Write result to file.
    }
}
