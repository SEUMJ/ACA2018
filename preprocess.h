#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "common.h"
#include <fstream>

class Preprocess {
    public:
        char  data_id[FULL_SIZE][ID_SIZE];                          //ID
        float dense_data_buf[FULL_SIZE*CONTINOUS_SIZE];             //整数型结果
        float dense_data_s_buf[FULL_SIZE*CONTINOUS_SIZE];           //整数型结果的平方
        unsigned int sparse_data_buf[FULL_SIZE*CATEGORIAL_SIZE];    //类别型结果
        int process(std::istream &, unsigned int);                  //对读取的数据进行处理，并将结果保存在buf中
};

#endif
