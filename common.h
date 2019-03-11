#ifndef COMMON_H
#define COMMON_H

//数量
#define ID_SIZE 100
#define CONTINOUS_SIZE 13
#define CATEGORIAL_SIZE 26

//在预测文件中的起始位置
#define ID_POS 0
#define CONTINOUS_POS 2
#define CATEGORIAL_POS 15

#define FACTOR_SIZE 10
#define SPARSE_SIZE 89188
#define FC_SIZE 400

#define FULL_SIZE       1000000
#define FRACTION_SIZE   500000
#define CORE_NUM        8
#define CORE_DAT_SIZE   (FRACTION_SIZE / CORE_NUM)
// #define BLOCK_NUM		2

#endif // COMMON_H
