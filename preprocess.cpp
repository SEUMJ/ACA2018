#include "preprocess.h"
#include "parameter.h"
#include "common.h"
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>

using namespace std;

void SplitString(const std::string&, std::vector<std::string>&, const std::string&);
int dicts_find(unsigned int, unsigned int [][2], int);

int Preprocess::process (std::istream& infile, unsigned int Max_size)
{
    unsigned int cnt = 0;
    unsigned int i = 0;
    unsigned int k = 0;
    string data; //每一行数据
    /**依次处理每一行*/
    while (getline(infile,data))
    {
        /**数据预处理分割*/
        vector<string> v;
        SplitString(data, v,","); //可按多个字符来分隔

        /**获取data_id*/
        strcpy(data_id[cnt],v[ID_POS].c_str());

        /**获取dense_data_buf和dense_data_s_buf*/
        float dense_temp;
        unsigned int dense_offset;
        dense_offset = cnt*CONTINOUS_SIZE;
        //计算
        for(i = 0; i < CONTINOUS_SIZE; i++) {
            dense_temp = (atoi(v[i + CONTINOUS_POS].c_str()) - dists[i][0])/dists[i][1];//计算的结果
            dense_data_buf[dense_offset] = dense_temp;//归一化
            dense_data_s_buf[dense_offset] = dense_temp * dense_temp;//求平方CATEGORIAL_SIZE
            dense_offset ++;
        }

        /**获取sparse_data_buf*/
        //处理末尾为空的情形
        if(data[data.size()-1] ==',') v.push_back("");

        //结合二分查找法找到one-hot编码
        unsigned int value;
        int index;
        unsigned int sparse_offset;

        sparse_offset = cnt*CATEGORIAL_SIZE;//基础偏移

        for (i = 0; i < CATEGORIAL_SIZE; i++) {
            //如果没有数据证明是unk
            if(v[i + CATEGORIAL_POS] == "") {
                sparse_data_buf[sparse_offset++] = unk[i];
                continue;
            }

            sscanf(v[i + CATEGORIAL_POS].c_str(),"%x",&value);

            index = dicts_find(value, dicts, 89162); //二分法查找

            if (index == -1) {//如果没有找到，则默认为unk
                sparse_data_buf[sparse_offset++] = unk[i];
            } else if (dicts[index][1] < offset[i]) {//如果小于，就去后面查找是否有，没考虑数组末尾，因为数组末尾人为查看了不相等
                for(k=1; k < CATEGORIAL_SIZE; k++) {
                    if(dicts[index+k][0] != value) { //一旦发现不等于，就是没有
                       sparse_data_buf[sparse_offset++] = unk[i];
                       break;
                    } else if (dicts[index+k][1] < offset[i]) { //如果还是小于，继续查找
                       continue;
                    } else if (dicts[index+k][1] >= offset[i+1]){ //大于也是没找到
                       sparse_data_buf[sparse_offset++] = unk[i];
                       break;
                    } else {
                       sparse_data_buf[sparse_offset++] = dicts[index+k][1]; //如果找到了，直接赋值
                       break;
                    }
                }
            } else if (dicts[index][1] >= offset[i+1]) {//如果大于，就去前面找，如果找不到符合要去的就是unk
                for(k=1; k < CATEGORIAL_SIZE; k++) {
                    if(dicts[index-k][0] != value) { //一旦发现不等于，就是没有
                       sparse_data_buf[sparse_offset++] = unk[i];
                       break;
                    } else if (dicts[index-k][1] >= offset[i+1]) { //如果还是大于，继续查找
                       continue;
                    } else if (dicts[index-k][1] < offset[i]){ //小于也是没找到
                       sparse_data_buf[sparse_offset++] = unk[i];
                       break;
                    } else {
                       sparse_data_buf[sparse_offset++] = dicts[index-k][1]; //如果找到了，直接赋值
                       break;
                    }
                }
            } else { //如果符合，则直接取值
                sparse_data_buf[sparse_offset++] = dicts[index][1];
            }
        }

        /**更新偏移量*/
        if(cnt < Max_size - 1) { //0~FRACTION_SIZE-1
            cnt++;
        } else {
            break;
        }
    }
    return 0;
}

int dicts_find(unsigned int m,unsigned int a[][2],int n)
{
	int o=0, h = n-1, i;
	while(o <= h)
	{
		i = (o+h)/2;
		if(a[i][0]==m)
		{
			return i;
		}
		if(a[i][0]<m)
		{
			o=i+1;
		}
		else
			h=i-1;
	}

	return -1;
}


void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}
