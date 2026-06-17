#ifndef MLP_H
#define MLP_H
#include <stdio.h>
#include <stdlib.h>
#define LEAKY_RATE 0.01f //  可调节项
#define ACT_FUNC(x) ((x) > 0 ? (x) : LEAKY_RATE *(x)) //  可调节项
#define BATCH_SIZE 128 // 批量大小 //  可调节项
#define TRAIN_IMG_NUM 60000 // 训练数据数量
#define TEST_IMG_NUM 10000 // 测试数据数量
#define INPUT_LAYER_SIZE 784 // 输入层大小
#define HIDDEN_LAYER_COUNT 2 // 隐藏层数量 //  可调节项
#define HIDDEN_LAYER_1_SIZE 100 // 隐藏层1大小 //  可调节项
#define HIDDEN_LAYER_2_SIZE 100 // 隐藏层2大小 //  可调节项
#define OUTPUT_LAYER_SIZE 10 // 输出层大小
#define LAYER_MAX HIDDEN_LAYER_1_SIZE // 除了输入层，最大的层 //  可调节项

extern float train_images[INPUT_LAYER_SIZE][TRAIN_IMG_NUM];
extern unsigned char train_labels[TRAIN_IMG_NUM];
extern float test_images[INPUT_LAYER_SIZE][TEST_IMG_NUM];
extern unsigned char test_labels[TEST_IMG_NUM];
// 神经网络模型参数
extern double ***w; //  [HIDDEN_LAYER_COUNT + 1][后层][前层]
extern double **b;
// 中间值
extern double ***z; //  [HIDDEN_LAYER_COUNT + 1][BATCH_SIZE][在本层中的次序]
extern double p[BATCH_SIZE][OUTPUT_LAYER_SIZE];
extern double ***dw; // [HIDDEN_LAYER_COUNT + 1][后层][前层]
extern double **dz; //  [HIDDEN_LAYER_COUNT + 1][在本层中的次序] // loss对z(或b)的导数的平均值
// 模型常量
extern const int layer_array[]; // 从input layer到output layer // 可调节项
extern const int param_num; // 参数的数量 // 可调节项
// 状态数据
extern int img_tag;
extern int epoch;
extern double learning_rate[2][HIDDEN_LAYER_COUNT + 1]; // w:[0], b:[1]
extern double train_loss;
extern double train_loss_min;
extern double test_loss;
extern double test_loss_min;
extern double test_loss_save;
extern int train_accuracy;
extern int train_accuracy_max;
extern int test_accuracy;
extern int test_accuracy_max;
extern double param_avg_abs;

void data_loader(void);
void model_initializer(void);
void model_saver(void);
void forward(void);
void backward(void);
void parameter_update(void);
void train_acc_param_avg(void);
void test_acc_loss(int size);
void reset(void);

#endif