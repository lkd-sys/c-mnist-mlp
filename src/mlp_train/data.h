#ifndef DATA_H
#define DATA_H

float train_images[INPUT_LAYER_SIZE][TRAIN_IMG_NUM];
unsigned char train_labels[TRAIN_IMG_NUM];
float test_images[INPUT_LAYER_SIZE][TEST_IMG_NUM];
unsigned char test_labels[TEST_IMG_NUM];
// 神经网络模型参数
double ***w; // 权重weight
double **b; // 偏置bias
// 中间值
double ***z; // 线性输出
double p[BATCH_SIZE][OUTPUT_LAYER_SIZE]; // softmax输出
double ***dw; // 权重梯度
double **dz; // 偏置梯度
// 模型常量
const int layer_array[] = {INPUT_LAYER_SIZE,HIDDEN_LAYER_1_SIZE,HIDDEN_LAYER_2_SIZE,OUTPUT_LAYER_SIZE}; // 可调节项
const int param_num = HIDDEN_LAYER_1_SIZE + HIDDEN_LAYER_2_SIZE + OUTPUT_LAYER_SIZE + INPUT_LAYER_SIZE * HIDDEN_LAYER_1_SIZE + HIDDEN_LAYER_1_SIZE * HIDDEN_LAYER_2_SIZE + HIDDEN_LAYER_2_SIZE * OUTPUT_LAYER_SIZE; // 可调节项
// 状态数据
int img_tag = 0;
int epoch;
double learning_rate[2][HIDDEN_LAYER_COUNT + 1] =  //  可调节项
//w_hidden   w_output   b_hidden   b_output
// { 0.5f, 0.1f, 1.0f, 0.5f}
{{0.6f, 0.3f, 0.1f}, // w
{1.0f, 0.6f, 0.3f}} // b
// { 0.05f, 0.01f, 0.1f, 0.05f}
;
double train_loss;
double train_loss_min = BATCH_SIZE * 10;
double test_loss = TEST_IMG_NUM * 10;
double test_loss_min = TEST_IMG_NUM * 10;
double test_loss_save = TEST_IMG_NUM / 5; //  可调节项
int train_accuracy;
int train_accuracy_max;
int test_accuracy;
int test_accuracy_max;
double param_avg_abs;

clock_t start, clock1, clock2;

#endif