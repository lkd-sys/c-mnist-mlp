#include "mlp.h"

// 文件头部信息：
// 模型基本信息：隐藏层个数
// 模型基本信息：隐藏层数组
// 模型基本信息：激活函代号
// 训练信息：Epoch
// 训练信息：Batch Size
// 训练信息：learning rate二维数组
void model_saver(void) {
    FILE* model;
    do{
        model = fopen("../../models/MLP_models/MLP.bin", "wb"); // "wb" : 以二进制方式打开文件，若不存在则创建，若存在则清空
        printf("Error opening model file.\n");
    } while (!model); // 父目录不存在或文件被其他进程锁定（互斥锁）可以一直等用户创建父目录或杀掉那个锁定文件的进程
    int n = HIDDEN_LAYER_COUNT;
    fwrite(&n, sizeof(int), 1, model); // 模型基本信息：隐藏层个数
    fwrite(layer_array + 1, sizeof(int), HIDDEN_LAYER_COUNT, model); // 模型基本信息：隐藏层数组
    n = 0;
    fwrite(&n, sizeof(int), 1, model); // 模型基本信息：激活函代号(leaky relu: 0)
    fwrite(&epoch, sizeof(int), 1, model); // 训练信息：Epoch
    int batch_size = BATCH_SIZE;
    fwrite(&batch_size, sizeof(int), 1, model); // 训练信息：Batch Size
    fwrite(&learning_rate, sizeof(double), 2 * (HIDDEN_LAYER_COUNT + 1), model); // 训练信息：learning rate二维数组
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        fwrite(b[i], sizeof(double), layer_array[i + 1], model);
        for (int j = 0; j < layer_array[i + 1]; j++) {
            fwrite(w[i][j], sizeof(double), layer_array[i], model);
        }
    }
    fclose(model);
    printf("\nModel Is Saved\n\n");
}
