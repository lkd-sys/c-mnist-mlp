#include "mlp.h"

void backward(void) {
    double *p1, *p2;
    p1 = (double*)malloc(sizeof(double) * LAYER_MAX); // 上次的数据
    p2 = (double*)malloc(sizeof(double) * LAYER_MAX); // 待写入的数据
    if (!p1 || !p2) {
        printf("Memory allocation failed in backward function.\n");
        exit(1);
    }
    for (int i = 0; i < BATCH_SIZE; i ++) {
        for (int j = HIDDEN_LAYER_COUNT; j >= 0 ; j--) {
            double* p3 = p1;  p1 = p2;  p2 = p3;
            for (int k = 0; k < LAYER_MAX; k++) // initializer
                p2[k] = 0.0f;
            for (int k = 0; k < layer_array[j + 1]; k++) { // k : 后一层
                if (j == HIDDEN_LAYER_COUNT)
                    dz[j][k] += (p1[k] = (p[i][k] - ((k == train_labels[(img_tag + i) % TRAIN_IMG_NUM]) ? 1 : 0)) / BATCH_SIZE);
                for (int l = 0; l < layer_array[j]; l++) { // l : 前一层
                    // dw
                    if (j == 0){
                        dw[j][k][l] += train_images[l][(img_tag + i) % TRAIN_IMG_NUM] * p1[k];
                    }else{
                        dw[j][k][l] += ACT_FUNC(z[j - 1][i][l]) * p1[k];
                        // dz
                        double d = p1[k] * w[j][k][l];
                        if (z[j - 1][i][l] > 0) {
                            dz[j - 1][l] += d;
                            p2[l] += d;
                        }else{
                            dz[j - 1][l] += LEAKY_RATE * d;
                            p2[l] += LEAKY_RATE * d;
                        }
                    }
                }
            }
        }
    }
    free(p1);
    free(p2);
}
