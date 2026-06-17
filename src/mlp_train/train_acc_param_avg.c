#include "mlp.h"
#include <math.h>

void train_acc_param_avg(void) {
    // 1
    train_accuracy = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        int n = 0;
        for (int j = 1; j < OUTPUT_LAYER_SIZE; j++) {
            if (z[HIDDEN_LAYER_COUNT][i][j] > z[HIDDEN_LAYER_COUNT][i][n]) {
                n = j;
            }
        }
        if (((int)train_labels[(img_tag + i) % TRAIN_IMG_NUM]) == n) {
            train_accuracy++;
        }
    }
    if (train_accuracy > train_accuracy_max)
        train_accuracy_max = train_accuracy;
    // 2
    param_avg_abs = 0;
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        for (int j = 0; j < layer_array[i + 1]; j++) {
            param_avg_abs += b[i][j] / param_num;
            for (int k = 0; k < layer_array[i]; k++) {
                param_avg_abs += w[i][j][k] / param_num;
            }
        }
    }
    // 3
    train_loss = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            train_loss += fabs(p[i][j] - ((j == train_labels[(img_tag + i) % TRAIN_IMG_NUM]) ? 1 : 0));
        }
    }
    if (train_loss < train_loss_min || train_loss_min == 0) {
        train_loss_min = train_loss;
    }
}
