#include "mlp.h"
#include <math.h>
// #include <x86_64-linux-gnu/bits/mathcalls.h>

void forward(void) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < HIDDEN_LAYER_COUNT + 1; j++) {
            for (int k = 0; k < layer_array[j + 1]; k++) {
                z[j][i][k] = b[j][k];
                for (int l = 0; l < layer_array[j]; l++) {
                    z[j][i][k] += w[j][k][l] * ((j == 0) ? (train_images[l][(img_tag + i) % TRAIN_IMG_NUM]) : ACT_FUNC(z[j - 1][i][l]));
                }
            }
        }
        // SoftMax
        double e_z[OUTPUT_LAYER_SIZE] = {0};
        double sum = 0;
        for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            e_z[j] = exp((z[HIDDEN_LAYER_COUNT][i][j]));
            sum += e_z[j];
        }
        for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            p[i][j] = e_z[j] / sum;    
        }
    }
}
