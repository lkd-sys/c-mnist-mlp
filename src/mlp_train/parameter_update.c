#include "mlp.h"

void parameter_update(void) {
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        for (int j = 0; j < layer_array[i + 1]; j++) {
            b[i][j] -= dz[i][j] * learning_rate[1][i];
            for (int k = 0; k < layer_array[i]; k++) {
                w[i][j][k] -= dw[i][j][k] * learning_rate[0][i];
            }
        }
    }
}
