#include "mlp.h"

void reset(void) {
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        for (int j = 0; j < layer_array[i + 1]; j++) {
            dz[i][j] = 0.0f;
            for (int k = 0; k < layer_array[i]; k++) {
                dw[i][j][k] = 0.0f;
            }
        }
    }
}