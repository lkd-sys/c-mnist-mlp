// #include <time.h>
#include "mlp.h"

void model_initializer(void) {
    srand((unsigned int) 153264); 
    // srand((unsigned int) time(NULL)); // 随机数种子
    int r;

    // 偏执初始化
    b = (double**)malloc((HIDDEN_LAYER_COUNT + 1) * sizeof(double*));
    if (!b) {
        printf("Memory allocation failed for b in model_initializer.\n");
        exit(1);
    }
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        b[i] = (double*) malloc(layer_array[i + 1] * sizeof(double));
        if (!b[i]) {
            printf("Memory allocation failed for b[%d] in model_initializer.\n", i);
            exit(1);
        }
        for (int j = 0; j < layer_array[i + 1]; j++) {
            r = rand();
            b[i][j] = /*(((r % 2) == 0) ? 1 : -1) */ (r/((double)RAND_MAX)) * 1.0;
        }
    }

    // 权重初始化
    w = (double***)malloc((HIDDEN_LAYER_COUNT + 1) * sizeof(double**));
    if (!w) {
        printf("Memory allocation failed for w in model_initializer.\n");
        exit(1);
    }
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        w[i] = (double**)malloc(layer_array[i + 1] * sizeof(double*));
        if (!w[i]) {
            printf("Memory allocation failed for w[%d] in model_initializer.\n", i);
            exit(1);
        }
        for (int j = 0; j < layer_array[i + 1]; j++) {
            w[i][j] = (double*)malloc(layer_array[i] * sizeof(double));
            if (!w[i][j]) {
                printf("Memory allocation failed for w[%d][%d] in model_initializer.\n", i, j);
                exit(1);
            }
            for (int k = 0; k < layer_array[i]; k++) {
                r = rand();
                int aaa = /*(((r % 2) == 0) ? 1 : -1) */ (r/((double)RAND_MAX));
                w[i][j][k] = 0.1 * aaa;
            }
        }
    }

    // 中间值初始化
    z = (double***) malloc((HIDDEN_LAYER_COUNT + 1) * sizeof(double**));
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        z[i] = (double**)malloc(BATCH_SIZE * sizeof(double*));
        for (int j = 0; j < BATCH_SIZE; j++) {
            z[i][j] = (double*)malloc(layer_array[i + 1] * sizeof(double));
            if (!z[i][j]) {
                printf("Memory allocation failed for z[%d][%d] in model_initializer.\n", i, j);
                exit(1);
            }
        }
    }

    dw = (double***) malloc((HIDDEN_LAYER_COUNT + 1) * sizeof(double**));
    if (!dw) {
        printf("Memory allocation failed for dw in model_initializer.\n");
        exit(1);
    }
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        dw[i] = (double**)malloc(layer_array[i + 1] * sizeof(double*));
        if (!dw[i]) {
            printf("Memory allocation failed for dw[%d] in model_initializer.\n", i);
            exit(1);
        }
        for (int j = 0; j < layer_array[i + 1]; j++) {
            dw[i][j] = (double*)malloc(layer_array[i] * sizeof(double));
            if (!dw[i][j]) {
                printf("Memory allocation failed for dw[%d][%d] in model_initializer.\n", i, j);
                exit(1);
            }
        }
    }
    dz = (double**)malloc((HIDDEN_LAYER_COUNT + 1) * sizeof(double*));
    if (!dz) {
        printf("Memory allocation failed for dz in model_initializer.\n");
        exit(1);
    }
    for (int i = 0; i < HIDDEN_LAYER_COUNT + 1; i++) {
        dz[i] = (double*) malloc(layer_array[i + 1] * sizeof(double));
        if (!dz[i]) {
            printf("Memory allocation failed for dz[%d] in model_initializer.\n", i);
            exit(1);
        }
    }
}
