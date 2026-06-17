#include "mlp.h"
#include <math.h>
// #include <x86_64-linux-gnu/bits/mathcalls.h>

void test_acc_loss(int size) {
    test_loss = 0;
    test_accuracy = 0;
    double *p1, *p2;
    p1 = (double*)malloc(sizeof(double) * LAYER_MAX);
    p2 = (double*)malloc(sizeof(double) * LAYER_MAX);
    if (!p1 || !p2) {
        printf("Memory allocation failed in test_acc_loss function.\n");
        exit(1);
    }
    
    for (int i = 0; i < size; i++) {
        //forward
        for (int j = 0; j < HIDDEN_LAYER_COUNT + 1; j++) {
            double* p3 = p1;  p1 = p2;  p2 = p3;
            for (int k = 0; k < layer_array[j + 1]; k++) {
                p1[k] = b[j][k];
                for (int l = 0; l < layer_array[j]; l++) {
                    p1[k] += w[j][k][l] * ((j == 0) ? (test_images[l][i]) : ACT_FUNC(p2[l]));
                }
            }
        }
        // loss
        double e_z[OUTPUT_LAYER_SIZE] = {0};
        double sum = 0;
        for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            e_z[j] = exp(p1[j]);
            sum += e_z[j];
        }
        for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            test_loss += fabs((e_z[j] / sum) - (j == (int)(test_labels[i]) ? 1 : 0));
        }
        // acc
        int n = 0;
        for (int j = 1; j < OUTPUT_LAYER_SIZE; j++) {
            if (p1[j] > p1[n]) {
                n = j;
            }
        }
        if (((int)test_labels[i]) == n) {
            test_accuracy++;
        }
    }
    if (test_accuracy > test_accuracy_max)
        test_accuracy_max = test_accuracy;
    
    free(p1);
    free(p2);
    if (test_loss < test_loss_min || test_loss_min == 0) {
        test_loss_min = test_loss;
    }
}