#include "mlp.h"
#include <string.h>

void data_loader(void) {
    FILE *pash_p = fopen("path.txt", "r");
    if (!pash_p) {
        printf("Error opening path file.\n");
        exit(1);
    }
    char train_img_path[530] = {0};
    char train_label_path[530] = {0};
    char test_img_path[530] = {0};
    char test_label_path[530] = {0};
    int ch;
    int char_num = 0;
    while ((ch = fgetc(pash_p)) != EOF && ch != '\n' && ch != '\r' && ch != ' ' && ch != '\t') {
        if (ch == '\\') {
            ch = '/';  // 替换反斜杠为正斜杠
        }
        train_img_path[strlen(train_img_path)] = (char)ch;
        train_label_path[strlen(train_label_path)] = (char)ch;
        test_img_path[strlen(test_img_path)] = (char)ch;
        test_label_path[strlen(test_label_path)] = (char)ch;
        char_num++;
        if (char_num == 500) {
            printf("Path too long in path file.\n");
            fclose(pash_p);
            exit(1);
        }
    }
    fclose(pash_p);
    for (int i = 0; i < 25; i++) {
        train_img_path[strlen(train_img_path)] = "/train-images.idx3-ubyte"[i];
        train_label_path[strlen(train_label_path)] = "/train-labels.idx1-ubyte"[i];
    }
    for (int i = 0; i < 24; i++) {
        test_img_path[strlen(test_img_path)] = "/t10k-images.idx3-ubyte"[i];
        test_label_path[strlen(test_label_path)] = "/t10k-labels.idx1-ubyte"[i];
    }
    FILE *train_img = fopen(train_img_path, "rb");
    FILE *train_label = fopen(train_label_path, "rb");
    FILE *test_img = fopen(test_img_path, "rb");
    FILE *test_label = fopen(test_label_path, "rb");

    if (!train_img) {
        printf("Error opening train_image file.\n");
        exit(1);
    }
    if (!train_label) {
        printf("Error opening train_label file.\n");
        fclose(train_img);
        exit(1);
    }
    if (!test_img) {
        printf("Error opening test_image file.\n");
        fclose(train_img);
        fclose(train_label);
        exit(1);
    }
    if (!test_label) {
        printf("Error opening test_label file.\n");
        fclose(train_img);
        fclose(train_label);
        fclose(test_img);
        exit(1);
    }

    int a[4];
    fread(a, sizeof(int), 4, train_img);
    fread(a, sizeof(int), 2, train_label);
    fread(a, sizeof(int), 4, test_img);
    fread(a, sizeof(int), 2, test_label);

    unsigned char im;
    for (int i = 0; i < TRAIN_IMG_NUM; i++) {
        for (int j = 0; j < INPUT_LAYER_SIZE; j++) {
            fread(&im, sizeof(unsigned char), 1, train_img);
            train_images[j][i] = im / 255.0f;  // 归一化
       }
    }
    fread(train_labels, sizeof(unsigned char), TRAIN_IMG_NUM, train_label);
    
    for (int i = 0; i < TEST_IMG_NUM; i++) {
        for (int j = 0; j < INPUT_LAYER_SIZE; j++) {
            fread(&im, sizeof(unsigned char), 1, test_img);
            test_images[j][i] = im / 255.0f;  // 归一化
       }
    }
    fread(test_labels, sizeof(unsigned char), TEST_IMG_NUM, test_label);

    fclose(train_img);
    fclose(train_label);
    fclose(test_img);
    fclose(test_label);
}
