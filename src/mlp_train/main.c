// bash:gcc -o MLP_train backward.c forward.c model_initializer.c train_acc_param_avg.c main.c model_saver.c reset.c data_loader.c parameter_update.c test_acc_loss.c -lm
#include <time.h>
#include "mlp.h"
#include "data.h"

int main(void) {
    data_loader();
    model_initializer();
    for (epoch = 1; ; epoch++) {
        forward();
        backward();
        parameter_update();
        train_acc_param_avg();
        if (epoch % 100 == 0) {
            test_acc_loss(TEST_IMG_NUM);
            printf("Epoch %4d, Acc_Rate %-9.6f%%, MAX_Acc_Rate %-9.6f%%, Loss %f, MIN_Loss %f\n", epoch, test_accuracy / (float) (TEST_IMG_NUM * 0.01), test_accuracy_max / (float) (TEST_IMG_NUM * 0.01), test_loss, test_loss_min);
        }
        if (test_loss < test_loss_save) {
            test_loss_save = test_loss;
            model_saver();
        }
        img_tag = (img_tag + BATCH_SIZE) % TRAIN_IMG_NUM;
        printf("Epoch %5d, Acc_Rate %-9.6f%%, MAX_Acc_Rate %-9.6f%%, Loss %f, MIN_Loss %f\n", epoch, train_accuracy / (float) (BATCH_SIZE * 0.01), train_accuracy_max / (float) (BATCH_SIZE * 0.01), train_loss, train_loss_min);
        reset();
    }
    
    return 0;
}