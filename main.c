#include "io.h"
#include "neural_net.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

enum task {HELP, DEMO, EXPORT_MNIST};

void export_examples_BMP(Example *images, size_t num_examples);
void parse_opt(char *option, int *task_arr, int *opt_argc, char **option_argv);
void print_help(const char *exec_name);
void demo();
void export_MNIST(const char* fname_images, const char *fname_labels);


int main(int argc, char* argv[]) {
    srand(time(NULL));

    //TEST
    activation_f acts[] = { &ReLu, &ReLu };
    MLP* net = newMLP(2, 2, 1, 2, acts);

    int8_t data_W[] = { 1, 1, 1, 1 };
    fill_from_array_M(&(net->weights[0]), data_W, 4);

    int8_t data_c[] = { 0, -1 };
    fill_from_array_M(&(net->biases[0]), data_c, 2);

    int8_t data_w[] = { 1, -2 };
    fill_from_array_M(&(net->weights[1]), data_w, 2);


    Matrix* in = newMatrix(4, 2);
    int8_t data_x[] = { 0, 0, 0, 1, 1, 0, 1, 1 };

    fill_from_array_M(in, data_x, sizeof(data_x) / sizeof(data_x[0]));

    printf("input: -------------\n");

    print_M(in);

    Matrix *res = feedForward(net, in);

    printf("output: -------------\n");
    print_M(res);


    freeMLP(net);


    //TEST

    // each command line option creates a task, which is stored here, with a 
    // non zero number, which is the index in the opt_argv, where the arguments
    // of the given option are stored
    int task_arr[] = {0, 0, 0};

    // opt_argv is an array of strings, holds the arguments of the different options
    char **opt_argv = malloc(sizeof(char *) * argc);
    for (int i = 0; i < argc; i++) {
        opt_argv[i] = NULL;
    }

    int opt_argc = 1;

    if (argc == 1) {
        print_help(argv[0]);
    } else {
        for (int n = 1; n < argc; n++) {
            parse_opt(argv[n], task_arr, &opt_argc, opt_argv);
        }
    }

    if (task_arr[HELP] != 0) {
        print_help(argv[0]);
        return 0;
    }

    if (task_arr[DEMO] != 0) {
        demo();
    }

    if (task_arr[EXPORT_MNIST] != 0) {
        export_MNIST(opt_argv[task_arr[EXPORT_MNIST]], opt_argv[task_arr[EXPORT_MNIST]+1]);
    }


    return 0;
    
    
}


void export_examples_BMP(Example *images, size_t num_examples) {
    char dirname[10];
    char filename[40];
    Matrix *temp_M = newMatrix(28, 28);

    check_mkdir("images");

    for (int n = 0; n < 10; n++) {
        sprintf(dirname, "images/%d", n);
        check_mkdir(dirname);
    }

    for (size_t i = 0; i < num_examples; i++) {
        sprintf(filename, "images/%d/%d.bmp", images[i].label, i);

        fill_from_array_M(temp_M, images[i].data_array, 28 * 28);

        write_Matrix_BMP(filename, temp_M);
        if (i % 1000 == 0) {
            putchar('#');
            fflush(stdout);
        }

    }
    putchar('\n');
    free(temp_M);
}


void parse_opt(char *option, int *task_arr, int *opt_argc, char **option_argv) {
    if (option[0] != '-') {
        option_argv[*opt_argc] = malloc(sizeof(char) * strlen(option) + 1);
        strcpy(option_argv[*opt_argc], option);
        *opt_argc += 1;
    } else if (strcmp(option, "-d") == 0 || strcmp(option, "--demo") == 0) {
        task_arr[DEMO] = *opt_argc;
    } else if (strcmp(option, "-E") == 0 || strcmp(option, "--export-MNIST") == 0) {
        task_arr[EXPORT_MNIST] = *opt_argc;
    } else if (strcmp(option, "-h") == 0 || strcmp(option, "--help") == 0) {
        task_arr[HELP] = *opt_argc;
    } else {
        printf("Invalid option: '%s'\n", option);
        task_arr[HELP] = *opt_argc;
    }
}


void print_help(const char* exec_name) {
    printf("Usage: %s [OPTION]...\n", exec_name);
    printf("Options:\n   -h, --help: Print this message\n");
    printf("   -d, --demo: Show the inner workings of the implemented functions\n");
    printf("   -E images labels, --export-MNIST images labels: export MNIST data from images and the corresponding labels file.\n");
}


void demo(){
    Matrix *M = newMatrix(8, 1);

    print_M(M);

    Matrix *X = newMatrix(4, 2);
    Matrix *W = newMatrix(2, 2);

    uint8_t data_x[] = {0, 0, 0, 1, 1, 0, 1, 1};

    fill_from_array_M(X, data_x, sizeof(data_x)/sizeof(data_x[0]));

    printf("X------------------\n");
    print_M(X);


    uint8_t data_w[] = {1, 1, 1, 1};

    fill_from_array_M(W, data_w, sizeof(data_w)/sizeof(data_w[0]));

    printf("W------------------\n");
    print_M(W);

    Matrix *Prod = product_M(X, W);

    printf("Prod---------------\n");
    print_M(Prod);

    freeMatrix(Prod);

    rand_M(M, 0.0, 5);
    printf("Rand----------------\n");
    
    print_M(M);

    fill_from_array_M(M, data_x, sizeof(data_x)/sizeof(data_x[0]));
    Matrix *Tr = transpose_M(M);
    printf("Tr------------------\n");

    print_M(Tr);

    
    Prod = product_M(M, Tr);
    printf("M*M-----------------\n");
    print_M(Prod);

    freeMatrix(Prod);
    freeMatrix(Tr);


    rand_M(X, 0, 50);
    printf("X------------------\n");
    print_M(X);
    sum_M(X, X);
    printf("X+X-----------------\n");
    print_M(X);


    Matrix *Scalar_p = scalar_p_M(X, 3.14); 
    printf("Scalar_p--3.14------\n");
    print_M(Scalar_p);

    printf("Random int: %d, %d, %d\n", randint(1, 5), randint(5, 10), randint(6, 7));

    size_t len = 3;
    int *arr = malloc(sizeof(int) * len);

    arr[0] = 2;
    arr[1] = 1;
    arr[2] = 0;

    swap(&arr[0], &arr[2], sizeof(int));

    printf("swapped: %d %d %d\n", arr[0], arr[1], arr[2]);

    shuffle(arr, sizeof(int), len);

    printf("shuffled: %d %d %d\n", arr[0], arr[1], arr[2]);

    Matrix *random = newMatrix(125, 125);
    for (unsigned int i = 0; i < random->rows; i++) {
        for (unsigned int j = 0; j < random->columns; j++) {
            M_index(random, i, j) = 255;
        }
    }
    

    write_Matrix_BMP("White.bmp", random);

    rand_M(random, 0, 255);

    write_Matrix_BMP("Random.bmp", random);


    freeMatrix(random);
}


void export_MNIST(const char* fname_images, const char* fname_labels) {
    size_t num_examples;

    Example *images;
    
    read_MNIST_data(fname_images, fname_labels, &images, &num_examples);
    
    export_examples_BMP(images, num_examples);

    for (size_t i = 0; i < num_examples; i++) {
        free(images[i].data_array);
    }

    free(images);

}


