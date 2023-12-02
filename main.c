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

    printf("WIP -- demo operations\n");
    printf("MLP implementation of XOR with preset weigths and biases:\n");

    // 3 layers, input 2, hidden 2, output 1,
    activation_f acts[] = {&ReLu_M, &ReLu_M};
    MLP *net = newMLP(2, 2, 2, 1, acts);
    MLP_data *neuron_values = newMLP_data(net, 4);

    // hidden layer weights and biases
    double data_w0[] = {1.0, 1.0, 1.0, 1.0};
    fill_from_array_M(&(net->weights[0]), data_w0, 4);

    double data_b0[] = {0.0, -1.0};
    fill_from_array_M(&(net->biases[0]), data_b0, 2);

    // output layer weights (0 for biases)
    double data_w1[] = {1.0, -2.0};
    fill_from_array_M(&(net->weights[1]), data_w1, 2);


    // input data
    Matrix *in = newMatrix(4, 2);
    double data_in[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};

    fill_from_array_M(in, data_in, sizeof(data_in) / sizeof(data_in[0]));

    printf("\ninput-------------\n");

    print_M(in);

    Matrix *res = feedForward(net, in, neuron_values);

    printf("\nlayers-------\n");

    for (int i = 0; i < neuron_values->depth; i++) {
        printf("\nlayer %d:\n", i);
        print_M(&(neuron_values->pre_activated_values[i]));
        printf("activated:\n");
        print_M(&(neuron_values->activated_values[i]));
    }

    printf("\noutput------------\n");
    print_M(res);

    write_MLP("xor.bin", net);

    MLP *readed = read_MLP("xor.bin");
    neuron_values->origin = readed;

    freeMatrix(res);

    res = feedForward(readed, in, neuron_values);
    
    MLP *grads = newMLP(2, 2, 2, 1, NULL);
    
    back_propagate(readed, grads, neuron_values, in, res, 4);

    printf("\noutput2-----------\n");
    print_M(res);


    freeMatrix(in);
    freeMatrix(res);
    freeMLP(net);

    // demonstration of matrix operations:
    printf("\nNew matrix M------\n");

    Matrix *M = newMatrix(8, 1);
    print_M(M);

    Matrix *X = newMatrix(4, 2);
    Matrix *W = newMatrix(2, 2);

    double data_x[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};

    fill_from_array_M(X, data_x, sizeof(data_x)/sizeof(data_x[0]));

    printf("\nX-----------------\n");
    print_M(X);


    double data_w[] = {1.0, 1.0, 1.0, 1.0};

    fill_from_array_M(W, data_w, sizeof(data_w)/sizeof(data_w[0]));

    printf("\nW-----------------\n");
    print_M(W);

    Matrix *Prod = product_M(X, W);

    printf("\nProduct-----------\n");
    print_M(Prod);

    freeMatrix(Prod);

    printf("\nSoftmax-X---------\n");
    soft_max_M(X);
    print_M(X);

    rand_M(M, 0.0, 5);
    printf("\nRand--------------\n");
    
    print_M(M);

    fill_from_array_M(M, data_x, sizeof(data_x)/sizeof(data_x[0]));
    Matrix *Tr = transpose_M(M);
    printf("\nTr----------------\n");

    print_M(Tr);

    
    Prod = product_M(M, Tr);
    printf("\nM*M----------------\n");
    print_M(Prod);

    freeMatrix(Prod);
    freeMatrix(Tr);


    rand_M(X, 0, 50);
    printf("\nX------------------\n");
    print_M(X);
    sum_M(X, X);
    printf("\nX+X----------------\n");
    print_M(X);


    Matrix *Scalar_p = scalar_p_M(X, 3.14); 
    printf("\nScalar_p--3.14-----\n");
    print_M(Scalar_p);

    printf("\nRandom int: %d, %d, %d\n", randint(1, 5), randint(5, 10), randint(6, 7));

    size_t len = 3;
    int *arr = malloc(sizeof(int) * len);

    arr[0] = 2;
    arr[1] = 1;
    arr[2] = 0;

    swap(&arr[0], &arr[2], sizeof(int));

    printf("\nswapped: %d %d %d\n", arr[0], arr[1], arr[2]);

    shuffle(arr, sizeof(int), len);

    printf("\nshuffled: %d %d %d\n", arr[0], arr[1], arr[2]);

    printf("\nWriting 'White.bmp'\n");
    Matrix *random = newMatrix(125, 125);
    for (unsigned int i = 0; i < random->rows; i++) {
        for (unsigned int j = 0; j < random->columns; j++) {
            M_index(random, i, j) = 255;
        }
    }
    

    write_Matrix_BMP("White.bmp", random);

    printf("\nWriting 'Random.bmp'\n");
    rand_M(random, 0, 255);

    write_Matrix_BMP("Random.bmp", random);


    freeMatrix(random);

    double rand_x = randf(0, 1);
    printf("\nsigmoid %lf: %lf\n", rand_x, sigmoid(rand_x));

    printf("MAX(%d, %d) = %d\n", arr[0], arr[1], MAX(arr[0], arr[1]));

    Matrix *test_output = newMatrix(3, 3);
    rand_M(test_output, -2, 2);
    soft_max_M(test_output);

    Matrix *test_correct_out = newMatrix(3, 3);
    M_index(test_correct_out, 0, 0) = 1.0;
    M_index(test_correct_out, 1, 1) = 1.0;
    M_index(test_correct_out, 1, 0) = 1.0;
    M_index(test_correct_out, 2, 2) = 1.0;

    printf("\nTest ouput:\n");
    print_M(test_output);

    printf("\nCorrect output:\n");
    print_M(test_correct_out);

    printf("\nCross entropy: %lf\n", cross_entropy(test_output, test_correct_out));

    freeMatrix(test_output);
    freeMatrix(test_correct_out);


}


void export_examples_BMP(Example *images, size_t num_examples) {
    char dirname[10];
    char filename[40];
    Matrix *temp_M = newMatrix(28, 28);
    double *data = malloc(sizeof(double)*28*28);
    check_malloc(data);

    check_mkdir("images");

    for (int n = 0; n < 10; n++) {
        sprintf(dirname, "images/%d", n);
        check_mkdir(dirname);
    }

    for (size_t i = 0; i < num_examples; i++) {
        sprintf(filename, "images/%d/%d.bmp", images[i].label, (int) i);

        uint8_to_double(data, images[i].data_array, 28*28);

        fill_from_array_M(temp_M, data, 28 * 28);

        write_Matrix_BMP(filename, temp_M);
        if (i % 1000 == 0) {
            putchar('#');
            fflush(stdout);
        }

    }
    putchar('\n');
    free(temp_M);
    free(data);
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


