#include "io.h"
#include "neural_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main() {
    srand(clock());

    Matrix *M = newMatrix(8, 1);

    print_M(M);

    Matrix *X = newMatrix(4, 2);
    Matrix *W = newMatrix(2, 2);

    double data_x[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};

    fill_from_array_M(X, data_x, sizeof(data_x)/sizeof(data_x[0]));

    printf("X------------------\n");
    print_M(X);


    double data_w[] = {1.0, 1.0, 1.0, 1.0};

    fill_from_array_M(W, data_w, sizeof(data_w)/sizeof(data_w[0]));

    printf("W------------------\n");
    print_M(W);

    Matrix *Prod = product_M(X, W);

    printf("Prod---------------\n");
    print_M(Prod);

    free(Prod);

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

    size_t num_examples;

    Matrix *images;
    int *labels;
    
    read_MNIST_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &num_examples, &images, &labels);


    return 0;
}