#include "io.h"
#include "neural_net.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void export_examples_BMP(Example *images, size_t num_examples) {
    char dirname[10];
    char filename[40];
    Matrix *temp_M = newMatrix(28, 28);

    if (mkdir("images", 0755) != 0) {
        fprintf(stderr, "Failed creating directory!\n");
        exit(EXIT_FAILURE);
    }

    for (int n = 0; n < 10; n++) {
        sprintf(dirname, "images/%d", n);
        if (mkdir(dirname, 0755) != 0) {
            fprintf(stderr, "Failed creating directory!\n");
            exit(EXIT_FAILURE);
        }
    }

    for (size_t i = 0; i < num_examples; i++) {
        sprintf(filename, "images/%d/%d.bmp", images[i].label, i);

        fill_from_array_M(temp_M, images[i].data_array, 28 * 28);

        write_Matrix_BMP(filename, temp_M);
        if (i % 1000 == 0) {
            putchar('#');
            fflush(stdin);
        }

    }
    putchar('\n');
    free(temp_M);
}


int main() {
    srand(clock());

    // Matrix *M = newMatrix(8, 1);

    // print_M(M);

    // Matrix *X = newMatrix(4, 2);
    // Matrix *W = newMatrix(2, 2);

    // double data_x[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};

    // fill_from_array_M(X, data_x, sizeof(data_x)/sizeof(data_x[0]));

    // printf("X------------------\n");
    // print_M(X);


    // double data_w[] = {1.0, 1.0, 1.0, 1.0};

    // fill_from_array_M(W, data_w, sizeof(data_w)/sizeof(data_w[0]));

    // printf("W------------------\n");
    // print_M(W);

    // Matrix *Prod = product_M(X, W);

    // printf("Prod---------------\n");
    // print_M(Prod);

    // free(Prod);

    // rand_M(M, 0.0, 5);
    // printf("Rand----------------\n");
    
    // print_M(M);

    // fill_from_array_M(M, data_x, sizeof(data_x)/sizeof(data_x[0]));
    // Matrix *Tr = transpose_M(M);
    // printf("Tr------------------\n");

    // print_M(Tr);

    
    // Prod = product_M(M, Tr);
    // printf("M*M-----------------\n");
    // print_M(Prod);


    // rand_M(X, 0, 50);
    // printf("X------------------\n");
    // print_M(X);
    // sum_M(X, X);
    // printf("X+X-----------------\n");
    // print_M(X);


    // Matrix *Scalar_p = scalar_p_M(X, 3.14); 
    // printf("Scalar_p--3.14------\n");
    // print_M(Scalar_p);

    // printf("Random int: %d, %d, %d\n", randint(1, 5), randint(5, 10), randint(6, 7));

    // size_t len = 3;
    // int *arr = malloc(sizeof(int) * len);

    // arr[0] = 2;
    // arr[1] = 1;
    // arr[2] = 0;

    // swap(&arr[0], &arr[2], sizeof(int));

    // printf("swapped: %d %d %d\n", arr[0], arr[1], arr[2]);

    // shuffle(arr, sizeof(int), len);

    // printf("shuffled: %d %d %d\n", arr[0], arr[1], arr[2]);

    size_t num_examples;

    Example *images;
    
    read_MNIST_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", &images, &num_examples);

    shuffle(images, sizeof(Example), num_examples);
    
    export_examples_BMP(images, num_examples);

    
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


    for (size_t i = 0; i < num_examples; i++) {
        free(images[i].data_array);
    }

    free(images);

    return 0;
}