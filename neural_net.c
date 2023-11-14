#include "neural_net.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


Matrix *newMatrix(int rows, int columns) {
    Matrix *Res = malloc(sizeof(Matrix));
    Res->rows = rows;
    Res->columns = columns;
    Res->data = malloc(sizeof(double) * rows * columns);

    if (Res->data == NULL) {
        fprintf(stderr, "Failed allocation!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            M_index(Res, i, j) = 0.0;
        }
    }

    return Res;
}


Matrix *product_M(Matrix *A, Matrix *B) {
    Matrix *Res;
    if (A->columns != B->rows) {
        fprintf(stderr, "Not matching column and row size!\n");
        exit(EXIT_FAILURE);
    }

    Res = newMatrix(A->rows, B->columns);

    for (unsigned int i = 0; i < Res->rows; i++) {
        for (unsigned int j = 0; j < Res->columns; j++) {
            for (unsigned int k = 0; k < B->columns; k++) {
                M_index(Res, i, j) += M_index(A, i, k) * M_index(B, k, j);
            } 
        }
    }

    return Res;

}


Matrix *scalar_p_M(Matrix *A, double n) {
    Matrix *Res = newMatrix(A->rows, A->columns);

    for (unsigned int i = 0; i < Res->rows; i++) {
        for (unsigned int j = 0; j < Res->columns; j++) {
            M_index(Res, i, j) = M_index(A, i, j) * n;
        }
    }

    return Res;

}


Matrix *transpose_M(Matrix *A) {
    Matrix *Res = newMatrix(A->columns, A->rows);

    for (unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            M_index(Res, j, i) = M_index(A, i, j);
        }
    }

    return Res;

}


void sum_M(Matrix *Res, Matrix *A) {
    if (Res->rows != A->rows && Res->columns != A->columns) {
        fprintf(stderr, "Not matching dimensions!\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < Res->rows; i++) {
        for (unsigned int j = 0; j < Res->columns; j++) {
            M_index(Res, i, j) += M_index(A, i, j);
        }
    }

}


void print_M(Matrix *A) {
    for (unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            printf("%lf, ", M_index(A, i, j));
        }
        printf("\n");
    }
}


void fill_from_array_M(Matrix *A, double *arr, unsigned int len) {
    if (A->rows * A->columns != len) {
        fprintf(stderr, "Invalid number of elements in array!\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            M_index(A, i, j) = arr[i * A->columns + j];
        }
    }
}


double randf(double min, double max) {
    double scale = rand() / (double) RAND_MAX;
    return min + scale * (max - min);
}


int randint(int min, int max) {
    return min + ( rand() % (max-min+1) );
}


void rand_M(Matrix *A, double min, double max) {
    for (unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            M_index(A, i, j) = randf(min, max);
        }
    }
}


void swap(void *a, void *b, size_t size) {
    char *tmp = malloc(size);
    memcpy(tmp, a, size);
    memcpy(a, b, size);
    memcpy(b, tmp, size);

    free(tmp);
}


void shuffle(void *arr, size_t size, size_t len) {
    int n;
    for (size_t i = 0; i < len; i++) {
        n = randint(0, len-1);

        // arr[i] <-> arr[n]
        swap(arr + (size * i), arr + (size * n), size);
    }
}