#ifndef NNET_H
#define NNET_H
#define M_index(A, i, j) (A)->data[(i) * (A)->columns + (j)]
#include <stddef.h>

/**
 * @struct Matrix
 * @brief Contains doubles.
 */
typedef struct {
    unsigned int rows;
    unsigned int columns;
    double *data;
} Matrix;


Matrix *newMatrix(int rows, int columns);
Matrix *product_M(Matrix *A, Matrix *B);
Matrix *scalar_p_M(Matrix *A, double n);
Matrix *transpose_M(Matrix *A);
void sum_M(Matrix *Res, Matrix *A);
void print_M(Matrix *A);
void fill_from_array_M(Matrix *A, double *arr, unsigned int len);
double randf(double min, double max);
int randint(int min, int max);
void rand_M(Matrix *A, double min, double max);
void swap(void *a, void *b, size_t size);
void shuffle(void *arr, size_t size, size_t len);






#endif