#ifndef NNET_H
#define NNET_H
#define M_index(A, i, j) (A)->data[(i) * (A)->columns + (j)]

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
double randf();
void rand_M(Matrix *A, double min, double max);






#endif