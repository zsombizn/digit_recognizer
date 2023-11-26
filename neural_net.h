#ifndef NNET_H
#define NNET_H
#define M_index(A, i, j) (A)->data[(i) * (A)->columns + (j)]
#include <stddef.h>
#include <stdint.h>


typedef struct {
    unsigned int rows;
    unsigned int columns;
    double *data;
} Matrix;


typedef struct {
    uint8_t label;
    uint8_t *data_array;
} Example;


// neural network, multi layer perceptron
// each layers weight matrix is stored in the weights array
// and the corresponding bias vector in the biases array
// same with the activation functions, as function pointers in the activaiton array
// (they must have the double func(double) prototype)
// 
typedef double (*activation_f)(double);


typedef struct {
    int depth;
    Matrix *weights;
    Matrix *biases;
    activation_f* activations;
} MLP;


Matrix *newMatrix(int rows, int columns);
void newMatrixAt(Matrix *dest, int rows, int columns);
void freeMatrix(Matrix *dest);
Matrix *product_M(Matrix *A, Matrix *B);
Matrix *scalar_p_M(Matrix *A, double n);
Matrix *transpose_M(Matrix *A);
void sum_M(Matrix *Res, Matrix *A);
void print_M(Matrix *A);
void fill_from_array_M(Matrix *A, int8_t *arr, unsigned int len);
void rand_M(Matrix *A, double min, double max);
void ascii_print_M(Matrix *A);
MLP *newMLP(int depth, int input_size, int output_size, int hidden_layer_size, activation_f* activations);
void freeMLP(MLP *dest);
void add_row_V_M(Matrix* dest, Matrix* row_V);
void activate_M(Matrix* dest, activation_f a);
Matrix* feedForward(MLP* network, Matrix* input);
double ReLu(double x);


#endif