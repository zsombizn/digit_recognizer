#ifndef NNET_H
#define NNET_H
#define M_index(A, i, j) (A)->data[(i) * (A)->columns + (j)]
#define MAX(A, B) (A) > (B) ? (A) : (B)
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
// same with the activation functions, as function pointers in the activate array
// (they must have the void func(Matrix*) prototype)
// 
typedef void (*activation_f)(Matrix *);


typedef struct {
    int depth;
    Matrix *weights;
    Matrix *biases;
    activation_f* activate;
} MLP;


enum activation_function {ReLu_CODE, sigmoid_CODE, softmax_CODE};


Matrix *newMatrix(int rows, int columns);
void newMatrixAt(Matrix *dest, int rows, int columns);
void freeMatrix(Matrix *dest);
Matrix *product_M(Matrix *A, Matrix *B);
Matrix *scalar_p_M(Matrix *A, double n);
Matrix *transpose_M(Matrix *A);
void sum_M(Matrix *Res, Matrix *A);
void print_M(Matrix *A);
void fill_from_array_M(Matrix *A, double *arr, unsigned int len);
void rand_M(Matrix *A, double min, double max);
void ascii_print_M(Matrix *A);
MLP *newMLP(int depth, int input_size, int hidden_layer_size, int output_size, activation_f* activate);
void freeMLP(MLP *dest);
void add_row_V_M(Matrix* dest, Matrix* row_V);
Matrix* feedForward(MLP* network, Matrix* input);
double ReLu(double x);
void ReLu_M(Matrix* dest);
double sigmoid(double x);
void sigmoid_M(Matrix *dest);
void soft_max_M(Matrix *dest);
double MSE(Matrix* output, Matrix* y);
double cross_entropy(Matrix *output, Matrix *y);


#endif