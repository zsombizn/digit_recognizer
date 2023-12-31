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
typedef void (*activation_f_M)(Matrix *);

typedef double (*activation_f)(double);

typedef struct {
    int depth;
    Matrix *weights;
    Matrix *biases;
    activation_f_M* activate;
} MLP;


// It contains values for one training example in each row, 
// it holds data for a batch of examples, this way these values don't have to be
// computed again during backwards propagation
// pre activated values of the neurons (one matrix for each layer)
// same with activated
// origin: the MLP which it belongs to
typedef struct {
    int depth;
    MLP *origin;
    Matrix *pre_activated_values;
    Matrix *activated_values;
} MLP_data;


enum activation_functions_M{ReLu_CODE, sigmoid_CODE, softmax_CODE};


Matrix *newMatrix(int rows, int columns);
void newMatrixAt(Matrix *dest, int rows, int columns);
void freeMatrix(Matrix *dest);
void resetMatrix(Matrix* dest);
Matrix *product_M(Matrix *A, Matrix *B);
Matrix *scalar_p_M(Matrix *A, double n);
Matrix *transpose_M(Matrix *A);
void sum_M(Matrix *Res, Matrix *A);
void print_M(Matrix *A);
void fill_from_array_M(Matrix *A, double *arr, unsigned int len);
void rand_M(Matrix *A, double min, double max);
void ascii_print_M(Matrix *A);
void copy_values_M(Matrix *dest, Matrix *source);
void fill_row_from_array_M(Matrix* dest, unsigned int row, double* arr, int len);
void fill_row_from_int_array_scaled_M(Matrix* dest, unsigned int row, uint8_t* arr, int len);
MLP *newMLP(int depth, int input_size, int hidden_layer_size, int output_size, activation_f_M* activate);
void freeMLP(MLP *dest);
MLP_data *newMLP_data(MLP *neural_net, int batch_size);
void freeMLP_data(MLP_data *dest);
void add_row_V_M(Matrix* dest, Matrix* row_V);
void rand_weights_biases(MLP* network);
void feedForward(MLP* network, Matrix *input, Matrix *output, MLP_data *neuron_values);
void back_propagate(MLP *network, MLP *gradients, MLP_data *neuron_values, Matrix *inputs, Matrix *desired_outputs);
void modify_weights_biases(MLP* net, MLP* gradients, double learning_rate);
double ReLu(double x);
double ReLu_d(double x);
void ReLu_M(Matrix* dest);
double sigmoid(double x);
double sigmoid_d(double x);
void sigmoid_M(Matrix *dest);
void softmax_M(Matrix *dest);
double MSE(Matrix* output, Matrix* y);
double cross_entropy(Matrix *output, Matrix *y);
activation_f derivative(activation_f func);


#endif