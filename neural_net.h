#ifndef NNET_H
#define NNET_H
#define M_index(A, i, j) (A)->data[(i) * (A)->columns + (j)]
#include <stddef.h>
#include <stdint.h>


/**
 * @brief Structure representing a matrix.
 */
typedef struct {
    unsigned int rows;
    unsigned int columns;
    double *data;
} Matrix;


/**
 * @brief Structure representing an example.
 */
typedef struct {
    uint8_t label;        /**< Label associated with the example. */
    uint8_t *data_array;  /**< Array containing data for the examples features. */
} Example;


/**
 * @brief Activation function type for neural networks.
 *
 * Function pointers with the prototype `void func(Matrix*)` are used to represent activation functions.
 */
typedef void (*activation_f)(Matrix *);


/**
 * @brief Structure representing a Multi-Layer Perceptron (MLP).
 *
 * Each layer's weight matrix is stored in the weights array,
 * the corresponding bias vector in the biases array,
 * and the activation function as function pointers in the activate array.
 */
typedef struct {
    int depth;               /**< Number of layers in the MLP. (Excluding input layer)*/
    Matrix *weights;         /**< Array of weight matrices for each layer. */
    Matrix *biases;          /**< Array of bias vectors for each layer. */
    activation_f* activate;  /**< Array of activation functions for each layer. */
} MLP;


/**
 * @brief Enumeration representing activation functions for neural networks.
 * 
 * These are used to encode activation functions as integers, this way
 * they can be stored in binary files, independent of the function pointer values.
 */
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


#endif