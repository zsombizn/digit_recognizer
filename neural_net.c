#include "neural_net.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


/**
 * @brief Create a new matrix with specified dimensions.
 *
 * This function initializes a new matrix with the given number of rows and columns.
 * The matrix is initialized with zeros.
 *
 * @param rows Number of rows in the matrix.
 * @param columns Number of columns in the matrix.
 * @return A pointer to the newly created matrix.
 */
Matrix *newMatrix(int rows, int columns) {
    Matrix *Res = malloc(sizeof(Matrix));
    check_malloc(Res);
    Res->rows = rows;
    Res->columns = columns;
    Res->data = (double *)malloc(sizeof(double) * rows * columns);

    check_malloc(Res->data);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            M_index(Res, i, j) = 0.0;
        }
    }

    return Res;
}


/**
 * @brief Initialize an existing matrix with specified dimensions.
 *
 * This function initializes an existing matrix with the given number of rows and columns.
 * The matrix is initialized with zeros.
 *
 * @param dest Pointer to the destination matrix.
 * @param rows Number of rows in the matrix.
 * @param columns Number of columns in the matrix.
 */
void newMatrixAt(Matrix *dest, int rows, int columns) {
    dest->rows = rows;
    dest->columns = columns;
    dest->data = (double *)malloc(sizeof(double) * rows * columns);

    check_malloc(dest->data);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            M_index(dest, i, j) = 0.0;
        }
    }
}


/**
 * @brief Free memory allocated for a matrix.
 *
 * This function frees the memory allocated for the data array and the matrix itself.
 *
 * @param dest Pointer to the matrix to be freed.
 */
void freeMatrix(Matrix *dest) {
    if (dest == NULL) {
        return;
    }

    free(dest->data);
    free(dest);
}


/**
 * @brief Perform matrix multiplication.
 *
 * This function computes the product of two matrices.
 *
 * @param A Matrix A.
 * @param B Matrix B.
 * @return Resulting matrix of the multiplication.
 */
Matrix *product_M(Matrix *A, Matrix *B) {
    Matrix *Res;
    if (A->columns != B->rows) {
        fprintf(stderr, "Not matching column and row size!\n");
        exit(EXIT_FAILURE);
    }

    Res = newMatrix(A->rows, B->columns);

    for (unsigned int i = 0; i < Res->rows; i++) {
        for (unsigned int j = 0; j < Res->columns; j++) {
            for (unsigned int k = 0; k < B->rows; k++) {
                M_index(Res, i, j) += M_index(A, i, k) * M_index(B, k, j);
            } 
        }
    }

    return Res;

}


/**
 * @brief Multiply a matrix by a scalar.
 *
 * This function multiplies each element of a matrix by a scalar.
 *
 * @param A Matrix to be multiplied.
 * @param n Scalar value.
 * @return Resulting matrix after scalar multiplication.
 */
Matrix *scalar_p_M(Matrix *A, double n) {
    Matrix *Res = newMatrix(A->rows, A->columns);

    for (unsigned int i = 0; i < Res->rows; i++) {
        for (unsigned int j = 0; j < Res->columns; j++) {
            M_index(Res, i, j) = M_index(A, i, j) * n;
        }
    }

    return Res;

}


/**
 * @brief Transpose a matrix.
 *
 * This function computes the transpose of a matrix.
 *
 * @param A Matrix to be transposed.
 * @return Transposed matrix.
 */
Matrix *transpose_M(Matrix *A) {
    Matrix *Res = newMatrix(A->columns, A->rows);

    for (unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            M_index(Res, j, i) = M_index(A, i, j);
        }
    }

    return Res;

}


/**
 * @brief Add two matrices element-wise.
 *
 * This function adds each element of matrix A to the corresponding element of matrix B.
 *
 * @param Res Destination matrix for the sum.
 * @param A Matrix A.
 */
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



/**
 * @brief Print the elements of a matrix to the console.
 *
 * This function prints the elements of a matrix to the console.
 *
 * @param A Matrix to be printed.
 */
void print_M(Matrix *A) {
    for (unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            printf("%lf", M_index(A, i, j));
            if (j != A->columns - 1) {
                printf(", ");
            }
        }
        putchar('\n');
    }
}


/**
 * @brief Fill a matrix from a one-dimensional array.
 *
 * This function fills a matrix with data from an array.
 *
 * @param A Matrix to be filled.
 * @param arr Array containing data.
 * @param len Length of the array.
 */
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


/**
 * @brief Generate random values in a matrix within a specified range.
 *
 * This function fills a matrix with random values within the specified range.
 *
 * @param A Matrix to be filled with random values.
 * @param min Minimum value for the random range.
 * @param max Maximum value for the random range.
 */
void rand_M(Matrix *A, double min, double max) {
    for (unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            M_index(A, i, j) = randf(min, max);
        }
    }
}


/**
 * @brief Print an ASCII representation of a matrix to the console.
 *
 * This function prints an ASCII representation of a matrix to the console.
 * Values above a certain threshold are represented by '#' and '.' characters.
 *
 * @param A Matrix to be printed.
 */
void ascii_print_M(Matrix *A) {
    double n;
    for(unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            n = M_index(A, i, j);
            if (n > 125) {
                putchar('#');
            } else if (n > 50) {
                putchar('.');
            } else {
                putchar(' ');
            }
        }
        putchar('\n');
    }
}


/**
 * @brief Create a new multilayer perceptron (MLP).
 *
 * This function initializes a new MLP with specified architecture and activation functions.
 *
 * @param depth Number of layers in the MLP.
 * @param input_size Number of neurons in the input layer.
 * @param hidden_layer_size Number of neurons in each hidden layer.
 * @param output_size Number of neurons in the output layer.
 * @param activate Array of activation functions for each layer.
 * @return A pointer to the newly created MLP.
 */
MLP *newMLP(int depth, int input_size, int hidden_layer_size, int output_size, activation_f* activate) {
    MLP *Res = (MLP *)calloc(1, sizeof(MLP));
    check_malloc(Res);
    Res->depth = depth;
    Res->weights = (Matrix *)calloc(depth, sizeof(Matrix));
    Res->biases = (Matrix *)malloc(sizeof(Matrix) * depth);;
    Res->activate = activate;



    // matrices of the first hidden layer
    // bias is a transposed vector
    newMatrixAt(&(Res->weights[0]), (unsigned int)input_size, (unsigned int)hidden_layer_size);
    newMatrixAt(&(Res->biases[0]), 1, hidden_layer_size);

    // the rest of the hidden layers
    for (int i = 1; i < depth - 1; i++) {
        newMatrixAt(&(Res->weights[i]), (unsigned int)hidden_layer_size, (unsigned int)hidden_layer_size);
        newMatrixAt(&(Res->biases[i]), 1, hidden_layer_size);
    }

    // output layer
    newMatrixAt(&(Res->weights[depth - 1]), (unsigned int)hidden_layer_size, (unsigned int)output_size);
    newMatrixAt(&(Res->biases[depth - 1]), 1, output_size);

    return Res;
}


void freeMLP(MLP *dest) {
    if (dest == NULL) {
        return;
    }

    for (int i = 0; i < dest->depth; i++) {
        free(dest->weights[i].data);
        free(dest->biases[i].data);
    }
    free(dest->weights);
    free(dest->biases);
    free(dest);
}


/**
 * @brief Add a row vector to a matrix.
 *
 * This function adds a row vector to each row in the destination matrix.
 *
 * @param dest Destination matrix.
 * @param row_V Row vector to be added.
 */
void add_row_V_M(Matrix *dest, Matrix *row_V) {
    if (dest->columns != row_V->columns) {
        fprintf(stderr, "Not matching column size!\n");
        exit(EXIT_FAILURE);
    }
    if (row_V->rows > 1) {
        fprintf(stderr, "Row vector contains more than one row!\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < dest->rows; i++) {
        for (unsigned int j = 0; j < dest->columns; j++) {
            M_index(dest, i, j) += M_index(row_V, 0, j);
        }
    }
}


/**
 * @brief Perform feedforward operation on the MLP.
 *
 * This function computes the output of the MLP for a given input using the feedforward algorithm.
 *
 * @param network Pointer to the MLP.
 * @param input Input matrix.
 * @return Output matrix of the MLP.
 */
Matrix* feedForward(MLP* network, Matrix* input) {
    Matrix *z;

    Matrix *prev_z = product_M(input, &(network->weights[0]));
    add_row_V_M(prev_z, &(network->biases[0]));
    network->activate[0](prev_z);

    for (int i = 1; i < network->depth; i++) {
        z = product_M(prev_z, &(network->weights[i]));
        add_row_V_M(z, &(network->biases[i]));
        network->activate[i](z);
        

        freeMatrix(prev_z);
        prev_z = z;
    }

    return z;
}


/**
 * @brief Rectified Linear Unit (ReLU) activation function.
 *
 * @param x Input value.
 * @return Output value after applying ReLU.
 */
double ReLu(double x) {
    if (x < 0) {
        return 0;
    }
    else {
        return x;
    }
}


/**
 * @brief Apply ReLU activation to a matrix.
 *
 * This function applies the ReLU activation function to each element of the matrix.
 *
 * @param dest Matrix to be modified.
 */
void ReLu_M(Matrix* dest) {
    if (dest == NULL) {
        fprintf(stderr, "NULL matrix pointer!\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < dest->rows; i++) {
        for (unsigned int j = 0; j < dest->columns; j++) {
            M_index(dest, i, j) = ReLu(M_index(dest, i, j));
        }
    }
}


/**
 * @brief Sigmoid activation function.
 *
 * @param x Input value.
 * @return Output value after applying sigmoid.
 */
double sigmoid(double x) {
    return 1 / (double)(1 + exp(-x));
}


/**
 * @brief Apply sigmoid activation to a matrix.
 *
 * This function applies the sigmoid activation function to each element of the matrix.
 *
 * @param dest Matrix to be modified.
 */
void sigmoid_M(Matrix *dest) {
    if (dest == NULL) {
        fprintf(stderr, "NULL matrix pointer!\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < dest->rows; i++) {
        for (unsigned int j = 0; j < dest->columns; j++) {
            M_index(dest, i, j) = sigmoid(M_index(dest, i, j));
        }
    }
}


/**
 * @brief Apply softmax activation to a matrix.
 *
 * This function applies the softmax activation function to each row of the matrix.
 *
 * @param dest Matrix to be modified.
 */
void soft_max_M(Matrix *dest) {
    if (dest == NULL) {
        fprintf(stderr, "NULL matrix pointer!\n");
        exit(EXIT_FAILURE);
    }
    double sum = 0.0;

    for (unsigned int i = 0; i < dest->rows; i++) {
        for (unsigned int j = 0; j < dest->columns; j++) {
            M_index(dest, i, j) = exp(M_index(dest, i, j));
            sum += M_index(dest, i, j);
        }
        for (unsigned int j = 0; j < dest->columns; j++) {
            M_index(dest, i, j) = M_index(dest, i, j) / sum;
        }
        sum = 0.0;
    }
}


/**
 * @brief Mean Squared Error (MSE) cost function for a batch of outputs.
 *
 * This function calculates the mean of mean square error of a batch of outputs compared to the desired output.
 *
 * @param output Output matrix from the MLP.
 * @param y Desired output matrix.
 * @return Mean Squared Error (MSE) value.
 */
double MSE(Matrix* output, Matrix* y) {
    if (output->rows != y->rows || output->columns != y->columns) {
        fprintf(stderr, "Not matching matrix dimensions in cost!\n");
        exit(EXIT_FAILURE);
    }
    
    double sum = 0.0;
    double sum_per_row = 0.0;
    double d = 0.0;
    for (unsigned int i = 0; i < y->rows; i++) {
        for (unsigned int j = 0; j < y->columns; j++) {
            d = M_index(y, i, j) - M_index(output, i, j);
            sum_per_row += d * d;
        }
        sum += sum_per_row / (double)y->columns;     // MSE of the row
        sum_per_row = 0.0;
    }

    return sum / (double)y->rows;
}