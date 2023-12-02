#include "neural_net.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


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


void freeMatrix(Matrix *dest) {
    if (dest == NULL) {
        return;
    }

    free(dest->data);
    free(dest);
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
            for (unsigned int k = 0; k < B->rows; k++) {
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
            printf("%lf", M_index(A, i, j));
            if (j != A->columns - 1) {
                printf(", ");
            }
        }
        putchar('\n');
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

void rand_M(Matrix *A, double min, double max) {
    for (unsigned int i = 0; i < A->rows; i++) {
        for (unsigned int j = 0; j < A->columns; j++) {
            M_index(A, i, j) = randf(min, max);
        }
    }
}


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


void copy_values_M(Matrix *dest, Matrix *source) {
    if (dest->rows != source->rows && dest->columns != source->columns) {
        fprintf(stderr, "Not matching dimensions!\n");
        exit(EXIT_FAILURE);
    }

    copy_double_arr(dest->data, source->data, dest->rows * dest->columns);
}


MLP *newMLP(int depth, int input_size, int hidden_layer_size, int output_size, activation_f* activate) {
    MLP *Res = (MLP *)malloc(sizeof(MLP));
    check_malloc(Res);
    Res->depth = depth;
    Res->weights = (Matrix *)malloc(sizeof(Matrix) * depth);
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


MLP_data *newMLP_data(MLP *neural_net, int batch_size) {
    MLP_data *Res = (MLP_data*)malloc(sizeof(MLP_data));
    Res->depth = neural_net->depth;
    Res->origin = neural_net;
    Res->pre_activated_values = (Matrix*)malloc(sizeof(Matrix) * neural_net->depth);
    Res->activated_values = (Matrix*)malloc(sizeof(Matrix) * neural_net->depth);

    for (int i = 0; i < neural_net->depth; i++) {
        newMatrixAt(&(Res->pre_activated_values[i]), batch_size, neural_net->weights[i].columns);
        newMatrixAt(&(Res->activated_values[i]), batch_size, neural_net->weights[i].columns);
    }

    return Res;
}


void freeMLP_data(MLP_data *dest) {
    if (dest == NULL) {
        return;
    }

    for (int i = 0; i < dest->depth; i++) {
        free(dest->pre_activated_values[i].data);
        free(dest->activated_values[i].data);
    }
    free(dest->pre_activated_values);
    free(dest->activated_values);
    free(dest);

} 


// adds row vector for each row in dest
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


Matrix* feedForward(MLP* network, Matrix* input, MLP_data *neuron_values) {
    if (network != neuron_values->origin) {
        fprintf(stderr, "The MLP_data does not belong to the given network!\n");
        exit(EXIT_FAILURE);
    }
    Matrix *z;

    Matrix *prev_z = product_M(input, &(network->weights[0]));
    add_row_V_M(prev_z, &(network->biases[0]));

    copy_values_M(&(neuron_values->pre_activated_values[0]), prev_z);

    network->activate[0](prev_z);

    copy_values_M(&(neuron_values->activated_values[0]), prev_z);

    for (int i = 1; i < network->depth; i++) {
        z = product_M(prev_z, &(network->weights[i]));
        add_row_V_M(z, &(network->biases[i]));


        copy_values_M(&(neuron_values->pre_activated_values[i]), z);

        network->activate[i](z);

        copy_values_M(&(neuron_values->activated_values[i]), z);


        freeMatrix(prev_z);
        prev_z = z;
    }

    return z;
}


double ReLu(double x) {
    if (x < 0) {
        return 0;
    }
    else {
        return x;
    }
}


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


double sigmoid(double x) {
    return 1 / (double)(1 + exp(-x));
}


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


// Mean of mean square error of a batch of outputs
// y: desired output matrix
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


// Mean of cross entropy error of a batch of outputs
// y: desired output matrix
double cross_entropy(Matrix *output, Matrix *y) {
    if (output->rows != y->rows || output->columns != y->columns) {
        fprintf(stderr, "Not matching matrix dimensions in cost!\n");
        exit(EXIT_FAILURE);
    }
    double sum = 0.0;
    double sum_per_row = 0.0;
    double predicted = 0.0;
    for (unsigned int i = 0; i < y->rows; i++) {
        for (unsigned int j = 0; j < y->columns; j++) {
            // because of the log, the predicted value can't be zero, in that case it uses a small value
            predicted = MAX(1e-13, M_index(output, i, j));  // since predicted values are probabilities
            sum_per_row += M_index(y, i, j) * log(predicted);
        }
        sum_per_row *= -1.0;
        sum += sum_per_row;   // cross entropy of the row
        sum_per_row = 0.0;
    }

    return sum / (double)y->rows;
}