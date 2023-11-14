#include "io.h"
#include "neural_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


void msb_to_lsb(void *target, size_t size) {
    for (size_t i = 0; i < size / 2; i++) {
        swap(target + i, target + (size - i - 1), 1);
    }
}


void read_MNIST_data(const char *images_fname, const char *labels_fname, size_t *len, Matrix **images, uint8_t **labels) {
    *images = NULL;
    *labels = NULL;
    FILE *raw_images = fopen(images_fname, "rb");
    FILE *raw_labels = fopen(labels_fname, "rb");

    if (raw_images == NULL || raw_labels == NULL) {
        fprintf(stderr, "Failed opening file!\n");
        exit(EXIT_FAILURE);
    }

    // reading headers

    uint32_t images_magic, labels_magic;
    fread(&images_magic, 4, 1, raw_images);
    fread(&labels_magic, 4, 1, raw_labels);

    msb_to_lsb(&images_magic, 4);
    msb_to_lsb(&labels_magic, 4);

    printf("Magic numbers are: %d, %d\n", images_magic, labels_magic);

    uint32_t images_len, labels_len;
    fread(&images_len, 4, 1, raw_images);
    fread(&labels_len, 4, 1, raw_labels);

    msb_to_lsb(&images_len, 4);
    msb_to_lsb(&labels_len, 4);

    if (labels_len != images_len) {
        fprintf(stderr, "Number of labels and images are not equal!\n");
        exit(EXIT_FAILURE);
    }

    *len = (size_t) images_len;

    uint32_t rows, columns;
    fread(&rows, 4, 1, raw_images);
    fread(&columns, 4, 1, raw_images);
    msb_to_lsb(&rows, 4);
    msb_to_lsb(&columns, 4);


    // reading images

    uint8_t data;
    double *arr = malloc(sizeof(double) * rows * columns);
    *images = malloc(sizeof(Matrix) * images_len);
    Matrix *T_M;

    
    for (size_t i = 0; i < *len; i++) {
        T_M = newMatrix(rows * columns, 1);
        memcpy(&(*images)[i], T_M, sizeof(Matrix));
        free(T_M);

        for (size_t k = 0; k < rows * columns; k++) {
            fread(&data, 1, 1, raw_images);
            arr[k] = data;
            
        }
        fill_from_array_M(&(*images)[i], arr, rows * columns);
        if (i % 1000 == 0){
            putchar('#');
            fflush(stdout);
        }
    }
    printf("\n");

    // reading labels

    *labels = malloc(sizeof(uint8_t) * labels_len);

    for (size_t i = 0; i < *len; i++) {
        fread(&data, 1, 1, raw_labels);
        (*labels)[i] = data;
    }



    fclose(raw_images);
    fclose(raw_labels);

}