#include "io.h"
#include "neural_net.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


void msb_to_lsb(void *target, size_t size) {
    for (size_t i = 0; i < size / 2; i++) {
        swap((char *)target + i, (char*)target + (size - i - 1), 1);
    }
}


void read_MNIST_data(const char *images_fname, const char *labels_fname, Example **images, size_t *len) {
    *images = NULL;
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
    *images = malloc(sizeof(Example) * images_len);
    check_malloc(images);
    
    for (size_t i = 0; i < *len; i++) {

        (*images)[i].data_array = malloc(sizeof(uint8_t) * rows * columns);
        check_malloc((*images)[i].data_array);

        for (size_t k = 0; k < rows * columns; k++) {
            fread(&data, 1, 1, raw_images);
            (*images)[i].data_array[k] = data;
            
        }
        if (i % 1000 == 0){
            putchar('#');
            fflush(stdout);
        }
    }
    printf("\n");

    // reading labels

    for (size_t i = 0; i < *len; i++) {
        fread(&data, 1, 1, raw_labels);
        (*images)[i].label = data;
    }

    fclose(raw_images);
    fclose(raw_labels);

}

// the size of the rows should include the paddig 0-s
void write_Matrix_BMP(const char *fname, Matrix *M) {
    BMP_HEADER header = {
        .signature = 0x4d42,
        .fileSize = sizeof(BMP_HEADER) + sizeof(BITPMAPINFOHEADER) + ( ((M->columns * 3 + 3)/4)*4 ) * M->rows,
        .reserved1 = 0,
        .reserved2 = 0,
        .offset = sizeof(BMP_HEADER) + sizeof(BITPMAPINFOHEADER)
    };

    BITPMAPINFOHEADER dib = {
        .header_size = sizeof(BITPMAPINFOHEADER),
        .width = M->columns,
        .height = M->rows,
        .color_planes = 1,
        .depth = 24,
        .compression = 0,
        .image_size = 0,
        .horizontal_res = 50190,
        .vertical_res = 50190,
        .color_palette = 0,
        .important_colors = 0
    };

    FILE *image = fopen(fname, "wb");
    if (image == NULL) {
        fprintf(stderr, "Failed creating file!\n");
        exit(EXIT_FAILURE);
    }

    fwrite(&header, sizeof(BMP_HEADER), 1, image);
    fwrite(&dib, sizeof(BITPMAPINFOHEADER), 1, image);

    // writing pixel data https://en.wikipedia.org/wiki/BMP_file_format#Pixel_storage

    uint8_t pixel_value;
    int i, j;
    int row_width;  // in bytes
    for (i = M->rows-1; i >= 0; i--) {
        for (j = 0; j < (int) M->columns; j++) {
            pixel_value = (uint8_t) M_index(M, i, j);
            fwrite(&pixel_value, 1, 1, image);
            fwrite(&pixel_value, 1, 1, image);
            fwrite(&pixel_value, 1, 1, image);
        }
        // padding of rows

        row_width = j * 3;

        while (row_width % 4 != 0) {
            fputc(0, image);
            row_width++;
        }
    }

    fclose(image);

    
}


void check_mkdir(char *path) {
    #ifdef _WIN32
        struct _stat info;
        if (_stat(path, &info) == 0 && (info.st_mode & _S_IFDIR) != 0 ) { 
    #else
        struct stat info;
        if (stat(path, &info) == 0 && S_ISDIR(info.st_mode)) {
    #endif
    }
    else if (mkdir(path, 0755) != 0) {
        fprintf(stderr, "Failed creating directory!\n");
        exit(EXIT_FAILURE);
    }

}


int encode_activation(activation_f f) {
    if (f == &ReLu_M) {
        return ReLu_CODE;
    } else if(f == &sigmoid_M) {
        return sigmoid_CODE;
    } else if(f == &soft_max_M) {
        return softmax_CODE;
    } else {
        return -1;
    }
}


activation_f decode_activation(int f) {
    switch (f) {
    case ReLu_CODE:
        return &ReLu_M;

    case sigmoid_CODE:
        return &sigmoid_M;

    case softmax_CODE:
        return &soft_max_M;

    default:
        return NULL;
        break;
    }
}


// MLP *newMLP(int depth, int input_size, int hidden_layer_size, int output_size, activation_f* activate)
void write_MLP(char *fname, MLP *model) {
    FILE *fp = fopen(fname, "wb");
    int encoded_activation;

    // depth
    fwrite(&(model->depth), sizeof(int), 1, fp);

    // input size
    fwrite(&(model->weights[0].rows), sizeof(unsigned int), 1, fp);

    // hidden_layer size
    fwrite(&(model->weights[0].columns), sizeof(unsigned int), 1, fp);

    // output size
    fwrite(&(model->weights[model->depth-1].columns), sizeof(unsigned int), 1, fp);

    // activation functions
    for (int i = 0; i < model->depth; i++) {
        encoded_activation = encode_activation(model->activate[i]);
        fwrite(&encoded_activation, sizeof(int), 1, fp);
    }

    // weights
    for (int i = 0; i < model->depth; i++) {
        fwrite(model->weights[i].data, sizeof(double), model->weights[i].rows * model->weights[i].columns, fp);
    }

    // biases
    for (int i = 0; i < model->depth; i++) {
        fwrite(model->biases[i].data, sizeof(double), model->biases[i].rows * model->biases[i].columns, fp);
    }

    fclose(fp);
}


MLP *read_MLP(char *fname) {
    FILE *fp = fopen(fname, "rb");
    int encoded_activation;
    int depth, input_size, hidden_layer_size, output_size;
    activation_f* activate;

    MLP *Res;

    // depth
    fread(&depth, sizeof(int), 1, fp);

    activate = malloc(sizeof(activation_f) * depth);
    check_malloc(activate);

    // input size
    fread(&input_size, sizeof(int), 1, fp);

    // hidden_layer size
    fread(&hidden_layer_size, sizeof(int), 1, fp);

    // output size
    fread(&output_size, sizeof(int), 1, fp);

    // activation functions
    for (int i = 0; i < depth; i++) {
        fread(&encoded_activation, sizeof(int), 1, fp);
        activate[i] = decode_activation(encoded_activation);
    }

    Res = newMLP(depth, input_size, hidden_layer_size, output_size, activate);

    // weights
    for (int i = 0; i < depth; i++) {
        fread(Res->weights[i].data, sizeof(double), Res->weights[i].rows * Res->weights[i].columns, fp);
    }

    // biases
    for (int i = 0; i < depth; i++) {
        fread(Res->biases[i].data, sizeof(double), Res->biases[i].rows * Res->biases[i].columns, fp);
    }

    fclose(fp);

    return Res;
}