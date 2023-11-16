#include "io.h"
#include "neural_net.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


void msb_to_lsb(void *target, size_t size) {
    for (size_t i = 0; i < size / 2; i++) {
        swap(target + i, target + (size - i - 1), 1);
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
    
    for (size_t i = 0; i < *len; i++) {

        (*images)[i].vector_M = newMatrix(rows * columns, 1);

        for (size_t k = 0; k < rows * columns; k++) {
            fread(&data, 1, 1, raw_images);
            M_index((*images)[i].vector_M, k, 1) = data;
            
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

    int written_bytes = 0;
    FILE *image = fopen(fname, "wb");

    fwrite(&header, sizeof(BMP_HEADER), 1, image);
    fwrite(&dib, sizeof(BITPMAPINFOHEADER), 1, image);

    written_bytes += sizeof(BMP_HEADER) + sizeof(BITPMAPINFOHEADER);

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
            written_bytes += 3;
        }
        // padding of rows

        row_width = j * 3;

        while (row_width % 4 != 0) {
            fputc(0, image);
            written_bytes ++;
            row_width++;
        }
    }

    printf("%d written out. Header size: %d\n", written_bytes, sizeof(BMP_HEADER) + sizeof(BITPMAPINFOHEADER));

    fclose(image);

    
}