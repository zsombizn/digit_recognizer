#ifndef IO_H
#define IO_H
#include "neural_net.h"
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <io.h>
#define mkdir(dir, mode) _mkdir(dir)
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define mkdir(dir, mode) mkdir(dir, mode)
#endif

#pragma pack(push, 1)

// Based on:
// https://en.wikipedia.org/wiki/BMP_file_format

typedef struct {
    uint16_t signature;          // header field used to identify the BMP and DIB file is 0x42 0x4D in hexadecimal, same as BM in ASCII.
    uint32_t fileSize;           // size of the BMP file in bytes.
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;             // offset, i.e. starting address, of the byte where the bitmap image data (pixel array) can be found.
} BMP_HEADER;


typedef struct {
    uint32_t header_size;        // size of this header, in bytes (40)
    int32_t width;               // bitmap width in pixels (signed integer)
    int32_t height;              // bitmap height in pixels (signed integer) 
    uint16_t color_planes;       // number of color planes (must be 1)
    uint16_t depth;              // number of bits per pixel, which is the color depth of the image.
    uint32_t compression;        // compression method being used.
    uint32_t image_size;         // image size. This is the size of the raw bitmap data
    int32_t horizontal_res;      // horizontal resolution of the image. (pixel per metre, signed integer)
    int32_t vertical_res;        // vertical resolution of the image. (pixel per metre, signed integer)
    uint32_t color_palette;      // number of colors in the color palette, or 0 to default to 2^n
    uint32_t important_colors;   // number of important colors used, or 0 when every color is important; generally ignored 

} BITPMAPINFOHEADER;

#pragma pack(pop)

void msb_to_lsb(void *target, size_t size);
void read_MNIST_data(const char *images_fname, const char *labels_fname, Example **images, size_t *len);
void write_Matrix_BMP(const char *fname, Matrix *M);
Matrix *read_Matrix_BMP(const char *fname);
void write_Matrix_txt(FILE *fp, Matrix *A);
void write_model_txt(const char *fname, MLP *net);
void write_neruons_txt(const char *fname, MLP_data *neuron_vals);
void check_mkdir(const char *path);
void check_file(const char *fname);
void write_MLP(char *fname, MLP *model);
MLP *read_MLP(const char *fname);


#endif