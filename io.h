#ifndef IO_H
#define IO_H
#include "neural_net.h"
#include <stdint.h>

#ifdef _WIN32
#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>
#define mkdir(dir, mode) _mkdir(dir)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define mkdir(dir, mode) mkdir(dir, mode)
#endif

#pragma pack(push, 1)

// Based on:
// https://en.wikipedia.org/wiki/BMP_file_format


/**
 * @brief Structure representing the header of a BMP file.
 */
typedef struct {
    uint16_t signature;  /**< Header field used to identify the BMP and DIB file (0x42 0x4D in hexadecimal). */
    uint32_t fileSize;   /**< Size of the BMP file in bytes. */
    uint16_t reserved1;  /**< Reserved field (not used). */
    uint16_t reserved2;  /**< Reserved field (not used). */
    uint32_t offset;     /**< Offset, i.e., starting address, of the byte where the bitmap image data (pixel array) can be found. */
} BMP_HEADER;

/**
 * @brief Structure representing the header information of a BMP file.
 */
typedef struct {
    uint32_t header_size;        /**< Size of this header, in bytes (40). */
    int32_t width;               /**< Bitmap width in pixels (signed integer). */
    int32_t height;              /**< Bitmap height in pixels (signed integer). */
    uint16_t color_planes;       /**< Number of color planes (must be 1). */
    uint16_t depth;              /**< Number of bits per pixel, which is the color depth of the image. */
    uint32_t compression;        /**< Compression method being used. */
    uint32_t image_size;         /**< Image size. This is the size of the raw bitmap data. */
    int32_t horizontal_res;      /**< Horizontal resolution of the image (pixels per meter, signed integer). */
    int32_t vertical_res;        /**< Vertical resolution of the image (pixels per meter, signed integer). */
    uint32_t color_palette;      /**< Number of colors in the color palette, or 0 to default to 2^n. */
    uint32_t important_colors;   /**< Number of important colors used, or 0 when every color is important (generally ignored). */
} BITPMAPINFOHEADER;

#pragma pack(pop)

void msb_to_lsb(void *target, size_t size);
void read_MNIST_data(const char *images_fname, const char *labels_fname, Example **images, size_t *len);
void write_Matrix_BMP(const char *fname, Matrix *M);
void check_mkdir(char *path);
void write_MLP(char *fname, MLP *model);
MLP *read_MLP(char *fname);


#endif