#ifndef IO_H
#define IO_H
#include "neural_net.h"
#include <stdint.h>

void msb_to_lsb(void *target, size_t size);
void read_MNIST_data(const char *images_fname, const char *labels_fname, size_t *len, Matrix **images, uint8_t **labels);


#endif