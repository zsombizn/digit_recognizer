#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

double randf(double min, double max) {
    double scale = rand() / (double) RAND_MAX;
    return min + scale * (max - min);
}


int randint(int min, int max) {
    return min + ( rand() % (max-min+1) );
}


void swap(void *a, void *b, size_t size) {
    char *tmp = malloc(size);
    memcpy(tmp, a, size);
    memcpy(a, b, size);
    memcpy(b, tmp, size);

    free(tmp);
}


void shuffle(void *arr, size_t size, size_t len) {
    int n;
    for (size_t i = 0; i < len; i++) {
        n = randint(0, len-1);

        // arr[i] <-> arr[n]
        swap((char*)arr + (size * i), (char*)arr + (size * n), size);
    }
}


void check_malloc(void *ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Failed allocating memory!\n");
        exit(EXIT_FAILURE);
    }
}