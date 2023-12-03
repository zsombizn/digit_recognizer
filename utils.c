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


void uint8_to_double(double *dest, uint8_t *source, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dest[i] = (double) source[i];
    }
}


void copy_double_arr(double *dest, double *source, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dest[i] = source[i];
    }
}


int str_to_int(const char* str) {
    char* endptr;
    long l_value = strtol(str, &endptr, 10);

    if (endptr == str) {
        fprintf(stderr, "Clear input string!\n");
        exit(EXIT_FAILURE);
    }

    if (*endptr != '\0') {
        printf("Not a valid integer!\n");
    }

    return (int)l_value;
}


double str_to_double(const char* str) {
    char* endptr;
    double res = strtod(str, &endptr);

    if (endptr == str) {
        fprintf(stderr, "Clear input string!\n");
        exit(EXIT_FAILURE);
    }

    if (*endptr != '\0') {
        printf("Not a valid double!\n");
    }

    return res;
}


double *one_hot(double *dest, uint8_t n, int len) {
    if (n > len) {
        fprintf(stderr, "It is not possible to one-hot encode!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < len; i++) {
        if (i == n) {
            dest[i] = 1.0;
        }
        else {
            dest[i] = 0.0;
        }
    }
}