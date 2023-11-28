#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


/**
 * @brief Generates a random double precision floating-point number within a specified range.
 *
 * @param min Minimum value of the range.
 * @param max Maximum value of the range.
 * @return Random double within the specified range.
 */
double randf(double min, double max) {
    double scale = rand() / (double) RAND_MAX;
    return min + scale * (max - min);
}


/**
 * @brief Generates a random integer within a specified range.
 *
 * @param min Minimum value of the range.
 * @param max Maximum value of the range.
 * @return Random integer within the specified range.
 */
int randint(int min, int max) {
    return min + ( rand() % (max-min+1) );
}


/**
 * @brief Swaps the content of two memory locations.
 *
 * @param a Pointer to the first memory location.
 * @param b Pointer to the second memory location.
 * @param size Size of each element in bytes.
 */
void swap(void *a, void *b, size_t size) {
    char *tmp = malloc(size);
    memcpy(tmp, a, size);
    memcpy(a, b, size);
    memcpy(b, tmp, size);

    free(tmp);
}


/**
 * @brief Shuffles an array randomly.
 *
 * @param arr Pointer to the array to be shuffled.
 * @param size Size of each element in bytes.
 * @param len Number of elements in the array.
 */
void shuffle(void *arr, size_t size, size_t len) {
    int n;
    for (size_t i = 0; i < len; i++) {
        n = randint(0, len-1);

        // arr[i] <-> arr[n]
        swap((char*)arr + (size * i), (char*)arr + (size * n), size);
    }
}


/**
 * @brief Checks if memory allocation was successful.
 *
 * @param ptr Pointer to the allocated memory.
 */
void check_malloc(void *ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Failed allocating memory!\n");
        exit(EXIT_FAILURE);
    }
}


/**
 * @brief Converts an array of uint8_t to an array of double.
 *
 * @param dest Pointer to the destination array of double.
 * @param source Pointer to the source array of uint8_t.
 * @param len Number of elements in the arrays.
 */
void uint8_to_double(double *dest, uint8_t *source, size_t len){
    for (size_t i = 0; i < len; i++) {
        dest[i] = (double) source[i];
    }
}