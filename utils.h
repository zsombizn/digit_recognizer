#ifndef UTILS_H
#define UTILS_H
#include <stddef.h>
#include <stdint.h>

double randf(double min, double max);
int randint(int min, int max);
void swap(void *a, void *b, size_t size);
void shuffle(void *arr, size_t size, size_t len);
void check_malloc(void *ptr);
void uint8_to_double(double *dest, uint8_t *source, size_t len);
void copy_double_arr(double *dest, double *source, size_t len);
int str_to_int(const char* str);
double str_to_double(const char* str);
double* one_hot(double* dest, uint8_t n, int len);

#endif