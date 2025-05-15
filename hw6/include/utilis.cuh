#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <sys/time.h>
#include <cuda_runtime.h>

// Define a small epsilon for floating point comparisons
#define EPISILON 0.0001

/**
 * @brief Compares two floating-point numbers considering epsilon tolerance.
 * 
 * @param x First floating-point number.
 * @param y Second floating-point number.
 * @return true if the numbers are equal or their difference is less than EPISILON.
 * @return false otherwise.
 */
bool is_data_eq(float x, float y);

/**
 * @brief Prints the corners of a matrix in a readable way.
 * 
 * @param data Pointer to the matrix data.
 * @param m Rows of the matrix.
 * @param n Columns of the matrix.
 */
void printf_matrix_corner(float *data, int m, int n);

/**
 * @brief Randomizes the values in a matrix of size m x n.
 * 
 * @param mat Pointer to the matrix data.
 * @param m Rows of the matrix.
 * @param n Columns of the matrix.
 */
void randomize_matrix(float *mat, int m, int n);

/**
 * @brief A simpler version of randomize_matrix that fills the matrix with a repeating pattern.
 * 
 * @param mat Pointer to the matrix data.
 * @param m Rows of the matrix.
 * @param n Columns of the matrix.
 */
void randomize_matrix_simply(float *mat, int m, int n);

/**
 * @brief Prints the current GPU device information.
 */
void get_device_info();

/**
 * @brief Prints a matrix with a reminder string.
 * 
 * @param s Reminder string describing the matrix.
 * @param data Pointer to the matrix data.
 * @param m Rows of the matrix.
 * @param n Columns of the matrix.
 */
void print_all_matrix(std::string s, float *data, int m, int n);

#endif // UTILS_H
