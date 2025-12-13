#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif

#ifndef NN_FREE
#define NN_FREE free
#endif

#ifndef NN_ASSERT
#define NN_ASSERT assert
#endif

#define NN_ARRAY_LEN(xs) (sizeof((xs)) / sizeof((xs)[0]))

typedef enum {
    ACTIVATION_SIG,
    ACTIVATION_RELU,
    ACTIVATION_TANH,
    ACTIVATION_SIN,
} ActivationType;

typedef struct {
    float (*activation_func)(float x);
    float (*derivative_func)(float y);
} NN_Activation;

void            nn_srand(uint64_t seed);
float           nn_random_float(void);

float           nn_activate(float x, ActivationType act);
float           nn_activate_deriv(float y, ActivationType act);

/**
 * @brief A simple linear memory allocator (Arena).
 * Used to group allocations for matrices and networks so they can be freed all at once.
**/
typedef struct {
    size_t capacity;
    size_t size;
    uintptr_t *words; // The memory block
} NN_Region;

NN_Region       nn_region_alloc(size_t capacityInBytes);
void           *nn_region_alloc_ptr(NN_Region *r, size_t size_bytes);
void            nn_region_free(NN_Region *r);

/** @brief Resets the arena pointer to the beginning (does not free memory). **/
#define nn_region_reset(r)               ((r) != NULL ? (r)->size = 0 : (void)0)
#define nn_region_occupied_bytes(r)      ((r) != NULL ? (r)->size * sizeof(*(r)->words) : 0)


typedef struct {
    size_t rows;
    size_t cols;
    float *elements;
} NN_Matrix;

typedef struct {
    size_t cols;
    float *elements;
} NN_Row;

// Access Macros
#define NN_ROW_AT(row, col)              (row).elements[col]
#define NN_MATRIX_AT(m, i, j)            (m).elements[(i) * (m).cols + (j)]

// Matrix & Row Operations
NN_Matrix       nn_matrix_alloc(NN_Region *r, size_t rows, size_t cols);
void            nn_matrix_fill(NN_Matrix m, float x);
void            nn_matrix_rand(NN_Matrix m, float low, float high);
NN_Row          nn_matrix_row(NN_Matrix m, size_t row);
void            nn_matrix_copy(NN_Matrix dst, NN_Matrix src);
void            nn_matrix_dot(NN_Matrix dst, NN_Matrix a, NN_Matrix b);
void            nn_matrix_sum(NN_Matrix dst, NN_Matrix a);
void            nn_matrix_activate(NN_Matrix m, ActivationType act);
void            nn_matrix_print(NN_Matrix m, const char *name, size_t padding);

// Row Helpers (Inline View Conversions)
static inline NN_Matrix nn_row_to_matrix(NN_Row row) {
    return (NN_Matrix){1, row.cols, row.elements};
}

static inline NN_Row nn_row_slice(NN_Row row, size_t i, size_t cols) {
    if (i >= row.cols || i + cols > row.cols) return (NN_Row){0, NULL};
    return (NN_Row){cols, &NN_ROW_AT(row, i)};
}

// Row Macros wrapping Matrix functions
#define nn_row_alloc(r, cols)            nn_matrix_row(nn_matrix_alloc(r, 1, cols), 0)
#define nn_row_rand(row, low, high)      nn_matrix_rand(nn_row_to_matrix(row), low, high)
#define nn_row_fill(row, x)              nn_matrix_fill(nn_row_to_matrix(row), x)
#define nn_row_print(row, name, padding) nn_matrix_print(nn_row_to_matrix(row), name, padding)
#define nn_row_copy(dst, src)            nn_matrix_copy(nn_row_to_matrix(dst), nn_row_to_matrix(src))


typedef struct {
    size_t *arch;
    size_t archCount;
    NN_Matrix *weights;
    NN_Row *biases;
    NN_Row *activations;
    ActivationType actType;
} NeuralNetwork;

#define NN_NETWORK_INPUT(nn)            ((nn).archCount > 0 ? (nn).activations[0] : (NN_Row){0})
#define NN_NETWORK_OUTPUT(nn)           ((nn).archCount > 0 ? (nn).activations[(nn).archCount - 1] : (NN_Row){0})

NeuralNetwork   nn_network_alloc(NN_Region *r, size_t *arch, size_t archCount, ActivationType act);
void            nn_network_zero(NeuralNetwork nn);
void            nn_network_rand(NeuralNetwork nn, float low, float high);
void            nn_network_forward(NeuralNetwork nn);
float           nn_network_cost(NeuralNetwork nn, NN_Matrix trainingData);

// Training
// Note: 'gradient' (gradient network) must be allocated by the user before calling backprop.
// This allows memory reuse across training epochs.
void            nn_network_backprop(NeuralNetwork nn, NeuralNetwork gradient, NN_Matrix trainingData);
void            nn_network_finite_diff(NeuralNetwork nn, NeuralNetwork gradient, NN_Matrix trainingData, float eps);
void            nn_network_learn(NeuralNetwork nn, NeuralNetwork gradient, float learningRate);


// Computes the cost using multiple threads.
// thread_count: Number of threads to spawn. If <= 1 / not specified - it runs single-threaded.
float           nn_network_cost_mt(NeuralNetwork nn, NN_Matrix trainingData, int threadCount);

// Performs backpropagation using multiple threads.
// Spawns 'thread_count' threads, each processing a subset of trainingData.
// Gradients are accumulated into thread-local buffers and summed into 'g' at the end.
void            nn_network_backprop_mt(NeuralNetwork nn, NeuralNetwork gradient, NN_Matrix trainingData, int threadCount);

#endif // NN_V1_H_
