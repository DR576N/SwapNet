#include "nn.h"
#include <time.h>
#include <pthread.h>

// -----------------------------------------------------------------------------
// Math & Random Helpers
// -----------------------------------------------------------------------------

// Fast Linear Congruential Generator
// Much faster than standard rand() and portable.
static uint64_t rng_state = 0x853242653;

/**
 * @brief Seed random number generator
**/
void nn_srand(uint64_t seed) {
    rng_state = seed;
}

float nn_random_float(void) {
    // Linear Congruential Generator constants
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)rng_state / (double)UINT64_MAX); // [0, 1]
}

float sigmoid_activation(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float y) {
    return y * (1.0f - y);
}

// Leaky ReLU
float relu_activation(float x) {
    return x > 0.0f ? x : x * NN_RELU_PARAM;
}

float relu_derivative(float y) {
    return y >= 0.0f ? 1.0f : NN_RELU_PARAM;
}


float tanh_activation(float x) {
    return tanh(x);
}


float tanh_derivative(float y) {
    return 1.0f - y * y;
}


float sin_activation(float x) {
    return sinf(x);
}


float sin_derivative(float y) {
    return sqrtf(1.0f - y * y);
}

static NN_Activation activations[] = {
    [ACTIVATION_SIG]  = {sigmoid_activation,sigmoid_derivative},
    [ACTIVATION_RELU] = {relu_activation,   relu_derivative},
    [ACTIVATION_TANH] = {tanh_activation,   tanh_derivative},
    [ACTIVATION_SIN]  = {sin_activation,    sin_derivative},
};

/**
 * @brief Applies the selected activation function to an input value.
 *
 * @param x The input value to the activation function.
 * @param act The type of activation (sigmoid, ReLU, tanh, or sin).
 *
 * @return The output of the activation function.
**/
float nn_activate(float x, ActivationType act) {
    if (act < 0 || act >= NN_ARRAY_LEN(activations)) return 0.0f;
    return activations[act].activation_func(x);
}

/**
 * @brief Computes the derivative of the activation function.
 *
 * The derivative is evaluated with respect to the activation output (y).
 *
 * @param y The output value from the activation function.
 * @param act The type of activation function used.
 *
 * @return The derivative value.
**/
float nn_activate_deriv(float y, ActivationType act) {
    if (act < 0 || act >= NN_ARRAY_LEN(activations)) return 0.0f;
    return activations[act].derivative_func(y);
}

/**
 * @brief Allocates a new memory arena.
 *
 * @param capacityInBytes The desired capacity of the arena in bytes.
**/
NN_Region nn_region_alloc(size_t capacityInBytes) {
    NN_Region r = {0};
    size_t word_size = sizeof(*r.words);
    size_t capacity_words = (capacityInBytes + word_size - 1) / word_size;
    r.words = NN_MALLOC(capacity_words * word_size);
    r.capacity = capacity_words;
    return r;
}

/**
 * @brief Allocates memory within the arena.
 *
 * If the arena is NULL, falls back to standard malloc.
 *
 * @param r Pointer to the arena (may be NULL).
 * @param size_bytes Number of bytes to allocate.
 *
 * @return Pointer to the newly allocated memory.
 */
void *nn_region_alloc_ptr(NN_Region *r, size_t size_bytes) {
    if (!r) return NN_MALLOC(size_bytes);
    size_t word_size = sizeof(*r->words);
    size_t size_words = (size_bytes + word_size - 1) / word_size;

    if (r->size + size_words > r->capacity) {
        NN_ASSERT(0 && "Region Out of Memory");
        return NULL;
    }

    void *result = &r->words[r->size];
    r->size += size_words;
    return result;
}

/**
 * @brief Frees all memory associated with the arena.
 *
 * @param r Pointer to the arena to free.
**/
void nn_region_free(NN_Region *r) {
    if (r->words) free(r->words);
    r->words = NULL;
    r->size = 0;
    r->capacity = 0;
}

// -----------------------------------------------------------------------------
// Matrix Operations
// -----------------------------------------------------------------------------

/**
 * @brief Allocates a matrix structure from the arena.
 *
 * @param r Pointer to the arena (may be NULL for malloc fallback).
 *
 * @return An initialized NN_Matrix with allocated elements.
**/
NN_Matrix nn_matrix_alloc(NN_Region *r, size_t rows, size_t cols) {
    NN_Matrix m = {rows, cols, NULL};
    m.elements = nn_region_alloc_ptr(r, sizeof(*m.elements) * rows * cols);
    return m;
}

/**
 * @brief Performs matrix multiplication: dst += a × b.
 * The destination is first zeroed. Dimensions must satisfy a.cols == b.rows
 * and dst dimensions match the result size.
 *
 * @param dst - Destination matrix (result)
**/
void nn_matrix_dot(NN_Matrix dst, NN_Matrix a, NN_Matrix b) {
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    nn_matrix_fill(dst, 0.0f);

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t k = 0; k < a.cols; ++k) {
            float a_val = NN_MATRIX_AT(a, i, k);
            for (size_t j = 0; j < b.cols; ++j) {
                NN_MATRIX_AT(dst, i, j) += a_val * NN_MATRIX_AT(b, k, j);
            }
        }
    }
}

/**
 * @brief Retrieves a view of a specific row in the matrix.
 *
 * @param m The source matrix.
 * @param row The zero-based row index.
 *
 * @return An NN_Row representing the requested row.
**/
NN_Row nn_matrix_row(NN_Matrix m, size_t row) {
    return (NN_Row){m.cols, &NN_MATRIX_AT(m, row, 0)};
}


/**
 * @brief Copies entire matrix from src (source) to dst (destination)
**/
void nn_matrix_copy(NN_Matrix dst, NN_Matrix src) {
    NN_ASSERT(dst.rows == src.rows && dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows * dst.cols; ++i) {
        dst.elements[i] = src.elements[i];
    }
}

/**
 * @brief Computes a matrix sum
 * @param dst - Destination matrix (result)
**/
void nn_matrix_sum(NN_Matrix dst, NN_Matrix a) {
    NN_ASSERT(dst.rows == a.rows && dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows * dst.cols; ++i) {
        dst.elements[i] += a.elements[i];
    }
}

/**
 * @brief Applies the activation function element-wise to the matrix (in-place).
 *
 * @param m The matrix to modify.
 * @param act The activation type to apply.
**/
void nn_matrix_activate(NN_Matrix m, ActivationType act) {
    for (size_t i = 0; i < m.rows * m.cols; ++i) {
        m.elements[i] = nn_activate(m.elements[i], act);
    }
}

/**
 * @brief Fills all elements of a matrix with a constant value.
 *
 * @param m The matrix to fill.
 * @param x The constant value.
**/
void nn_matrix_fill(NN_Matrix m, float x) {
    for (size_t i = 0; i < m.rows * m.cols; ++i) {
        m.elements[i] = x;
    }
}

/**
 * @brief Fills matrix elements with uniform random values.
 *
 * @param m The matrix to randomize.
 * @param low Lower bound of the random range (inclusive).
 * @param high Upper bound of the random range (inclusive).
**/
void nn_matrix_rand(NN_Matrix m, float low, float high) {
    for (size_t i = 0; i < m.rows * m.cols; ++i) {
        m.elements[i] = nn_random_float() * (high - low) + low;
    }
}

/**
 * @brief Prints a matrix to standard output.
 *
 * @param m The matrix to print.
 * @param name Optional name/label for the matrix.
 * @param padding Number of spaces to indent each line.
**/
void nn_matrix_print(NN_Matrix m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int)padding, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%8.4f ", NN_MATRIX_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

// -----------------------------------------------------------------------------
// Neural Network Core
// -----------------------------------------------------------------------------


/**
 * @brief Allocates a complete neural network from the arena.
 *
 * @param r Pointer to the arena (may be NULL).
 * @param arch Array specifying neuron counts per layer (architecture).
 * @param arch_count Number of elements in the architecture array.
 * @param act Activation type used for all hidden-to-output transitions.
 *
 * @return An initialized NeuralNetwork ready for use.
**/
NeuralNetwork nn_network_alloc(NN_Region *r, size_t *arch, size_t archCount, ActivationType act) {
    NeuralNetwork nn = {arch, archCount, NULL, NULL, NULL, act};
    if (archCount == 0) return nn;

    nn.weights = nn_region_alloc_ptr(r, sizeof(*nn.weights) * (archCount - 1));
    nn.biases = nn_region_alloc_ptr(r, sizeof(*nn.biases) * (archCount - 1));
    nn.activations = nn_region_alloc_ptr(r, sizeof(*nn.activations) * archCount);

    // Input layer
    nn.activations[0] = nn_row_alloc(r, arch[0]);

    // Hidden & Output layers
    for (size_t i = 1; i < archCount; ++i) {
        nn.weights[i - 1] = nn_matrix_alloc(r, arch[i - 1], arch[i]);
        nn.biases[i - 1] = nn_row_alloc(r, arch[i]);
        nn.activations[i] = nn_row_alloc(r, arch[i]);
    }
    return nn;
}


/**
 * @brief Sets all weights, biases, and activations to zero.
 *
 * @param nn The neural network to zero.
**/
void nn_network_zero(NeuralNetwork nn) {
    for (size_t i = 0; i < nn.archCount - 1; ++i) {
        nn_matrix_fill(nn.weights[i], 0);
        nn_row_fill(nn.biases[i], 0);
    }
    // Zero activations including input/output
    for(size_t i = 0; i < nn.archCount; ++i) {
        nn_row_fill(nn.activations[i], 0);
    }
}

/**
 * @brief Initializes weights and biases with random values.
 *
 * @param nn The neural network.
 * @param low Lower bound for randomization.
 * @param high Upper bound for randomization.
**/
void nn_network_rand(NeuralNetwork nn, float low, float high) {
    for (size_t i = 0; i < nn.archCount - 1; ++i) {
        nn_matrix_rand(nn.weights[i], low, high);
        nn_row_rand(nn.biases[i], low, high);
    }
}

/**
 * @brief Executes a forward pass through the network.
 * Input must be placed in the input activation row beforehand.
**/
void nn_network_forward(NeuralNetwork nn) {
    for (size_t i = 0; i < nn.archCount - 1; ++i) {
        NN_Matrix input_act = nn_row_to_matrix(nn.activations[i]);
        NN_Matrix output_act = nn_row_to_matrix(nn.activations[i + 1]);

        // y = x * w
        nn_matrix_dot(output_act, input_act, nn.weights[i]);
        // y = y + bias
        nn_matrix_sum(output_act, nn_row_to_matrix(nn.biases[i]));
        // y = \sigmoid(y) (for example)
        nn_matrix_activate(output_act, nn.actType);
    }
}


/**
 * @brief Calculates the Mean Squared Error cost over the training set.
 *
 * Each row in trainingData is concatenated [input | expected output].
 *
 * @param nn The neural network.
 * @param trainingData Matrix containing training examples.
 *
 * @return Average MSE cost across all samples.
**/
float nn_network_cost(NeuralNetwork nn, NN_Matrix trainingData) {
    NN_ASSERT(NN_NETWORK_INPUT(nn).cols + NN_NETWORK_OUTPUT(nn).cols == trainingData.cols);

    size_t n = trainingData.rows;
    float cost = 0.0f;
    size_t in_cols = NN_NETWORK_INPUT(nn).cols;
    size_t out_cols = NN_NETWORK_OUTPUT(nn).cols;

    for (size_t i = 0; i < n; ++i) {
        NN_Row row = nn_matrix_row(trainingData, i);
        NN_Row x = nn_row_slice(row, 0, in_cols);
        NN_Row y = nn_row_slice(row, in_cols, out_cols);

        nn_row_copy(NN_NETWORK_INPUT(nn), x);
        nn_network_forward(nn);

        for (size_t j = 0; j < out_cols; ++j) {
            float d = NN_ROW_AT(NN_NETWORK_OUTPUT(nn), j) - NN_ROW_AT(y, j);
            cost += d * d;
        }
    }
    return cost / n;
}

// -----------------------------------------------------------------------------
// Training / Backpropagation
// -----------------------------------------------------------------------------

/**
 * @brief Computes gradients via backpropagation.
 * Accumulates gradients into 'gradient' without allocating memory.
 * 'gradient' must have the same architecture as 'nn'.
 * Gradients are averaged and stored in the provided gradient network.
 *
 * @param nn The neural network.
 * @param gradient Pre-allocated network (same architecture) to hold gradients.
 * @param trainingData Training data matrix.
**/
void nn_network_backprop(NeuralNetwork nn, NeuralNetwork gradient, NN_Matrix trainingData) {
    size_t n = trainingData.rows;
    size_t inCols = NN_NETWORK_INPUT(nn).cols;
    size_t outCols = NN_NETWORK_OUTPUT(nn).cols;

    NN_ASSERT(inCols + outCols == trainingData.cols);

    // Assume 'gradient' is already allocated.
    // Zero it before accumulation.
    nn_network_zero(gradient);

    for (size_t i = 0; i < n; ++i) {
        NN_Row row = nn_matrix_row(trainingData, i);
        NN_Row in = nn_row_slice(row, 0, inCols);
        NN_Row expected = nn_row_slice(row, inCols, outCols);

        nn_row_copy(NN_NETWORK_INPUT(nn), in);

        nn_network_forward(nn);

        // Clear gradient activations for reuse as delta buffers
        for (size_t j = 0; j < nn.archCount; ++j) {
            nn_row_fill(gradient.activations[j], 0);
        }

        for (size_t j = 0; j < outCols; ++j) {
            float a = NN_ROW_AT(NN_NETWORK_OUTPUT(nn), j);
            float y = NN_ROW_AT(expected, j);
            // Delta = (a - y) [derivative of cost]
            NN_ROW_AT(NN_NETWORK_OUTPUT(gradient), j) = (a - y);
        }

        // Backpropagate
        // Iterate backwards from the last layer
        float s = 2.0f; // Scaling factor for MSE derivative

        for (size_t l = nn.archCount - 1; l > 0; --l) {
            NN_Matrix w = nn.weights[l - 1];
            NN_Row currentAct = nn.activations[l];
            NN_Row prevAct = nn.activations[l - 1];

            NN_Row currentGradAct = gradient.activations[l];   // Contains errors (deltas) from next layer
            NN_Row prevGradAct = gradient.activations[l - 1];  // We will accumulate errors here for previous layer

            for (size_t j = 0; j < currentAct.cols; ++j) {
                float a = NN_ROW_AT(currentAct, j);
                float da = NN_ROW_AT(currentGradAct, j); // The error arriving at this neuron
                float qa = nn_activate_deriv(a, nn.actType); // \sigma'(z)

                float delta = s * da * qa;

                // Accumulate Bias Gradient
                NN_ROW_AT(gradient.biases[l - 1], j) += delta;

                // Calculate Weight Gradients & Propagate Error to Previous Layer
                for (size_t k = 0; k < prevAct.cols; ++k) {
                    float pa = NN_ROW_AT(prevAct, k);
                    float weight_val = NN_MATRIX_AT(w, k, j);

                    // Accumulate Weight Gradient
                    NN_MATRIX_AT(gradient.weights[l - 1], k, j) += delta * pa;

                    // Propagate Error Backwards: delta * weight
                    NN_ROW_AT(prevGradAct, k) += delta * weight_val;
                }
            }
        }
    }

    // Average gradients
    for (size_t i = 0; i < gradient.archCount - 1; ++i) {
        for (size_t j = 0; j < gradient.weights[i].rows * gradient.weights[i].cols; ++j) {
            gradient.weights[i].elements[j] /= n;
        }
        for (size_t k = 0; k < gradient.biases[i].cols; ++k) {
            gradient.biases[i].elements[k] /= n;
        }
    }
}

/**
 * @brief Applies gradient descent update to network parameters.
 *
 * @param nn The neural network to update.
 * @param gradient Gradient network containing averaged gradients.
 * @param learningRate Step size for the update.
**/
void nn_network_learn(NeuralNetwork nn, NeuralNetwork g, float learningRate) {
    for (size_t i = 0; i < nn.archCount - 1; ++i) {
        for (size_t j = 0; j < nn.weights[i].rows * nn.weights[i].cols; ++j) {
            nn.weights[i].elements[j] -= learningRate * g.weights[i].elements[j];
        }
        for (size_t k = 0; k < nn.biases[i].cols; ++k) {
            nn.biases[i].elements[k] -= learningRate * g.biases[i].elements[k];
        }
    }
}

// -----------------------------------------------------------------------------
// Finite Difference
// -----------------------------------------------------------------------------

/**
 * @brief Approximates gradients using finite differences (for verification).
 *
 * @param nn The neural network.
 * @param gradient Network to store computed gradients.
 * @param trainingData Training data.
 * @param eps Perturbation size for numerical differentiation. (small wiggle variable)
**/
void nn_network_finite_diff(NeuralNetwork nn, NeuralNetwork g, NN_Matrix trainingData, float eps) {
    float saved;
    float c = nn_network_cost(nn, trainingData);

    for (size_t i = 0; i < nn.archCount - 1; ++i) {
        for (size_t j = 0; j < nn.weights[i].rows * nn.weights[i].cols; ++j) {
            saved = nn.weights[i].elements[j];
            nn.weights[i].elements[j] += eps;
            g.weights[i].elements[j] = (nn_network_cost(nn, trainingData) - c) / eps;
            nn.weights[i].elements[j] = saved;
        }

        for (size_t k = 0; k < nn.biases[i].cols; ++k) {
            saved = nn.biases[i].elements[k];
            nn.biases[i].elements[k] += eps;
            g.biases[i].elements[k] = (nn_network_cost(nn, trainingData) - c) / eps;
            nn.biases[i].elements[k] = saved;
        }
    }
}

// -----------------------------------------------------------------------------
// Threading Implementation
// -----------------------------------------------------------------------------

// Internal Helper: Clone network topology (weights/biases shared, activations new)
// Returns a network struct that owns its own activation memory (via malloc).
static NeuralNetwork nn_internal_clone_for_thread(NeuralNetwork src) {
    NeuralNetwork dst = src;
    // Duplicate activations. Weights and biases are read-only during backprop.
    dst.activations = NN_MALLOC(sizeof(NN_Row) * src.archCount);

    for (size_t i = 0; i < src.archCount; ++i) {
        size_t cols = src.arch[i];
        dst.activations[i].cols = cols;
        dst.activations[i].elements = NN_MALLOC(sizeof(float) * cols);
        // Copying initial state is not strictly necessary for activations (they get overwritten)
        // but good practice if stateful. Here we just zero.
        memset(dst.activations[i].elements, 0, sizeof(float) * cols);
    }
    return dst;
}

static void nn_internal_free_clone(NeuralNetwork net) {
    if (net.activations) {
        for (size_t i = 0; i < net.archCount; ++i) {
            NN_FREE(net.activations[i].elements);
        }
        NN_FREE(net.activations);
    }
}

// Internal Helper: Allocate full gradient structure (deep copy of topology)
// Since this is temporary for threads, using malloc directly.
static NeuralNetwork nn_internal_alloc_grad(NeuralNetwork src) {
    NeuralNetwork g = {0};
    g.arch = src.arch;
    g.archCount = src.archCount;
    g.actType = src.actType;

    g.activations = NN_MALLOC(sizeof(NN_Row) * src.archCount);
    for (size_t i = 0; i < src.archCount; ++i) {
        g.activations[i].cols = src.arch[i];
        g.activations[i].elements = NN_MALLOC(sizeof(float) * src.arch[i]);
        memset(g.activations[i].elements, 0, sizeof(float) * src.arch[i]);
    }

    // Safely allocate weights/biases only if they exist (archCount > 1)
    if (src.archCount > 1) {
        g.weights = NN_MALLOC(sizeof(NN_Matrix) * (src.archCount - 1));
        g.biases = NN_MALLOC(sizeof(NN_Row) * (src.archCount - 1));

        for (size_t i = 0; i < src.archCount - 1; ++i) {
            // Weights
            size_t rows = src.arch[i];
            size_t cols = src.arch[i+1];
            g.weights[i].rows = rows;
            g.weights[i].cols = cols;
            g.weights[i].elements = NN_MALLOC(sizeof(float) * rows * cols);
            memset(g.weights[i].elements, 0, sizeof(float) * rows * cols);

            // Biases
            g.biases[i].cols = cols;
            g.biases[i].elements = NN_MALLOC(sizeof(float) * cols);
            memset(g.biases[i].elements, 0, sizeof(float) * cols);
        }
    } else {
        g.weights = NULL;
        g.biases = NULL;
    }

    return g;
}

static void nn_internal_free_grad(NeuralNetwork gradient) {
    for (size_t i = 0; i < gradient.archCount; ++i) {
        NN_FREE(gradient.activations[i].elements);
    }

    if (gradient.archCount > 1) {
        for (size_t i = 0; i < gradient.archCount - 1; ++i) {
            NN_FREE(gradient.weights[i].elements);
            NN_FREE(gradient.biases[i].elements);
        }
        NN_FREE(gradient.weights);
        NN_FREE(gradient.biases);
    }

    NN_FREE(gradient.activations);
}

typedef struct {
    NeuralNetwork nn;    // Thread-local clone (activations unique)
    NeuralNetwork g;     // Thread-local accumulator (full unique)
    NN_Matrix data;      // Slice of data
    float costOut;       // Output for cost function
} ThreadArgs;

// Worker for Cost Calculation
static void *worker_cost(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;

    // Note: nn_network_cost averages the cost.
    float avgCost = nn_network_cost(args->nn, args->data);
    args->costOut = avgCost * args->data.rows;

    return NULL;
}

/**
 * @brief Multi-threaded cost computation.
 *
 * Falls back to single-threaded if threadCount <= 1.
 *
 * @param nn The neural network.
 * @param trainingData Training data matrix.
 * @param threadCount Desired number of worker threads.
 *
 * @return Average cost over the dataset.
 */
float nn_network_cost_mt(NeuralNetwork nn, NN_Matrix trainingData, int threadCount) {
    if (threadCount <= 1) return nn_network_cost(nn, trainingData);

    size_t n = trainingData.rows;
    size_t rowsPerThread = n / threadCount;
    size_t remainder = n % threadCount;

    pthread_t *threads = NN_MALLOC(sizeof(pthread_t) * threadCount);
    ThreadArgs *args = NN_MALLOC(sizeof(ThreadArgs) * threadCount);

    size_t currentRow = 0;
    for (int t = 0; t < threadCount; ++t) {
        size_t count = rowsPerThread + ((size_t)t < remainder ? 1 : 0);

        args[t].nn = nn_internal_clone_for_thread(nn);
        // Create a view into the matrix for this batch
        args[t].data = trainingData;
        args[t].data.rows = count;
        args[t].data.elements = &NN_MATRIX_AT(trainingData, currentRow, 0);

        currentRow += count;

        pthread_create(&threads[t], NULL, worker_cost, &args[t]);
    }

    float totalCost = 0.0f;
    for (int t = 0; t < threadCount; ++t) {
        pthread_join(threads[t], NULL);
        totalCost += args[t].costOut;
        nn_internal_free_clone(args[t].nn);
    }

    NN_FREE(threads);
    NN_FREE(args);

    return totalCost / n;
}

// Worker for Backprop
static void *worker_backprop(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;

    // Use the standard backprop, but use thread-local NN and Gradient accumulator.
    // Note: nn_network_backprop zeros the gradient at start.
    nn_network_backprop(args->nn, args->g, args->data);

    // nn_network_backprop averages the gradients at the end.
    // We need to un-average them (multiply by count) to sum them correctly later,
    // or just handle the weighted averaging in the main thread.
    // un-averaging here to get raw sums.

    float count = (float)args->data.rows;
    NeuralNetwork g = args->g;

    for (size_t i = 0; i < g.archCount - 1; ++i) {
        size_t weightLen = g.weights[i].rows * g.weights[i].cols;
        for (size_t k = 0; k < weightLen; ++k) g.weights[i].elements[k] *= count;

        size_t biasLen = g.biases[i].cols;
        for (size_t k = 0; k < biasLen; ++k) g.biases[i].elements[k] *= count;
    }

    return NULL;
}


/**
 * @brief Multi-threaded backpropagation.
 *
 * Gradients are accumulated across threads and averaged in the provided gradient network.
 *
 * @param nn The neural network.
 * @param gradient Pre-allocated gradient network.
 * @param trainingData Training data matrix.
 * @param threadCount Number of worker threads (<=1 uses single-threaded version).
 */
void nn_network_backprop_mt(NeuralNetwork nn, NeuralNetwork gradient, NN_Matrix trainingData, int threadCount) {
    if (threadCount <= 1) {
        nn_network_backprop(nn, gradient, trainingData);
        return;
    }

    size_t n = trainingData.rows;
    size_t rowsPerThread = n / threadCount;
    size_t remainder = n % threadCount;

    pthread_t *threads = NN_MALLOC(sizeof(pthread_t) * threadCount);
    ThreadArgs *args = NN_MALLOC(sizeof(ThreadArgs) * threadCount);

    size_t currentRow = 0;
    for (int t = 0; t < threadCount; ++t) {
        // Cast t to size_t to silence -Wsign-compare
        size_t count = rowsPerThread + ((size_t)t < remainder ? 1 : 0);

        // Clone NN structure (shared weights, new activations)
        args[t].nn = nn_internal_clone_for_thread(nn);

        // Allocate temporary gradient structure
        args[t].g = nn_internal_alloc_grad(nn);

        // Data slice
        args[t].data = trainingData;
        args[t].data.rows = count;
        args[t].data.elements = &NN_MATRIX_AT(trainingData, currentRow, 0);

        currentRow += count;

        pthread_create(&threads[t], NULL, worker_backprop, &args[t]);
    }

    // Zero the main gradient accumulator
    nn_network_zero(gradient);

    // Join and accumulate
    for (int t = 0; t < threadCount; ++t) {
        pthread_join(threads[t], NULL);

        NeuralNetwork local_g = args[t].g;

        // Sum local gradients into main g
        for (size_t i = 0; i < gradient.archCount - 1; ++i) {
            size_t w_len = gradient.weights[i].rows * gradient.weights[i].cols;
            for (size_t k = 0; k < w_len; ++k) {
                gradient.weights[i].elements[k] += local_g.weights[i].elements[k];
            }

            size_t b_len = gradient.biases[i].cols;
            for (size_t k = 0; k < b_len; ++k) {
                gradient.biases[i].elements[k] += local_g.biases[i].elements[k];
            }
        }

        // Cleanup thread-local memory
        nn_internal_free_clone(args[t].nn);
        nn_internal_free_grad(args[t].g);
    }

    // Final Average over total N
    float inv_n = 1.0f / n;
    for (size_t i = 0; i < gradient.archCount - 1; ++i) {
        size_t weightLen = gradient.weights[i].rows * gradient.weights[i].cols;
        for (size_t k = 0; k < weightLen; ++k) gradient.weights[i].elements[k] *= inv_n;

        size_t biasLen = gradient.biases[i].cols;
        for (size_t k = 0; k < biasLen; ++k) gradient.biases[i].elements[k] *= inv_n;
    }

    NN_FREE(threads);
    NN_FREE(args);
}
