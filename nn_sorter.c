#include <stdio.h>
#include <time.h>
#include "nn.h"

#define ARRAY_SIZE 10
#define SWAP_ZONES (ARRAY_SIZE - 1) // We can swap indices (0,1), (1,2)... up to (8,9)

#define TRAIN_SAMPLES 5000
#define EPOCHS 10000
#define LEARNING_RATE 0.1
#define THREAD_COUNT 16

// -----------------------------------------------------------------------------
// Data Generation: "The Supervisor"
// -----------------------------------------------------------------------------
// We teach the AI by showing it random arrays.
// For every adjacent pair, if Left > Right, we tell the AI "You should have swapped this".
void generate_sorting_policy_data(NN_Matrix data, float min_val, float max_val) {
    for (size_t i = 0; i < data.rows; ++i) {

        // Generate Input (The Array)
        NN_Row row = nn_matrix_row(data, i);
        NN_Row input = nn_row_slice(row, 0, ARRAY_SIZE);
        nn_row_rand(input, min_val, max_val);

        // Generate Target (The Swap Map)
        NN_Row output = nn_row_slice(row, ARRAY_SIZE, SWAP_ZONES);
        for (int j = 0; j < SWAP_ZONES; ++j) {
            float left = NN_ROW_AT(input, j);
            float right = NN_ROW_AT(input, j + 1);

            // If Left > Right, the correct "action" for this neuron is 1.0 (Swap)
            // Otherwise 0.0 (Don't touch)
            NN_ROW_AT(output, j) = (left > right) ? 1.0f : 0.0f;
        }
    }
}

// -----------------------------------------------------------------------------
// The "AI Controller" Loop
// -----------------------------------------------------------------------------
void ai_autonomous_sort(NeuralNetwork nn, float* arr, int n) {
    int max_passes = n+1; // Safety break (like a timeout)
    int pass = 0;

    printf("   [Start] ");
    for(int i=0; i<n; ++i) printf("%.0f ", arr[i]);
    printf("\n");

    while (pass < max_passes) {
        int any_swap_performed = 0;

        // 1. Observe the State (Load whole array into Input Layer)
        NN_Row input_layer = NN_NETWORK_INPUT(nn);
        for (int i = 0; i < n; ++i) {
            NN_ROW_AT(input_layer, i) = arr[i];
        }

        // 2. Think (Forward Prop)
        nn_network_forward(nn);

        // 3. Act (Read Output Layer: The Swap Map)
        NN_Row output_layer = NN_NETWORK_OUTPUT(nn);

        // We iterate through the AI's suggested moves.
        // We apply them sequentially in this pass to avoid race conditions.
        for (int i = 0; i < n - 1; ++i) {
            float confidence = NN_ROW_AT(output_layer, i);

            // If AI is > 51% sure we should swap index i and i+1
            if (confidence > 0.51f) {

                // Sanity Check for the "User" (us):
                // The AI might try to swap sorted elements if it's poorly trained.
                // But if trained well, it effectively implements the swap logic.

                // Perform Swap
                float temp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = temp;

                any_swap_performed = 1;
            }
        }

        if (!any_swap_performed) {
            printf("   [AI: \"I am done.\"]\n");
            break;
        }

        // Visualizing the AI's step
        printf("   [Step %d] ", pass + 1);
        for(int i=0; i<n; ++i) printf("%.0f ", arr[i]);
        printf("\n");

        pass++;
    }
}

int main(void) {
    // 1. Setup Arena
    NN_Region region = nn_region_alloc(4096 * 4096); // 1024MB
    nn_srand((uint64_t)time(0));

    // 2. Data Prep
    // Input: 10 (The Array) -> Output: 9 (The Swap Decisions)
    printf("--- [1] Generating Strategy Data ---\n");
    NN_Matrix trainData = nn_matrix_alloc(&region, TRAIN_SAMPLES, ARRAY_SIZE + SWAP_ZONES);
    generate_sorting_policy_data(trainData, -10.0f, 10.0f);                        // TRAINING DATA MATRIX

    // 3. Brain Architecture
    // Input: 10
    // Hidden: 16 (Needs enough neurons to look at local neighbors across the whole array)
    // Output: 9
    size_t arch[] = {ARRAY_SIZE, 40, 40, SWAP_ZONES};

    printf("--- [2] Initializing Autonomous Network ---\n");
    NeuralNetwork swapMap = nn_network_alloc(&region, arch, NN_ARRAY_LEN(arch), ACTIVATION_SIG);
    NeuralNetwork gradients = nn_network_alloc(&region, arch, NN_ARRAY_LEN(arch), ACTIVATION_SIG);
    nn_network_rand(swapMap, -1.0f, 1.0f); // SWAP MAP MATRIX

    printf("--- [3] Learning the Sorting Policy ---\n");
    for (int i = 0; i < EPOCHS; ++i) {
        // Using Multi-Threaded Backprop
        nn_network_backprop_mt(swapMap, gradients, trainData, THREAD_COUNT);
        nn_network_learn(swapMap, gradients, LEARNING_RATE);

        if (i % 1000 == 0) {
            // Using Multi-Threaded Cost Calculation
            float cost = nn_network_cost_mt(swapMap, trainData, THREAD_COUNT);
            printf("Epoch %d, Error: %f\n", i, cost);
        }
    }
    printf("Epoch %d, Error: %f\n", EPOCHS, nn_network_cost_mt(swapMap, trainData, THREAD_COUNT));

    printf("\n--- [4] Test: Giving the AI a random array ---\n");
    float test_arr[] = {-1, 9, 5,-10 ,-4 ,6 ,-8 ,1 ,-8 ,3};

        // Run the Autonomous Sort
        ai_autonomous_sort(swapMap, test_arr, ARRAY_SIZE);

        printf("   [Result] ");
        for(int i=0; i< ARRAY_SIZE; ++i) printf("%.0f ", test_arr[i]);
        printf("\n");

    nn_region_free(&region);
    return 0;
}
