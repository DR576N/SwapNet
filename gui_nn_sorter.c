#include "raylib_win/include/raylib.h"
#include "nn.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MAX_ARRAY_SIZE 50
#define MAX_HISTORY 500
#define MAX_INPUT_CHARS 16

// Visualization Colors
#define COL_BAR_BASE       (Color){ 200, 200, 200, 255 }
#define COL_BAR_ACTIVE     (Color){ 255, 230, 0, 255 }
#define COL_SWAP_INDICATOR (Color){ 0, 228, 48, 255 }
#define COL_NO_SWAP        (Color){ 50, 50, 50, 255 }
#define COL_ACCENT         (Color){ 100, 100, 255, 255 }
#define COL_BG             (Color){ 24, 24, 24, 255 }
#define COL_PANEL_BG       (Color){ 32, 32, 32, 255 }

typedef struct {
    char text[MAX_INPUT_CHARS];
    int cursor;
    bool active;
    Rectangle bounds;
} TextInput;

typedef struct {
    // Network parameters
    int arraySize;
    int hiddenNeurons;
    int trainSamples;
    float learningRate;
    int threadCount;
    int minVal;
    int maxVal;

    // Text inputs for settings
    TextInput inputArraySize;
    TextInput inputHidden;
    TextInput inputSamples;
    TextInput inputLearningRate;
    TextInput inputThreads;
    TextInput inputMinVal;
    TextInput inputMaxVal;

    // Neural network
    NeuralNetwork nn;
    NeuralNetwork grads;
    NN_Region region;
    NN_Matrix trainData;

    // Training state
    int epoch;
    float currentCost;
    float costHistory[MAX_HISTORY];
    int historyIndex;
    bool isTraining;

    // Simulation state
    int simArray[MAX_ARRAY_SIZE];
    float swapMapValues[MAX_ARRAY_SIZE - 1];
    int stepCounter;
    bool isSorted;
    bool autoRun;

    // UI state
    bool settingsVisible;
    bool needsReinit;
    bool initError;
    bool networkReady;
} AppState;

void InitTextInput(TextInput *input, const char *initialValue, Rectangle bounds) {
    strncpy(input->text, initialValue, MAX_INPUT_CHARS - 1);
    input->text[MAX_INPUT_CHARS - 1] = '\0';
    input->cursor = strlen(input->text);
    input->active = false;
    input->bounds = bounds;
}

void UpdateTextInput(TextInput *input) {
    Vector2 mousePoint = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        input->active = CheckCollisionPointRec(mousePoint, input->bounds);
    }

    if (input->active) {
        int key = GetCharPressed();
        while (key > 0) {
            if ((key >= 32) && (key <= 125) && (strlen(input->text) < MAX_INPUT_CHARS - 1)) {
                int len = strlen(input->text);
                input->text[len] = (char)key;
                input->text[len + 1] = '\0';
                input->cursor = len + 1;
            }
            key = GetCharPressed();
        }

        if (IsKeyPressed(KEY_BACKSPACE) && strlen(input->text) > 0) {
            input->text[strlen(input->text) - 1] = '\0';
            input->cursor = strlen(input->text);
        }
    }
}

void DrawTextInput(TextInput *input, const char *label, int labelY) {
    DrawText(label, input->bounds.x, labelY, 16, RAYWHITE);

    Color bgColor = input->active ? (Color){60, 60, 60, 255} : (Color){40, 40, 40, 255};
    DrawRectangleRec(input->bounds, bgColor);
    DrawRectangleLinesEx(input->bounds, 2, input->active ? COL_ACCENT : GRAY);

    DrawText(input->text, input->bounds.x + 5, input->bounds.y + 5, 16, RAYWHITE);

    if (input->active && ((int)(GetTime() * 2) % 2 == 0)) {
        int textWidth = MeasureText(input->text, 16);
        DrawLine(input->bounds.x + textWidth + 7, input->bounds.y + 5,
                 input->bounds.x + textWidth + 7, input->bounds.y + input->bounds.height - 5, RAYWHITE);
    }
}

bool GuiButton(Rectangle bounds, const char *text, int fontSize) {
    Vector2 mousePoint = GetMousePosition();
    bool clicked = false;
    Color color = DARKGRAY;

    if (CheckCollisionPointRec(mousePoint, bounds)) {
        color = GRAY;
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) color = BLACK;
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) clicked = true;
    }

    DrawRectangleRec(bounds, color);
    DrawRectangleLinesEx(bounds, 2, WHITE);

    int textW = MeasureText(text, fontSize);
    DrawText(text, bounds.x + bounds.width/2 - textW/2,
             bounds.y + bounds.height/2 - fontSize/2, fontSize, RAYWHITE);

    return clicked;
}

int GetCPUCoreCount() {
    #ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (int)sysinfo.dwNumberOfProcessors;
    #else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return (nprocs > 0) ? (int)nprocs : 4;
    #endif
}

void GenerateSortingData(NN_Matrix data, int arraySize, int minVal, int maxVal) {
    int swapZones = arraySize - 1;

    // Verify data dimensions
    if (data.cols != (size_t)(arraySize + swapZones)) {
        fprintf(stderr, "ERROR in GenerateSortingData: Expected %d cols, got %zu\n",
                arraySize + swapZones, data.cols);
        return;
    }

    for (size_t i = 0; i < data.rows; ++i) {
        NN_Row row = nn_matrix_row(data, i);
        NN_Row input = nn_row_slice(row, 0, arraySize);

        if (!input.elements) {
            fprintf(stderr, "ERROR: Failed to create input slice at row %zu\n", i);
            continue;
        }

        // Generate random integers within range
        for (int j = 0; j < arraySize; ++j) {
            int randomInt = minVal + (int)(nn_random_float() * (maxVal - minVal + 1));
            if (randomInt > maxVal) randomInt = maxVal;
            NN_ROW_AT(input, j) = (float)randomInt;
        }

        // Randomly introduce some duplicate values to help network learn
        // to handle equal elements (about 20% of the time)
        if (nn_random_float() < 0.1f) {
            int idx1 = (int)(nn_random_float() * arraySize);
            int idx2 = (int)(nn_random_float() * arraySize);
            if (idx1 != idx2) {
                NN_ROW_AT(input, idx2) = NN_ROW_AT(input, idx1);
            }
        }

        NN_Row output = nn_row_slice(row, arraySize, swapZones);

        if (!output.elements) {
            fprintf(stderr, "ERROR: Failed to create output slice at row %zu\n", i);
            continue;
        }

        for (int j = 0; j < swapZones; ++j) {
            float left = NN_ROW_AT(input, j);
            float right = NN_ROW_AT(input, j + 1);
            // Only swap if strictly greater (not equal)
            NN_ROW_AT(output, j) = (left > right) ? 1.0f : 0.0f;
        }
    }

    printf("Generated %zu training samples with %d inputs + %d outputs = %d total cols\n",
           data.rows, arraySize, swapZones, arraySize + swapZones);
}

void InitNetwork(AppState *app) {
    // Stop training before reinitialization
    app->isTraining = false;
    app->autoRun = false;
    app->initError = false;
    app->networkReady = false;

    // Free old memory if exists
    if (app->region.words) {
        nn_region_free(&app->region);
        app->region.words = NULL;
        app->region.size = 0;
        app->region.capacity = 0;
    }

    // Zero out network structures
    memset(&app->nn, 0, sizeof(NeuralNetwork));
    memset(&app->grads, 0, sizeof(NeuralNetwork));
    memset(&app->trainData, 0, sizeof(NN_Matrix));

    // Calculate required memory size
    // Network: 2 weight matrices + 2 bias vectors + 3 activation vectors
    // Training data: samples * (inputs + outputs)
    size_t weights_size = (app->arraySize * app->hiddenNeurons + app->hiddenNeurons * (app->arraySize - 1)) * sizeof(float);
    size_t biases_size = (app->hiddenNeurons + (app->arraySize - 1)) * sizeof(float);
    size_t activations_size = (app->arraySize + app->hiddenNeurons + (app->arraySize - 1)) * sizeof(float);
    size_t network_size = (weights_size + biases_size + activations_size) * 2; // *2 for gradient
    size_t training_size = app->trainSamples * (app->arraySize + (app->arraySize - 1)) * sizeof(float);

    size_t memory_size = network_size + training_size + (1024 * 1024); // +1MB buffer

    void* memory = malloc(memory_size);
    if (!memory) {
        fprintf(stderr, "Failed to allocate memory! Requested: %zu bytes\n", memory_size);
        app->initError = true;
        return;
    }

    memset(memory, 0, memory_size);

    app->region = (NN_Region){
        .capacity = memory_size / sizeof(uintptr_t),
        .size = 0,
        .words = (uintptr_t*)memory
    };

    size_t *arch = nn_region_alloc_ptr(&app->region, sizeof(size_t) * 3);
    arch[0] = (size_t)app->arraySize;
    arch[1] = (size_t)app->hiddenNeurons;
    arch[2] = (size_t)(app->arraySize - 1);

    printf("Creating network with architecture: [%zu, %zu, %zu]\n", arch[0], arch[1], arch[2]);

    nn_srand(time(NULL));
    app->nn = nn_network_alloc(&app->region, arch, 3, ACTIVATION_SIG);
    app->grads = nn_network_alloc(&app->region, arch, 3, ACTIVATION_SIG);

    // Verify network was allocated correctly
    if (app->nn.archCount != 3 || app->grads.archCount != 3) {
        fprintf(stderr, "ERROR: Network architecture count mismatch!\n");
        nn_region_free(&app->region);
        memset(&app->region, 0, sizeof(NN_Region));
        app->initError = true;
        return;
    }

    // Verify layer sizes
    if (app->nn.arch[0] != arch[0] || app->nn.arch[1] != arch[1] || app->nn.arch[2] != arch[2]) {
        fprintf(stderr, "ERROR: Network layer sizes don't match requested architecture!\n");
        fprintf(stderr, "Requested: [%zu, %zu, %zu], Got: [%zu, %zu, %zu]\n",
                arch[0], arch[1], arch[2],
                app->nn.arch[0], app->nn.arch[1], app->nn.arch[2]);
        nn_region_free(&app->region);
        memset(&app->region, 0, sizeof(NN_Region));
        app->initError = true;
        return;
    }

    // Check if allocation succeeded
    if (!app->nn.weights || !app->nn.biases || !app->nn.activations ||
        !app->grads.weights || !app->grads.biases || !app->grads.activations) {
        fprintf(stderr, "Failed to allocate neural network structures!\n");
        nn_region_free(&app->region);
        memset(&app->region, 0, sizeof(NN_Region));
        app->initError = true;
        return;
    }

    // Verify activation layer sizes
    if (app->nn.activations[0].cols != arch[0] ||
        app->nn.activations[2].cols != arch[2]) {
        fprintf(stderr, "ERROR: Activation layer size mismatch!\n");
        fprintf(stderr, "Input layer: expected %zu, got %zu\n", arch[0], app->nn.activations[0].cols);
        fprintf(stderr, "Output layer: expected %zu, got %zu\n", arch[2], app->nn.activations[2].cols);
        nn_region_free(&app->region);
        memset(&app->region, 0, sizeof(NN_Region));
        app->initError = true;
        return;
    }

    nn_network_rand(app->nn, -1.0f, 1.0f);

    size_t trainCols = (size_t)(app->arraySize + (app->arraySize - 1));
    printf("Allocating training data: %d samples x %zu cols\n", app->trainSamples, trainCols);

    app->trainData = nn_matrix_alloc(&app->region, app->trainSamples, trainCols);

    if (!app->trainData.elements) {
        fprintf(stderr, "Failed to allocate training data! Requested %d x %zu\n", app->trainSamples, trainCols);
        nn_region_free(&app->region);
        memset(&app->region, 0, sizeof(NN_Region));
        app->initError = true;
        return;
    }

    // Verify training data was allocated with correct dimensions
    if (app->trainData.cols != trainCols || app->trainData.rows != (size_t)app->trainSamples) {
        fprintf(stderr, "Training data dimension error! Expected %dx%zu, got %zux%zu\n",
                app->trainSamples, trainCols, app->trainData.rows, app->trainData.cols);
        nn_region_free(&app->region);
        memset(&app->region, 0, sizeof(NN_Region));
        app->initError = true;
        return;
    }

    // Verify dimensions match
    if (app->trainData.cols != trainCols) {
        fprintf(stderr, "Training data dimension mismatch! Expected %zu, got %zu\n",
                trainCols, app->trainData.cols);
        nn_region_free(&app->region);
        memset(&app->region, 0, sizeof(NN_Region));
        app->initError = true;
        return;
    }

    // Generate Training Samples
    GenerateSortingData(app->trainData, app->arraySize, -5.0f, 5.0f);

    // Final verification of dimensions
    size_t expectedCols = app->arraySize + (app->arraySize - 1);
    size_t nnInputCols = NN_NETWORK_INPUT(app->nn).cols;
    size_t nnOutputCols = NN_NETWORK_OUTPUT(app->nn).cols;

    printf("=== Network Initialization ===\n");
    printf("Architecture: %dx%dx%d\n", app->arraySize, app->hiddenNeurons, app->arraySize - 1);
    printf("NN Input cols: %zu, NN Output cols: %zu\n", nnInputCols, nnOutputCols);
    printf("Training data: %d samples x %zu cols\n", app->trainSamples, app->trainData.cols);
    printf("Expected cols: %zu, Actual cols: %zu\n", expectedCols, app->trainData.cols);
    printf("Dimension check: %zu + %zu = %zu, trainData.cols = %zu\n",
            nnInputCols, nnOutputCols, nnInputCols + nnOutputCols, app->trainData.cols);

    if (nnInputCols + nnOutputCols != app->trainData.cols) {
        fprintf(stderr, "ERROR: Dimension mismatch detected before training!\n");
        fprintf(stderr, "Network expects %zu cols but training data has %zu cols\n",
                nnInputCols + nnOutputCols, app->trainData.cols);
        nn_region_free(&app->region);
        memset(&app->region, 0, sizeof(NN_Region));
        app->initError = true;
        app->networkReady = false;
        return;
    }

    app->epoch = 0;
    app->currentCost = 1.0f;
    app->historyIndex = 0;
    app->needsReinit = false;
    app->networkReady = true;

    printf("Network initialized successfully!\n");
}

void InitApp(AppState *app) {
    app->arraySize = 20;
    app->hiddenNeurons = 10;
    app->trainSamples = 5000;
    app->learningRate = 0.1f;
    app->threadCount = GetCPUCoreCount();
    if (app->threadCount > 16) app->threadCount = 16;
    if (app->threadCount < 1) app->threadCount = 1;
    app->minVal = -100;
    app->maxVal = 100;

    app->settingsVisible = true;
    app->autoRun = false;
    app->isSorted = false;
    app->stepCounter = 0;
    app->initError = false;
    app->networkReady = false;

    // Initialize swap map to zero
    memset(app->swapMapValues, 0, sizeof(app->swapMapValues));

    // Initialize text inputs with default values
    char buf[16];
    sprintf(buf, "%d", app->arraySize);
    InitTextInput(&app->inputArraySize, buf, (Rectangle){0, 0, 0, 0});
    sprintf(buf, "%d", app->hiddenNeurons);
    InitTextInput(&app->inputHidden, buf, (Rectangle){0, 0, 0, 0});
    sprintf(buf, "%d", app->trainSamples);
    InitTextInput(&app->inputSamples, buf, (Rectangle){0, 0, 0, 0});
    sprintf(buf, "%.3f", app->learningRate);
    InitTextInput(&app->inputLearningRate, buf, (Rectangle){0, 0, 0, 0});
    sprintf(buf, "%d", app->threadCount);
    InitTextInput(&app->inputThreads, buf, (Rectangle){0, 0, 0, 0});
    sprintf(buf, "%d", app->minVal);
    InitTextInput(&app->inputMinVal, buf, (Rectangle){0, 0, 0, 0});
    sprintf(buf, "%d", app->maxVal);
    InitTextInput(&app->inputMaxVal, buf, (Rectangle){0, 0, 0, 0});

    InitNetwork(app);

    for (int i = 0; i < app->arraySize; i++) {
        app->simArray[i] = (int)(rand() % 200 - 100);
    }
}

void ApplySettings(AppState *app) {
    // Stop all operations first
    app->isTraining = false;
    app->autoRun = false;

    int newArraySize = atoi(app->inputArraySize.text);
    int newHiddenNeurons = atoi(app->inputHidden.text);
    int newTrainSamples = atoi(app->inputSamples.text);
    float newLearningRate = atof(app->inputLearningRate.text);
    int newThreadCount = atoi(app->inputThreads.text);
    float newMinVal = atof(app->inputMinVal.text);
    float newMaxVal = atof(app->inputMaxVal.text);

    // Clamp values
    if (newArraySize < 10) newArraySize = 10;
    if (newArraySize > 50) newArraySize = 50;
    if (newHiddenNeurons < 1) newHiddenNeurons = 1;
    if (newHiddenNeurons > 100) newHiddenNeurons = 100;
    if (newTrainSamples < 100) newTrainSamples = 100;
    if (newTrainSamples > 10000) newTrainSamples = 10000;
    if (newLearningRate < 0.001f) newLearningRate = 0.001f;
    if (newLearningRate > 1.0f) newLearningRate = 1.0f;
    if (newThreadCount < 1) newThreadCount = 1;
    if (newThreadCount > 16) newThreadCount = 16;

    // Apply validated values
    app->arraySize = newArraySize;
    app->hiddenNeurons = newHiddenNeurons;
    app->trainSamples = newTrainSamples;
    app->learningRate = newLearningRate;
    app->threadCount = newThreadCount;
    app->minVal = newMinVal;
    app->maxVal = newMaxVal;

    // Update text inputs with clamped values using snprintf for safety
    snprintf(app->inputArraySize.text, MAX_INPUT_CHARS, "%d", app->arraySize);
    app->inputArraySize.cursor = strlen(app->inputArraySize.text);

    snprintf(app->inputHidden.text, MAX_INPUT_CHARS, "%d", app->hiddenNeurons);
    app->inputHidden.cursor = strlen(app->inputHidden.text);

    snprintf(app->inputSamples.text, MAX_INPUT_CHARS, "%d", app->trainSamples);
    app->inputSamples.cursor = strlen(app->inputSamples.text);

    snprintf(app->inputLearningRate.text, MAX_INPUT_CHARS, "%.3f", app->learningRate);
    app->inputLearningRate.cursor = strlen(app->inputLearningRate.text);

    snprintf(app->inputThreads.text, MAX_INPUT_CHARS, "%d", app->threadCount);
    app->inputThreads.cursor = strlen(app->inputThreads.text);

    snprintf(app->inputMinVal.text, MAX_INPUT_CHARS, "%d", app->minVal);
    app->inputMinVal.cursor = strlen(app->inputMinVal.text);

    snprintf(app->inputMaxVal.text, MAX_INPUT_CHARS, "%d", app->maxVal);
    app->inputMaxVal.cursor = strlen(app->inputMaxVal.text);

    // Reinitialize network
    InitNetwork(app);

    // Generate new random array
    for (int i = 0; i < app->arraySize; i++) {
        app->simArray[i] = (int)(app->minVal + (float)rand() / RAND_MAX * (app->maxVal - app->minVal));
    }

    app->isSorted = false;
    app->stepCounter = 0;
}

void RunInference(AppState *app) {
    // Safety check
    if (!app->networkReady || !app->nn.weights || !app->nn.activations) return;

    NN_Row inputLayer = app->nn.activations[0];
    for (int i = 0; i < app->arraySize; ++i) {
        NN_ROW_AT(inputLayer, i) = app->simArray[i];
    }

    nn_network_forward(app->nn);

    NN_Row outputLayer = app->nn.activations[app->nn.archCount - 1];
    int swapZones = app->arraySize - 1;
    for (int i = 0; i < swapZones; ++i) {
        app->swapMapValues[i] = NN_ROW_AT(outputLayer, i);
    }
}

void PerformSortStep(AppState *app) {
    // Safety check
    if (!app->networkReady || !app->nn.weights || !app->nn.activations) return;

    RunInference(app);

    bool anySwap = false;
    int swapZones = app->arraySize - 1;

    for (int i = 0; i < swapZones; ++i) {
        if (app->swapMapValues[i] > 0.51f) {
            int temp = app->simArray[i];
            app->simArray[i] = app->simArray[i + 1];
            app->simArray[i + 1] = temp;
            anySwap = true;
        }
    }

    app->stepCounter++;
    app->isSorted = !anySwap;
}

int main(void) {
    int screenWidth = 1280;
    int screenHeight = 720;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(screenWidth, screenHeight, "Neural Network Sorter - Visualization");
    SetTargetFPS(300);

    AppState app;
    memset(&app, 0, sizeof(AppState));
    InitApp(&app);

    float sortTimer = 0;

    while (!WindowShouldClose()) {
        screenWidth = GetScreenWidth();
        screenHeight = GetScreenHeight();

        if (IsKeyPressed(KEY_H)) {
            app.settingsVisible = !app.settingsVisible;
        }

        int settingsPanelWidth = app.settingsVisible ? (int)(screenWidth * 0.25f) : 0;
        int vizWidth = screenWidth - settingsPanelWidth;

        if (app.isTraining) {
            int batchesPerFrame = 5;

            for (int k = 0; k < batchesPerFrame; k++) {
                nn_network_backprop_mt(app.nn, app.grads, app.trainData, app.threadCount);
                nn_network_learn(app.nn, app.grads, app.learningRate);
                app.epoch++;
            }

            if (app.epoch % 10 == 0) {
                app.currentCost = nn_network_cost_mt(app.nn, app.trainData, app.threadCount);

                if (app.historyIndex < MAX_HISTORY) {
                    app.costHistory[app.historyIndex++] = app.currentCost;
                } else {
                    for (int i = 0; i < MAX_HISTORY - 1; i++) {
                        app.costHistory[i] = app.costHistory[i + 1];
                    }
                    app.costHistory[MAX_HISTORY - 1] = app.currentCost;
                }
            }
        }

        RunInference(&app);

        if (app.autoRun && !app.isTraining && !app.isSorted && app.networkReady && app.nn.weights) {
            sortTimer += GetFrameTime();
            if (sortTimer > 0.15f) {
                PerformSortStep(&app);
                sortTimer = 0;
            }
        }

        BeginDrawing();
        ClearBackground(COL_BG);

        // Display error message if initialization failed
        if (app.initError) {
            const char *errorMsg = "ERROR: Failed to initialize neural network!";
            const char *errorMsg2 = "Try reducing training samples or array size.";
            int textW = MeasureText(errorMsg, 24);
            int textW2 = MeasureText(errorMsg2, 20);
            DrawText(errorMsg, screenWidth / 2 - textW / 2, screenHeight / 2 - 40, 24, RED);
            DrawText(errorMsg2, screenWidth / 2 - textW2 / 2, screenHeight / 2 + 10, 20, ORANGE);
            EndDrawing();
            continue;
        }

        // Settings Panel
        if (app.settingsVisible) {
            DrawRectangle(0, 0, settingsPanelWidth, screenHeight, COL_PANEL_BG);
            DrawLine(settingsPanelWidth, 0, settingsPanelWidth, screenHeight, GRAY);

            int yPos = 20;
            DrawText("SETTINGS", 20, yPos, 24, RAYWHITE);
            yPos += 40;

            DrawText("Press 'H' to hide", 20, yPos, 14, GRAY);
            yPos += 40;

            int inputWidth = settingsPanelWidth - 40;
            int inputHeight = 30;
            int spacing = 65;

            if (app.inputArraySize.bounds.width == 0) {
                app.inputArraySize.bounds = (Rectangle){20, yPos + 20, inputWidth, inputHeight};
                app.inputHidden.bounds = (Rectangle){20, yPos + spacing + 20, inputWidth, inputHeight};
                app.inputSamples.bounds = (Rectangle){20, yPos + spacing * 2 + 20, inputWidth, inputHeight};
                app.inputLearningRate.bounds = (Rectangle){20, yPos + spacing * 3 + 20, inputWidth, inputHeight};
                app.inputThreads.bounds = (Rectangle){20, yPos + spacing * 4 + 20, inputWidth, inputHeight};
                app.inputMinVal.bounds = (Rectangle){20, yPos + spacing * 5 + 20, inputWidth, inputHeight};
                app.inputMaxVal.bounds = (Rectangle){20, yPos + spacing * 6 + 20, inputWidth, inputHeight};
            }

            UpdateTextInput(&app.inputArraySize);
            DrawTextInput(&app.inputArraySize, "Array Size (10-50):", yPos);
            yPos += spacing;

            UpdateTextInput(&app.inputHidden);
            DrawTextInput(&app.inputHidden, "Hidden Neurons (1-100):", yPos);
            yPos += spacing;

            UpdateTextInput(&app.inputSamples);
            DrawTextInput(&app.inputSamples, "Training Samples (100-10000):", yPos);
            yPos += spacing;

            UpdateTextInput(&app.inputLearningRate);
            DrawTextInput(&app.inputLearningRate, "Learning Rate (0.001-1.0):", yPos);
            yPos += spacing;

            UpdateTextInput(&app.inputThreads);
            DrawTextInput(&app.inputThreads, "Threads (1-16):", yPos);
            yPos += spacing;

            UpdateTextInput(&app.inputMinVal);
            DrawTextInput(&app.inputMinVal, "Min Value:", yPos);
            yPos += spacing;

            UpdateTextInput(&app.inputMaxVal);
            DrawTextInput(&app.inputMaxVal, "Max Value:", yPos);
            yPos += spacing + 10;

            if (GuiButton((Rectangle){20, yPos, inputWidth, 40}, "APPLY CHANGES", 16)) {
                ApplySettings(&app);
            }
        }

        // Visualization Area
        int vizX = settingsPanelWidth + 20;
        int graphY = 20;

        DrawText("TRAINING CONTROL", vizX, graphY, 24, RAYWHITE);

        // Show network status
        if (!app.networkReady) {
            DrawText("Network: NOT READY", vizX, graphY + 35, 16, RED);
        } else {
            DrawText("Network: READY", vizX, graphY + 35, 16, GREEN);
        }

        char buf[128];
        sprintf(buf, "Epoch: %d", app.epoch);
        DrawText(buf, vizX, graphY + 60, 20, COL_ACCENT);

        sprintf(buf, "Error: %.6f", app.currentCost);
        DrawText(buf, vizX, graphY + 90, 20, (app.currentCost < 0.1f) ? GREEN : RED);

        // Buttons
        int btnY = graphY + 130;
        int btnSpacing = 130;
        if (GuiButton((Rectangle){vizX, btnY, 120, 35}, app.isTraining ? "PAUSE" : "TRAIN", 14)) {
            if (app.networkReady && !app.initError) {
                app.isTraining = !app.isTraining;
            }
        }
        if (GuiButton((Rectangle){vizX + btnSpacing, btnY, 120, 35}, "RESET NET", 14)) {
            app.isTraining = false;
            app.autoRun = false;
            if (app.networkReady && app.nn.weights) {
                nn_network_rand(app.nn, -1.0f, 1.0f);
            }
            app.epoch = 0;
            app.historyIndex = 0;
        }

        // Cost Graph
        int graphDrawY = graphY + 180;
        int graphDrawH = 150;
        int graphDrawW = vizWidth - 40;

        DrawRectangleLines(vizX, graphDrawY, graphDrawW, graphDrawH, Fade(RAYWHITE, 0.3f));
        DrawText("Error History", vizX, graphDrawY + graphDrawH + 5, 14, GRAY);

        float maxCostInHistory = 0.0001f;
        if (app.historyIndex > 0) {
            for (int i = 0; i < app.historyIndex; i++) {
                if (app.costHistory[i] > maxCostInHistory) {
                    maxCostInHistory = app.costHistory[i];
                }
            }
        }
        maxCostInHistory *= 1.1f;

        DrawText("Error History", vizX, graphDrawY + graphDrawH + 5, 14, GRAY);

        if (app.historyIndex > 1) {
            for (int i = 0; i < app.historyIndex - 1; i++) {
                float val1 = app.costHistory[i];
                float val2 = app.costHistory[i + 1];

                int y1 = graphDrawY + graphDrawH - (int)((val1 / maxCostInHistory) * graphDrawH);
                int y2 = graphDrawY + graphDrawH - (int)((val2 / maxCostInHistory) * graphDrawH);

                float stepX = (float)graphDrawW / MAX_HISTORY;
                int x1 = vizX + (int)(i * stepX);
                int x2 = vizX + (int)((i + 1) * stepX);

                if (y1 < graphDrawY) y1 = graphDrawY;
                if (y2 < graphDrawY) y2 = graphDrawY;
                if (y1 > graphDrawY + graphDrawH) y1 = graphDrawY + graphDrawH;
                if (y2 > graphDrawY + graphDrawH) y2 = graphDrawY + graphDrawH;

                DrawLine(x1, y1, x2, y2, YELLOW);
            }
        }

        // Array Visualization
        int arrayY = graphDrawY + graphDrawH + 40;

        DrawText("VISUALIZATION", vizX, arrayY, 24, RAYWHITE);
        DrawText("The Neural Network decides where to swap.", vizX, arrayY + 35, 14, GRAY);

        int ctrlY = arrayY + 70;
        if (GuiButton((Rectangle){vizX, ctrlY, 110, 35}, "RANDOMIZE", 14)) {
            app.autoRun = false;
            for (int i = 0; i < app.arraySize; i++) {
                app.simArray[i] = (int)(app.minVal + (float)rand() / RAND_MAX * (app.maxVal - app.minVal));
            }
            app.isSorted = false;
            app.stepCounter = 0;
        }
        if (GuiButton((Rectangle){vizX + 120, ctrlY, 110, 35}, "STEP", 14)) {
            app.autoRun = false;
            PerformSortStep(&app);
        }
        if (GuiButton((Rectangle){vizX + 240, ctrlY, 110, 35}, app.autoRun ? "PAUSE" : "AUTO SORT", 14)) {
            app.autoRun = !app.autoRun;
        }

        int barsY = ctrlY + 60;
        int barWidth = (vizWidth - 40) / app.arraySize - 5;
        if (barWidth < 10) barWidth = 10;
        int maxBarHeight = screenHeight - barsY - 150;

        float minArrayVal = app.minVal;
        float maxArrayVal = app.maxVal;

        for (int i = 0; i < app.arraySize; i++) {
            int val = app.simArray[i];
            float norm = (val - minArrayVal) / (maxArrayVal - minArrayVal);
            if (norm < 0) norm = 0;
            if (norm > 1) norm = 1;
            int h = (int)(norm * maxBarHeight);
            int x = vizX + i * (barWidth + 5);
            int y = barsY + maxBarHeight - h;

            Color col = COL_BAR_BASE;

            int swapZones = app.arraySize - 1;
            bool swappingLeft = (i > 0) && (app.swapMapValues[i - 1] > 0.51f);
            bool swappingRight = (i < swapZones) && (app.swapMapValues[i] > 0.51f);

            if (swappingLeft || swappingRight) col = COL_BAR_ACTIVE;

            DrawRectangle(x, y, barWidth, h, col);
            DrawRectangleLines(x, y, barWidth, h, BLACK);

            if (app.arraySize <= 25) {
                char valStr[16];
                sprintf(valStr, "%d", val);
                int textW = MeasureText(valStr, 10);
                DrawText(valStr, x + barWidth / 2 - textW / 2, y - 15, 10, RAYWHITE);
            }
        }

        // Swap Map
        int mapY = barsY + maxBarHeight + 40;  // Moved down from +20

        int swapZones = app.arraySize - 1;
        for (int i = 0; i < swapZones; i++) {
            int x1 = vizX + i * (barWidth + 5) + barWidth;
            int x2 = vizX + (i + 1) * (barWidth + 5);
            int cx = (x1 + x2) / 2;

            float confidence = app.swapMapValues[i];

            float radius = 5.0f + (confidence * 5.0f);
            Color c = ColorAlpha(COL_SWAP_INDICATOR, confidence);
            if (confidence < 0.5f) c = ColorAlpha(COL_NO_SWAP, 0.5f);

            DrawCircle(cx, mapY, radius, c);

            if (app.arraySize <= 25) {
                char confStr[16];
                sprintf(confStr, "%.2f", confidence);
                int textW = MeasureText(confStr, 10);
                DrawText(confStr, cx - textW / 2, mapY + 20, 10, GRAY);
            }

            if (confidence > 0.51f && app.arraySize <= 25) {
                const char *swapText = "<SWAP>";
                int textW = MeasureText(swapText, 10);
                DrawText(swapText, cx - textW / 2, mapY - 25, 10, GREEN);  // 25px above circle
            }
        }

        DrawText("SWAP MAP (NN Output)", vizX, mapY + 50, 18, GRAY);

        EndDrawing();
    }

    if (app.region.words) free(app.region.words);
    CloseWindow();

    return 0;
}
