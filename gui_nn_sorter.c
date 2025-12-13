#include "raylib.h"
#include "nn.h"
#include <stdio.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
#define ARRAY_SIZE 10
#define SWAP_ZONES (ARRAY_SIZE - 1)
#define MAX_HISTORY 500

// Visualization Colors
#define COL_BAR_BASE     (Color){ 200, 200, 200, 255 }
#define COL_BAR_ACTIVE   (Color){ 255, 230, 0, 255 }  // Yellow when moving
#define COL_SWAP_INDICATOR (Color){ 0, 228, 48, 255 } // Green for "SWAP NOW"
#define COL_NO_SWAP        (Color){ 50, 50, 50, 255 } // Dark for "No Swap"
#define COL_ACCENT         (Color){ 100, 100, 255, 255 }

// -----------------------------------------------------------------------------
// Data & Globals
// -----------------------------------------------------------------------------
typedef struct {
    NeuralNetwork nn;
    NeuralNetwork grads;
    NN_Region region;
    NN_Matrix trainData;

    // Training State
    int epoch;
    float currentCost;
    float costHistory[MAX_HISTORY];
    int historyIndex;
    bool isTraining;

    // Simulation State
    float simArray[ARRAY_SIZE]; // The array being sorted visually
    float swapMapValues[SWAP_ZONES]; // The raw NN output for the current array
    int stepCounter;
    bool isSorted;
    bool autoRun;
} AppState;

// -----------------------------------------------------------------------------
// Helper: Data Generation (Copied logic from nn_test1.c)
// -----------------------------------------------------------------------------
void generate_sorting_policy_data(NN_Matrix data, float min_val, float max_val) {
    for (size_t i = 0; i < data.rows; ++i) {
        NN_Row row = nn_matrix_row(data, i);
        NN_Row input = nn_row_slice(row, 0, ARRAY_SIZE);
        nn_row_rand(input, min_val, max_val);

        NN_Row output = nn_row_slice(row, ARRAY_SIZE, SWAP_ZONES);
        for (int j = 0; j < SWAP_ZONES; ++j) {
            float left = NN_ROW_AT(input, j);
            float right = NN_ROW_AT(input, j + 1);
            NN_ROW_AT(output, j) = (left > right) ? 1.0f : 0.0f;
        }
    }
}

// -----------------------------------------------------------------------------
// Helper: Application Initialization
// -----------------------------------------------------------------------------
void InitApp(AppState *app) {
    // 1. Initialize Memory Arena
    size_t memory_size = 1024 * 1024 * 16; // 16MB should be plenty
    void* memory = malloc(memory_size);
    app->region = (NN_Region){
        .capacity = memory_size,
        .size = 0,
        .words = (uintptr_t*)memory
    };

    // 2. Setup Network Architecture
    // Input: 10, Hidden: 40, 40, Output: 9
    size_t arch[] = {ARRAY_SIZE, 40, 40, SWAP_ZONES};

    // 3. Alloc NN
    nn_srand(time(NULL));
    app->nn = nn_network_alloc(&app->region, arch, NN_ARRAY_LEN(arch), ACTIVATION_SIG);
    app->grads = nn_network_alloc(&app->region, arch, NN_ARRAY_LEN(arch), ACTIVATION_SIG);
    nn_network_rand(app->nn, -1.0f, 1.0f);

    // 4. Generate Training Data
    // We keep a persistent set of training data for this session
    int samples = 2000;
    app->trainData = nn_matrix_alloc(&app->region, samples, ARRAY_SIZE + SWAP_ZONES);
    generate_sorting_policy_data(app->trainData, -100.0f, 100.0f);

    // 5. Init State
    app->epoch = 0;
    app->currentCost = 1.0f;
    app->historyIndex = 0;
    app->isTraining = false;
    app->autoRun = false;

    // Init Visual Array
    for(int i=0; i<ARRAY_SIZE; i++) app->simArray[i] = (float)(rand() % 100);
}

// -----------------------------------------------------------------------------
// Logic: Sorting Step
// -----------------------------------------------------------------------------
void RunInference(AppState *app) {
    // Copy simArray to network input
    NN_Row inputLayer = app->nn.activations[0];
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        NN_ROW_AT(inputLayer, i) = app->simArray[i];
    }

    // Forward pass
    nn_network_forward(app->nn);

    // Extract outputs (The Swap Map)
    NN_Row outputLayer = app->nn.activations[app->nn.archCount - 1];
    for (int i = 0; i < SWAP_ZONES; ++i) {
        app->swapMapValues[i] = NN_ROW_AT(outputLayer, i);
    }
}

void PerformSortStep(AppState *app) {
    RunInference(app);

    bool anySwap = false;

    // We apply swaps based on the NN's confidence.
    // To avoid conflicts (swapping index 1 with 0, then 1 with 2 in same frame),
    // we can do a simple left-to-right pass.
    for (int i = 0; i < SWAP_ZONES; ++i) {
        // Threshold 0.5 for boolean decision
        if (app->swapMapValues[i] > 0.5f) {
            float temp = app->simArray[i];
            app->simArray[i] = app->simArray[i+1];
            app->simArray[i+1] = temp;
            anySwap = true;
        }
    }

    app->stepCounter++;
    if (!anySwap) app->isSorted = true;
    else app->isSorted = false;
}

// Simple Raylib GUI implementation for button to avoid linking whole raygui
bool GuiButton(Rectangle bounds, const char *text) {
    Vector2 mousePoint = GetMousePosition();
    bool clicked = false;
    Color color = DARKGRAY;

    if (CheckCollisionPointRec(mousePoint, bounds)) {
        color = GRAY;
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) color = BLACK;
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) clicked = true;
    }

    DrawRectangleRec(bounds, color);
    DrawRectangleLinesEx(bounds, 1, WHITE);

    int textW = MeasureText(text, 10);
    DrawText(text, bounds.x + bounds.width/2 - textW/2, bounds.y + bounds.height/2 - 5, 10, RAYWHITE);

    return clicked;
}


// -----------------------------------------------------------------------------
// Main Loop
// -----------------------------------------------------------------------------
int main(void) {
    const int screenWidth = 1280;
    const int screenHeight = 720;

    InitWindow(screenWidth, screenHeight, "Neural Network Sorter - Visualization");
    SetTargetFPS(60);

    AppState app;
    InitApp(&app);

    while (!WindowShouldClose()) {
        // --- UPDATE ---

        // 1. Training Loop (Run multiple batches per frame for speed)
        if (app.isTraining) {
            int batchesPerFrame = 10;
            float lr = 0.1f;

            for (int k=0; k<batchesPerFrame; k++) {
                // Using single threaded for simplicity in GUI loop,
                // or use the _mt version with threadCount=1
                nn_network_backprop(app.nn, app.grads, app.trainData);
                nn_network_learn(app.nn, app.grads, lr);
                app.epoch++;
            }

            // Calculate cost occasionally for plotting
            if (app.epoch % 10 == 0) {
                app.currentCost = nn_network_cost(app.nn, app.trainData);

                // Add to history
                if (app.historyIndex < MAX_HISTORY) {
                    app.costHistory[app.historyIndex++] = app.currentCost;
                } else {
                    // Shift left
                    for(int i=0; i<MAX_HISTORY-1; i++) app.costHistory[i] = app.costHistory[i+1];
                    app.costHistory[MAX_HISTORY-1] = app.currentCost;
                }
            }
        }

        // 2. Simulation Loop
        // Always run inference to visualize the "Swap Map" in real-time
        RunInference(&app);

        if (app.autoRun && !app.isTraining && !app.isSorted) {
            // Slow down the sort so we can see it
            if (GetFrameTime() > 0) { // simple pacing
                static float timer = 0;
                timer += GetFrameTime();
                if (timer > 0.2f) { // 1 step every 200ms
                    PerformSortStep(&app);
                    timer = 0;
                }
            }
        }

        // --- DRAW ---
        BeginDrawing();
        ClearBackground(GetColor(0x181818FF)); // Dark Grey Background

        // -- LAYOUT --
        // Left: Training Stats & Controls
        // Right: The Array & Swap Map

        int leftPanelW = 300;
        int rightPanelX = 320;
        int rightPanelW = screenWidth - rightPanelX - 20;

        // [LEFT PANEL]
        DrawText("TRAINING CONTROL", 20, 20, 20, RAYWHITE);

        char buff[64];
        sprintf(buff, "Epoch: %d", app.epoch);
        DrawText(buff, 20, 60, 20, COL_ACCENT);

        sprintf(buff, "Error: %.5f", app.currentCost);
        DrawText(buff, 20, 90, 20, (app.currentCost < 0.1) ? GREEN : RED);

        // Buttons
        int btnY = 140;
        if (GuiButton((Rectangle){20, btnY, 120, 30}, app.isTraining ? "STOP" : "TRAIN")) {
            app.isTraining = !app.isTraining;
        }
        if (GuiButton((Rectangle){150, btnY, 120, 30}, "RESET NET")) {
            nn_network_rand(app.nn, -1.0f, 1.0f);
            app.epoch = 0;
            app.historyIndex = 0;
            app.isTraining = false;
        }

        // Cost Graph
        int graphH = 200;
        int graphY = 200;
        DrawRectangleLines(20, graphY, leftPanelW-40, graphH, Fade(RAYWHITE, 0.3f));

        if (app.historyIndex > 1) {
            for (int i = 0; i < app.historyIndex - 1; i++) {
                float val1 = app.costHistory[i];
                float val2 = app.costHistory[i+1];

                // Map values to height (assume max error around 4.0, min 0)
                float maxErr = 4.0f;
                int y1 = graphY + graphH - (int)((val1 / maxErr) * graphH);
                int y2 = graphY + graphH - (int)((val2 / maxErr) * graphH);

                // Map index to width
                float stepX = (float)(leftPanelW - 40) / MAX_HISTORY;
                int x1 = 20 + (int)(i * stepX);
                int x2 = 20 + (int)((i+1) * stepX);

                // Clamp
                if (y1 < graphY) y1 = graphY;
                if (y2 < graphY) y2 = graphY;

                DrawLine(x1, y1, x2, y2, YELLOW);
            }
        }
        DrawText("Error History", 20, graphY + graphH + 5, 10, GRAY);


        // [RIGHT PANEL] - The Simulation
        DrawText("VISUALIZATION", rightPanelX, 20, 20, RAYWHITE);
        DrawText("The Neural Network decides where to swap.", rightPanelX, 50, 10, GRAY);

        // Sim Controls
        if (GuiButton((Rectangle){rightPanelX, 80, 100, 30}, "RANDOMIZE")) {
            for(int i=0; i<ARRAY_SIZE; i++) app.simArray[i] = (float)(rand() % 100 - 50); // -50 to 50
            app.isSorted = false;
        }
        if (GuiButton((Rectangle){rightPanelX + 110, 80, 100, 30}, "STEP")) {
            PerformSortStep(&app);
        }
        if (GuiButton((Rectangle){rightPanelX + 220, 80, 100, 30}, app.autoRun ? "PAUSE" : "AUTO SORT")) {
            app.autoRun = !app.autoRun;
        }

        // Draw The Array
        int barWidth = (rightPanelW / ARRAY_SIZE) - 10;
        int baseY = 500;
        int maxBarHeight = 300;

        for (int i = 0; i < ARRAY_SIZE; i++) {
            float val = app.simArray[i];
            // Normalize for display (-100 to 100 approx)
            float norm = (val + 100) / 200.0f;
            int h = (int)(norm * maxBarHeight);
            int x = rightPanelX + i * (barWidth + 10);
            int y = baseY - h;

            Color col = COL_BAR_BASE;

            // Highlight if this element is involved in a high-confidence swap
            // Check left neighbor swap
            bool swappingLeft = (i > 0) && (app.swapMapValues[i-1] > 0.5f);
            // Check right neighbor swap
            bool swappingRight = (i < SWAP_ZONES) && (app.swapMapValues[i] > 0.5f);

            if (swappingLeft || swappingRight) col = COL_BAR_ACTIVE;

            DrawRectangle(x, y, barWidth, h, col);
            DrawRectangleLines(x, y, barWidth, h, BLACK);

            // Draw Value
            char valStr[16];
            sprintf(valStr, "%.0f", val);
            DrawText(valStr, x + barWidth/2 - MeasureText(valStr, 10)/2, y - 20, 10, RAYWHITE);
        }

        // Draw The Swap Map (Neural Network Output)
        // These are indicators BETWEEN the bars
        int mapBaseY = baseY + 40;

        for (int i = 0; i < SWAP_ZONES; i++) {
            int x1 = rightPanelX + i * (barWidth + 10) + barWidth;
            int x2 = rightPanelX + (i + 1) * (barWidth + 10);
            int cx = (x1 + x2) / 2;

            float confidence = app.swapMapValues[i]; // 0.0 to 1.0

            // Draw a circle indicating confidence
            // Low confidence = Small, Gray
            // High confidence = Large, Green

            float radius = 5.0f + (confidence * 15.0f);
            Color c = ColorAlpha(COL_SWAP_INDICATOR, confidence);
            if (confidence < 0.2f) c = ColorAlpha(COL_NO_SWAP, 0.5f);

            DrawCircle(cx, mapBaseY, radius, c);

            // Draw numeric confidence
            char confStr[16];
            sprintf(confStr, "%.2f", confidence);
            DrawText(confStr, cx - MeasureText(confStr, 10)/2, mapBaseY + 25, 10, GRAY);

            // If very confident, draw arrow
            if (confidence > 0.5f) {
                DrawText("<SWAP>", cx - MeasureText("<SWAP>", 10)/2, mapBaseY - 30, 10, GREEN);
            }
        }

        DrawText("SWAP MAP (NN Output)", rightPanelX, mapBaseY + 50, 20, GRAY);

        EndDrawing();
    }

    // Cleanup
    free(app.region.words);
    CloseWindow();

    return 0;
}

