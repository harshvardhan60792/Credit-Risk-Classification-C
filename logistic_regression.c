#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 1. Include the generated dataset
#include "dataset.h" 

#define NUM_FEATURES 20
#define EPOCHS 2000          // Number of times we loop through the data
#define LEARNING_RATE 0.05   // How fast the model learns

// Structure to hold a single row of data
typedef struct {
    double features[NUM_FEATURES];
    int label;  // 1 = Good, 2 = Bad
    int y;      // Mapped to 0 and 1 for the Sigmoid math
} DataPoint;

// Global model parameters
double weights[NUM_FEATURES];
double bias = 0.0;

// 2. The Sigmoid Activation Function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// 3. Feature Scaling: Min-Max Normalization (Crucial for Gradient Descent)
void normalizeData(DataPoint dataset[], int size) {
    if (size == 0) return;
    for (int i = 0; i < NUM_FEATURES; ++i) {
        double minVal = dataset[0].features[i];
        double maxVal = dataset[0].features[i];

        for (int j = 0; j < size; ++j) {
            if (dataset[j].features[i] < minVal) minVal = dataset[j].features[i];
            if (dataset[j].features[i] > maxVal) maxVal = dataset[j].features[i];
        }

        for (int j = 0; j < size; ++j) {
            if (maxVal - minVal == 0.0) {
                dataset[j].features[i] = 0.0;
            } else {
                dataset[j].features[i] = (dataset[j].features[i] - minVal) / (maxVal - minVal);
            }
        }
    }
}

// 4. Train using Gradient Descent
void trainLogisticRegression(DataPoint trainData[], int trainSize) {
    // Initialize weights to 0
    for(int i = 0; i < NUM_FEATURES; i++) {
        weights[i] = 0.0;
    }
    bias = 0.0;

    // Gradient Descent Loop
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        double weightGradients[NUM_FEATURES] = {0.0};
        double biasGradient = 0.0;

        // Calculate predictions and errors for the entire training set
        for(int i = 0; i < trainSize; i++) {
            // Calculate z = (w1*x1 + w2*x2 ... + b)
            double z = bias;
            for(int j = 0; j < NUM_FEATURES; j++) {
                z += weights[j] * trainData[i].features[j];
            }
            
            // Apply Sigmoid to get predicted probability (y_hat)
            double y_hat = sigmoid(z);
            double error = y_hat - trainData[i].y;

            // Accumulate gradients
            biasGradient += error;
            for(int j = 0; j < NUM_FEATURES; j++) {
                weightGradients[j] += error * trainData[i].features[j];
            }
        }

        // Update Weights and Bias using the Learning Rate
        bias -= LEARNING_RATE * (biasGradient / trainSize);
        for(int j = 0; j < NUM_FEATURES; j++) {
            weights[j] -= LEARNING_RATE * (weightGradients[j] / trainSize);
        }
    }
}

// 5. Make a Prediction
int predictLogisticRegression(DataPoint testPoint) {
    double z = bias;
    for(int j = 0; j < NUM_FEATURES; j++) {
        z += weights[j] * testPoint.features[j];
    }
    
    double probability = sigmoid(z);
    
    // Threshold at 0.5. If prob >= 0.5, predict Class 1 (Bad Credit)
    if (probability >= 0.5) {
        return 2; // Map back to original dataset label (Bad)
    } else {
        return 1; // Map back to original dataset label (Good)
    }
}

int main() {
    DataPoint dataset[NUM_ROWS];
    
    // 6. Load Data directly from memory
    for (int i = 0; i < NUM_ROWS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            dataset[i].features[j] = raw_data[i][j];
        }
        dataset[i].label = (int)raw_data[i][20]; 
        
        // Logistic regression math needs 0 and 1, not 1 and 2.
        // We map Label 1 (Good) -> y=0, and Label 2 (Bad) -> y=1
        if (dataset[i].label == 1) {
            dataset[i].y = 0;
        } else {
            dataset[i].y = 1;
        }
    }

    // Normalize Data (Mandatory for Logistic Regression)
    normalizeData(dataset, NUM_ROWS);

    // Train/Test Split (80% Train, 20% Test)
    int trainSize = (int)(NUM_ROWS * 0.8);
    int testSize = NUM_ROWS - trainSize;

    DataPoint trainData[trainSize];
    DataPoint testData[testSize];

    for (int i = 0; i < trainSize; i++) trainData[i] = dataset[i];
    for (int i = 0; i < testSize; i++) testData[i] = dataset[trainSize + i];

    // Train the Model
    trainLogisticRegression(trainData, trainSize);

    // Test the Model
    int correctPredictions = 0;
    for (int i = 0; i < testSize; ++i) {
        int prediction = predictLogisticRegression(testData[i]);
        if (prediction == testData[i].label) {
            correctPredictions++;
        }
    }

    double accuracy = ((double)correctPredictions / testSize) * 100.0;
    
    printf("--- Pure C Logistic Regression Results ---\n");
    printf("Total Valid Data Points Loaded: %d\n", NUM_ROWS);
    printf("Training Size: %d\n", trainSize);
    printf("Testing Size: %d\n", testSize);
    printf("Epochs run: %d | Learning Rate: %.2f\n", EPOCHS, LEARNING_RATE);
    printf("Accuracy: %.2f%%\n", accuracy);

    return 0;
}