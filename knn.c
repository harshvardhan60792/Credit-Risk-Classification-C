#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 1. INCLUDE THE DATASET DIRECTLY
// This pulls in the NUM_ROWS and the raw_data array from your generated header file.
#include "dataset.h" 

#define NUM_FEATURES 20

// Structure to hold a single row of data
typedef struct {
    double features[NUM_FEATURES];
    int label;
    double distance;
} DataPoint;

// 2. Mathematical Model: Euclidean Distance
double calculateDistance(double p1[], double p2[]) {
    double sum = 0.0;
    for (int i = 0; i < NUM_FEATURES; ++i) {
        sum += pow(p1[i] - p2[i], 2);
    }
    return sqrt(sum);
}

// Comparator function for qsort (sorting distances)
int compareDistance(const void *a, const void *b) {
    DataPoint *p1 = (DataPoint *)a;
    DataPoint *p2 = (DataPoint *)b;
    if (p1->distance < p2->distance) return -1;
    if (p1->distance > p2->distance) return 1;
    return 0;
}

// 3. Feature Scaling: Min-Max Normalization
void normalizeData(DataPoint dataset[], int size) {
    if (size == 0) return;

    for (int i = 0; i < NUM_FEATURES; ++i) {
        double minVal = dataset[0].features[i];
        double maxVal = dataset[0].features[i];

        // Find min and max for each feature column
        for (int j = 0; j < size; ++j) {
            if (dataset[j].features[i] < minVal) minVal = dataset[j].features[i];
            if (dataset[j].features[i] > maxVal) maxVal = dataset[j].features[i];
        }

        // Apply Min-Max formula
        for (int j = 0; j < size; ++j) {
            if (maxVal - minVal == 0.0) {
                dataset[j].features[i] = 0.0;
            } else {
                dataset[j].features[i] = (dataset[j].features[i] - minVal) / (maxVal - minVal);
            }
        }
    }
}

// 4. The KNN Algorithm
int predictKNN(DataPoint trainData[], int trainSize, DataPoint testPoint, int k) {
    // Calculate distance from test point to all training points
    for (int i = 0; i < trainSize; ++i) {
        trainData[i].distance = calculateDistance(trainData[i].features, testPoint.features);
    }

    // Sort the training data by distance (ascending)
    qsort(trainData, trainSize, sizeof(DataPoint), compareDistance);

    // Count frequencies of labels among the top K neighbors
    // (Labels in German Credit are 1 for Good, 2 for Bad)
    int count1 = 0;
    int count2 = 0;

    for (int i = 0; i < k; ++i) {
        if (trainData[i].label == 1) count1++;
        else if (trainData[i].label == 2) count2++;
    }

    // Return the majority vote
    return (count1 > count2) ? 1 : 2;
}

int main() {
    // dataset array to hold the formatted data
    DataPoint dataset[NUM_ROWS];
    
    // 5. LOAD DATA FROM THE C ARRAY
    for (int i = 0; i < NUM_ROWS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            dataset[i].features[j] = raw_data[i][j];
        }
        // FIXED: The 21st column (index 20) is the target label
        dataset[i].label = (int)raw_data[i][20]; 
    }

    printf("Successfully loaded %d rows directly from memory!\n", NUM_ROWS);

    // Normalize the data (Critical for KNN)
    normalizeData(dataset, NUM_ROWS);

    // Simple Train/Test Split (80% Train, 20% Test)
    int trainSize = (int)(NUM_ROWS * 0.8);
    int testSize = NUM_ROWS - trainSize;

    // Use Variable Length Arrays (VLA) or fixed size if your compiler is strict. 
    // Since NUM_ROWS is usually 1000, these arrays are small enough for the stack.
    DataPoint trainData[trainSize];
    DataPoint testData[testSize];

    // Copy data into train and test arrays
    for (int i = 0; i < trainSize; i++) {
        trainData[i] = dataset[i];
    }
    for (int i = 0; i < testSize; i++) {
        testData[i] = dataset[trainSize + i];
    }

    // Run KNN
    int k = 5;
    int correctPredictions = 0;

    for (int i = 0; i < testSize; ++i) {
        int prediction = predictKNN(trainData, trainSize, testData[i], k);
        if (prediction == testData[i].label) {
            correctPredictions++;
        }
    }

    double accuracy = ((double)correctPredictions / testSize) * 100.0;
    
    printf("--- Pure C KNN Classification Results ---\n");
    printf("Total Valid Data Points Loaded: %d\n", NUM_ROWS);
    printf("Training Size: %d\n", trainSize);
    printf("Testing Size: %d\n", testSize);
    printf("K Value used: %d\n", k);
    printf("Accuracy: %.2f%%\n", accuracy);

    return 0;
}