#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 1. Include the generated dataset
#include "dataset.h" 

#define NUM_FEATURES 20 
#define NUM_CLASSES 2   // Labels are 1 (Good) and 2 (Bad)
#define PI 3.14159265358979323846

// Structure to hold a single row of data
typedef struct {
    double features[NUM_FEATURES];
    int label;
} DataPoint;

// Arrays to store the mathematical model's parameters
// Index 0 = Class 1 (Good), Index 1 = Class 2 (Bad)
double means[NUM_CLASSES][NUM_FEATURES];
double variances[NUM_CLASSES][NUM_FEATURES];
double priors[NUM_CLASSES];

// 2. Train the Model (Calculate Means and Variances)
void trainNaiveBayes(DataPoint trainData[], int trainSize) {
    int classCounts[NUM_CLASSES] = {0};

    // Initialize arrays to zero
    for(int c = 0; c < NUM_CLASSES; c++) {
        for(int f = 0; f < NUM_FEATURES; f++) {
            means[c][f] = 0.0;
            variances[c][f] = 0.0;
        }
    }

    // Step A: Calculate the Sums and Class Counts
    for(int i = 0; i < trainSize; i++) {
        int c = trainData[i].label - 1; // Map label 1->0, label 2->1
        classCounts[c]++;
        for(int f = 0; f < NUM_FEATURES; f++) {
            means[c][f] += trainData[i].features[f];
        }
    }

    // Step B: Calculate the Priors and Means
    for(int c = 0; c < NUM_CLASSES; c++) {
        priors[c] = (double)classCounts[c] / trainSize;
        for(int f = 0; f < NUM_FEATURES; f++) {
            if (classCounts[c] > 0) {
                means[c][f] /= classCounts[c];
            }
        }
    }

    // Step C: Calculate the Variances
    for(int i = 0; i < trainSize; i++) {
        int c = trainData[i].label - 1;
        for(int f = 0; f < NUM_FEATURES; f++) {
            double diff = trainData[i].features[f] - means[c][f];
            variances[c][f] += (diff * diff);
        }
    }

    for(int c = 0; c < NUM_CLASSES; c++) {
        for(int f = 0; f < NUM_FEATURES; f++) {
            if (classCounts[c] > 1) {
                // Divide by N-1 for sample variance
                variances[c][f] /= (classCounts[c] - 1); 
            }
            // Add a tiny epsilon to prevent division by zero in the PDF
            if (variances[c][f] < 1e-9) variances[c][f] = 1e-9; 
        }
    }
}

// 3. Gaussian Probability Density Function (Log version to prevent underflow)
double calculateLogLikelihood(double x, double mean, double variance) {
    double log_coeff = -0.5 * log(2 * PI * variance);
    double exponent = -(x - mean) * (x - mean) / (2 * variance);
    return log_coeff + exponent;
}

// 4. Make a Prediction for a single test point
int predictNaiveBayes(DataPoint testPoint) {
    int bestClass = -1;
    double highestProb = -1e9; // Start with a very small number

    for(int c = 0; c < NUM_CLASSES; c++) {
        // Start with the log of the prior probability
        double logProb = log(priors[c]); 

        // Add the log-likelihood of every feature
        for(int f = 0; f < NUM_FEATURES; f++) {
            logProb += calculateLogLikelihood(testPoint.features[f], means[c][f], variances[c][f]);
        }

        // Keep track of the class with the highest probability
        if (logProb > highestProb) {
            highestProb = logProb;
            bestClass = c + 1; // Map index 0->1, index 1->2
        }
    }

    return bestClass;
}

int main() {
    DataPoint dataset[NUM_ROWS];
    
    // 5. Load Data directly from dataset.h memory
    for (int i = 0; i < NUM_ROWS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            dataset[i].features[j] = raw_data[i][j];
        }
        dataset[i].label = (int)raw_data[i][20]; // 21st column is the label
    }

    // NOTE: We intentionally DO NOT normalize the data here!
    // Gaussian Naive Bayes handles different scales automatically via variance.

    // Simple Train/Test Split (80% Train, 20% Test)
    int trainSize = (int)(NUM_ROWS * 0.8);
    int testSize = NUM_ROWS - trainSize;

    DataPoint trainData[trainSize];
    DataPoint testData[testSize];

    for (int i = 0; i < trainSize; i++) trainData[i] = dataset[i];
    for (int i = 0; i < testSize; i++) testData[i] = dataset[trainSize + i];

    // Train the Model
    trainNaiveBayes(trainData, trainSize);

    // Test the Model
    int correctPredictions = 0;
    for (int i = 0; i < testSize; ++i) {
        int prediction = predictNaiveBayes(testData[i]);
        if (prediction == testData[i].label) {
            correctPredictions++;
        }
    }

    double accuracy = ((double)correctPredictions / testSize) * 100.0;
    
    printf("--- Pure C Gaussian Naive Bayes Results ---\n");
    printf("Total Valid Data Points Loaded: %d\n", NUM_ROWS);
    printf("Training Size: %d\n", trainSize);
    printf("Testing Size: %d\n", testSize);
    printf("Accuracy: %.2f%%\n", accuracy);

    return 0;
}