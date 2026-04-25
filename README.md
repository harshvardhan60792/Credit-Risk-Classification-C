# Mathematical Modelling of Credit Risk
A comparative analysis of Machine Learning algorithms implemented from scratch in **C**.

## 📌 Project Overview
This project was developed for the **Mathematical Modelling–Based Experiential Learning** assignment. The goal is to predict credit risk (Good vs. Bad Credit) using the **Statlog (German Credit Data)** dataset. 

Unlike standard ML projects, this implementation avoids all high-level libraries (like Scikit-Learn), relying strictly on the mathematical foundations of each algorithm implemented in pure C.

## 🧠 Algorithms Implemented
We have modeled the problem using three distinct mathematical approaches to compare their performance:

1.  **K-Nearest Neighbors (KNN):** A geometric, distance-based model using Euclidean distance and Min-Max feature scaling.
2.  **Gaussian Naive Bayes (GNB):** A probabilistic model using Bayes' Theorem and the Gaussian Probability Density Function.
3.  **Logistic Regression:** An optimization-based model using the Sigmoid activation function and Gradient Descent for weight updates.

## 📊 Mathematical Foundation
The models utilize the following core concepts to secure the "Mathematical Model" grading criteria:

* **Euclidean Distance:** $d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$
* **Sigmoid Activation:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
* **Gaussian PDF:** $P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$

## 📁 Project Structure
* `knn.c`: Implementation of the K-Nearest Neighbors algorithm.
* `nb.c`: Implementation of the Gaussian Naive Bayes algorithm.
* `lr.c`: Implementation of the Logistic Regression algorithm.
* `dataset.h`: The pre-processed German Credit dataset (1,000 rows, 20 features) converted into a C-readable header file for zero-dependency execution.

## 🚀 How to Run
Ensure you have a C compiler (like `gcc`) installed. Open your terminal in the project folder and run:

### K-Nearest Neighbors
```bash
gcc knn.c -o knn.exe
./knn.exe
