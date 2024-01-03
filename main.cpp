#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include "GMM.cpp"
using namespace std;

void readCSV(string filename, INPUTDATA** inputData, int* ground_truth);
void printResults(INPUTDATA** inputData);
void printAccuracyStats(INPUTDATA** inputData, int* ground_truth);
void deallocateMemory(INPUTDATA** inputData);

#define NUMBER_OF_EXAMPLES		178
#define NUMBER_OF_CLASSES		3
#define NUMBER_OF_FEATURES		13
string CLASSES[] = {"Red", "White", "Rose"};

int main()
{
    int* ground_truth = new int[NUMBER_OF_EXAMPLES];
	INPUTDATA ** inputData;
	inputData = new INPUTDATA*[NUMBER_OF_EXAMPLES];
	readCSV("wine.csv", inputData, ground_truth);

	// training of the model
	GMM_model * gmm_model = new GMM_model();
	gmm_model->init(NUMBER_OF_CLASSES, NUMBER_OF_EXAMPLES, NUMBER_OF_FEATURES, inputData);
	gmm_model->train();

	// printing results
	printResults(inputData);

    printAccuracyStats(inputData,ground_truth);

	// deallocating memory	
	deallocateMemory(inputData);
}

// Function to read the csv file and store training data in array
void readCSV(string filename, INPUTDATA** inputData, int* ground_truth) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    string line;
	int index = 0;

    while (getline(file, line)) {
        istringstream ss(line);
        string token;

        INPUTDATA* dataItem = new INPUTDATA;
        dataItem->normal_probabilities = new double[NUMBER_OF_CLASSES];
        dataItem->data = new double[NUMBER_OF_FEATURES];
        int idx = 0;

        while (getline(ss, token, ',')) {
            if(idx==0) ground_truth[index] = stoi(token);
            else dataItem->data[idx-1] = stod(token);
            idx++;
        }
        
        dataItem->class_id = -1;
        inputData[index] = dataItem;
		index++;
    }
    file.close();
}

// Function to print results
void printResults(INPUTDATA** inputData) {
	int numExamples = NUMBER_OF_EXAMPLES;
	cout << "Final classification for each training example :-" << endl;
	cout << endl;
    for (int a = 0; a < numExamples; a++) {
        cout << "Example number " << a << " --> " << CLASSES[inputData[a]->class_id] << "\t:";
        for (int b = 0; b < NUMBER_OF_CLASSES; b++) {
            cout << " P(" << b << ") = " << inputData[a]->normal_probabilities[b] << "     ";
        }
        cout << endl;
    }
    cout << endl;
    cout << "-----------------------------------------------------" << endl << endl;
}

// Function to deallocate memory
void deallocateMemory(INPUTDATA** inputData) {
	int numExamples = NUMBER_OF_EXAMPLES;
    for (int i = 0; i < numExamples; i++) {
        delete[] inputData[i]->data;
        delete[] inputData[i]->normal_probabilities;
        delete inputData[i];
    }
    delete[] inputData;
}

void printAccuracyStats(INPUTDATA** inputData, int* ground_truth) {
    int correct_count = 0;
    int confusionMatrix[NUMBER_OF_CLASSES][NUMBER_OF_CLASSES] = {0};

    // Populate the confusion matrix and count correct predictions
    for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
        int predictedClass = inputData[i]->class_id + 1;
        int trueClass = ground_truth[i];
        confusionMatrix[trueClass - 1][predictedClass - 1]++;
        if (predictedClass == trueClass) {
            correct_count++;
        }
    }

    // Print the accuracy
    double accuracy = (static_cast<double>(correct_count) / NUMBER_OF_EXAMPLES) * 100.0;
    cout << "Accuracy: " << accuracy << " %" << endl;
    cout <<endl;

    // Print the confusion matrix
    cout << "Confusion Matrix:" << endl;
    cout << "(Predictions as rows and Ground truth as columns)"<<endl;
    cout <<endl;
    cout << "\tRed \tWhite \tRose"<<endl;
    for (int trueClass = 0; trueClass < NUMBER_OF_CLASSES; trueClass++) {
        cout<<CLASSES[trueClass]<<" \t";
        for (int predictedClass = 0; predictedClass < NUMBER_OF_CLASSES; predictedClass++) {
            cout << confusionMatrix[trueClass][predictedClass] << "\t";
        }
        cout << endl;
    }

    cout<<endl;
    // Calculate and print recall and precision for each class
    for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
        int truePositives = confusionMatrix[i][i];
        int falsePositives = 0;
        int falseNegatives = 0;

        for (int j = 0; j < NUMBER_OF_CLASSES; j++) {
            if (i != j) {
                falsePositives += confusionMatrix[j][i];
                falseNegatives += confusionMatrix[i][j];
            }
        }

        double recall = (truePositives + DBL_EPSILON) / (truePositives + falseNegatives + DBL_EPSILON);
        double precision = (truePositives + DBL_EPSILON) / (truePositives + falsePositives + DBL_EPSILON);
        cout << "Class - " << (i+1) << "(" << CLASSES[i] << ")" <<endl;
        cout << "   Recall: " << recall << endl;
        cout << "   Precision: " << precision << endl;
        cout <<endl;
    }
}
