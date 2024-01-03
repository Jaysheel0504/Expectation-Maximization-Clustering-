#include <bits/stdc++.h>
using namespace std;
#include <math.h>

typedef struct inputdata {
	double * data;
	int class_id;
	double * normal_probabilities;
} INPUTDATA;

class GMM_model{
public:
	GMM_model(void){
		maxNumberOfIterations = 10000;

		numExamplesEachClass = 0;
		probEachClass = 0;

		sumFeatureEachClass = 0;
		sumFeatureVariationEachClass = 0;
		meanFeatureEachClass = 0;
		covarianceFeatureEachClass = 0;
	}

	~GMM_model(void){
		delete[] numExamplesEachClass;
		delete[] probEachClass;

		for(int a=0; a<numberOfFeatures; a++) {
			delete[] sumFeatureEachClass[a];
			delete[] sumFeatureVariationEachClass[a];
			delete[] meanFeatureEachClass[a];
			delete[] covarianceFeatureEachClass[a];
		}
		delete[] sumFeatureEachClass;
		delete[] sumFeatureVariationEachClass;
		delete[] meanFeatureEachClass;
		delete[] covarianceFeatureEachClass;
	}

	void init(int numOfClasses, int numOfExamples, int numOfFeatures, INPUTDATA **inpDataList)
	{
		numberOfClasses = numOfClasses;
		numberOfExamples = numOfExamples;
		numberOfFeatures = numOfFeatures;
		inputDataList = inpDataList;
		numExamplesEachClass = new int[numberOfClasses];
		probEachClass = new double[numberOfClasses];
		sumFeatureEachClass = new double*[numberOfClasses];
		sumFeatureVariationEachClass = new double*[numberOfClasses];
		meanFeatureEachClass = new double*[numberOfClasses];
		covarianceFeatureEachClass = new double*[numberOfClasses];

		for (int i = 0; i < numberOfClasses; i++) {
			numExamplesEachClass[i] = 0;
			probEachClass[i] = 0;
			sumFeatureEachClass[i] = new double[numberOfFeatures];
			sumFeatureVariationEachClass[i] = new double[numberOfFeatures];
			meanFeatureEachClass[i] = new double[numberOfFeatures];
			covarianceFeatureEachClass[i] = new double[numberOfFeatures];

			for (int j = 0; j < numberOfFeatures; j++) {
				sumFeatureEachClass[i][j] = 0;
				sumFeatureVariationEachClass[i][j] = 0;
				meanFeatureEachClass[i][j] = 0;
				covarianceFeatureEachClass[i][j] = 0;
			}
		}
	}



	void train(){
		iStep();
		for(int i=0; i<maxNumberOfIterations; i++) {
			mStep();
			if(!has_class_changed){
				cout << "Training complete!!" << endl;
				cout << endl;
				break;
			}
			eStep(i+1);
		}
		printParameters();
	}

private:

	int maxNumberOfIterations;

	int numberOfClasses;						
	int numberOfFeatures;						
	int numberOfExamples;						
	INPUTDATA ** inputDataList;					

	bool has_class_changed;

	int * numExamplesEachClass;						
	double ** sumFeatureEachClass;				
	double ** sumFeatureVariationEachClass;		
	double ** meanFeatureEachClass;			
	double ** covarianceFeatureEachClass;	

	double * probEachClass;	

	void iStep(){
		for(int i=0; i<numberOfExamples; i++) {
			// assigning initial cluster randomly
			inputDataList[i]->class_id = i % numberOfClasses;
		}
		eStep(0);
	}

	void eStep(int iteration_number)
	{
		// reset
		for (int i = 0; i < numberOfClasses; i++) {
		numExamplesEachClass[i] = 0;
			for (int j = 0; j < numberOfFeatures; j++) {
				sumFeatureEachClass[i][j] = 0;
				sumFeatureVariationEachClass[i][j] = 0;
			}
		}

		// Find number of examples in each class and calculate sum of features for each class
		for (int i = 0; i < numberOfExamples; i++) {
			numExamplesEachClass[inputDataList[i]->class_id]++;
			for (int j = 0; j < numberOfFeatures; j++) {
				sumFeatureEachClass[inputDataList[i]->class_id][j] += inputDataList[i]->data[j]; 
			}
		}

		// Populate prob of each class and Find mean for each class
		for (int i = 0; i < numberOfClasses; i++) {
			probEachClass[i] = (double)((double)numExamplesEachClass[i] / (double)numberOfExamples);
			for (int j = 0; j < numberOfFeatures; j++) {
				meanFeatureEachClass[i][j] = (double)((double)sumFeatureEachClass[i][j] / (double)numExamplesEachClass[i]);
			}
		}

		// Calculate variance for each class 
		for (int i = 0; i < numberOfExamples; i++) {
			for (int j = 0; j < numberOfFeatures; j++) {
				sumFeatureVariationEachClass[inputDataList[i]->class_id][j] = sumFeatureVariationEachClass[inputDataList[i]->class_id][j] + 
					(inputDataList[i]->data[j] - meanFeatureEachClass[inputDataList[i]->class_id][j]) * 
					(inputDataList[i]->data[j] - meanFeatureEachClass[inputDataList[i]->class_id][j]);
			} 
		}

		// Calculate covariance for each class 
		for (int i = 0; i < numberOfClasses; i++) {
			for (int j = 0; j < numberOfFeatures; j++) {
				covarianceFeatureEachClass[i][j] = sumFeatureVariationEachClass[i][j] / (double)(numExamplesEachClass[i] - 1);
			}
		}
		
		// Printing the current iteration number
		cout << "Iteration number " << iteration_number << endl;

		// Printing number of data points classified into each class
		for (int a = 0; a < numberOfClasses; a++) {
			cout << "Cluster for class#" << a << " --> number of data points: " << numExamplesEachClass[a];

			// cout << "data[";
			// for (int z = 0; z < numberOfExamples; z++) {
			// 	if (inputDataList[z]->class_id == a)
			// 		cout << z << " ";
			// }
			// cout << "]" << endl;

			// cout << "mean:  ";
			// for (int b = 0; b < numberOfFeatures; b++) {
			// 	cout << "feat[" << b << "]:" << sumFeatureEachClass[a][b] << "/" << numExamplesEachClass[a] << ", ";
			// }
			// cout << endl;
			// cout << "var:   ";
			// for (int b = 0; b < numberOfFeatures; b++) {
			// 	cout << "feat[" << b << "]:" << sumFeatureVariationEachClass[a][b] << "/(" << numExamplesEachClass[a] - 1 << "), ";
			// }
			cout << endl;
		}
		cout << endl;
	}

	void mStep()
	{
		double * intermediateProbabilityEachClass = new double[numberOfClasses];
		double probGaussian = 1, probSum = 0;
		has_class_changed = false;

		for (int i = 0; i < numberOfExamples; i++) {
			probSum = 0;
			for (int j = 0; j < numberOfClasses; j++) {
				intermediateProbabilityEachClass[j] = 1;
				probGaussian = 1;
				for (int k = 0; k < numberOfFeatures; k++) {
					probGaussian = getGaussianProbability(meanFeatureEachClass[j][k], covarianceFeatureEachClass[j][k], inputDataList[i]->data[k]);
					intermediateProbabilityEachClass[j] *= probGaussian;
				}
				probSum += intermediateProbabilityEachClass[j];
			}

			double temp = 0;
			int max_class_id = 0;
			for (int j = 0; j < numberOfClasses; j++) {
				inputDataList[i]->normal_probabilities[j] = intermediateProbabilityEachClass[j] / probSum;
				if (temp < intermediateProbabilityEachClass[j]) {
					max_class_id = j;
					temp = intermediateProbabilityEachClass[j];
				}
			}
			if (inputDataList[i]->class_id != max_class_id) has_class_changed = true;
			inputDataList[i]->class_id = max_class_id;
		}

		delete[] intermediateProbabilityEachClass;
	}

	// Function to find gaussian probability for a value given the mean and variance
	inline double getGaussianProbability(double mean, double var, double value){
		double gauss = 1;
		const double pi = 3.14159265358979323846;
		gauss = (1 / sqrt(2 * pi * var)) * (exp((-1 * (value - mean) * (value - mean)) / (2*var)));
		return gauss;
	}

	// Printing the parameters(mean and variance) for each feature, for each class
	void printParameters(){
		for (int i = 0; i < numberOfClasses; i++) {
			cout << "---------------------------" << " CLASS " << i << " ---------------------------" << endl;
			cout << endl;
			cout << "Probability of (c" << i << ") = " << probEachClass[i] << endl;
			for (int j = 0; j < numberOfFeatures; j++) {
				cout << "Mean for feature #" << i+1 << " = " << meanFeatureEachClass[i][j] << ",\t";
				cout << "Variance for feature #" << i+1 << " = " << covarianceFeatureEachClass[i][j] << endl;
			}
			cout << endl;
		}
		cout << endl;
	}				
};