/*******************************************************************
* Neural Network Training Example
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
*********************************************************************/

//standard libraries
#include <iostream>
#include <ctime>
#include <string>
#include "letter_recognition_training_example.h"

//custom includes
#include "neuralNetwork.h"
#include "neuralNetworkTrainer.h"
#include "cann.h"

//use standard namespace
using namespace std;

//create neural network
uint layerSizesUint[3] = {16, 10, 3};
long layerSizesLong[3] = {16, 10, 3};

neuralNetwork nn(3, layerSizesUint);
Cann *cann = CannCreateFromArray(layerSizesLong, 3);


void callbackFunction(neuralNetwork *network, NNType *pattern, NNType *result){

    vector<NNType>w = network->weightsVector();
    long wC = w.size();
    
    printf("!callbackFunction: %llu (weights: %lu)\n", network->feedForwardCount(), wC);

    CannType *weightsArray = w.data();
    CannLoadWeights(cann, weightsArray, wC);
    CannType *cannResult = CannFeedForward(cann, pattern);
    
    uint outputCount = 0;
    NNType* ol = network->getOutputLayer(&outputCount);
    
    for ( int i = 0; i < outputCount; i++) {
    
        NNType nnOut = result[i];
        CannType cannOut = cannResult[i];
        
        if (nnOut != cannOut) {
            printf(" !!!! error nn out[%i] = %f, cann out[%i] = %f, diff = %f\n", i, nnOut, i, cannOut, cannOut - nnOut);
        }
    }
}

void example_nn(const char *inputFile, const char *outputFile, const char *logFile)
{		
	//seed random number generator
	srand( (unsigned int) time(0) );
	
	//create data set reader and load data file
	dataReader d;
	d.loadDataFile(inputFile,16,3);
	d.setCreationApproach( STATIC, 10 );	

    if (sizeof(NNType) != sizeof(CannType)) {
        printf("sizeof(NNType) != sizeof(CannType) !!!! = STOP");
        return;
        
    }
    
    
    nn.callback = &callbackFunction;
    
	//create neural network trainer
	neuralNetworkTrainer nT( &nn );
	nT.setTrainingParameters(0.001, 0.9, false);
	nT.setStoppingConditions(300, 95);
    
    if (logFile != NULL) {
    
        nT.enableLogging(logFile, 5);
    }
	
	
	//train neural network on data sets
	for (int i=0; i < d.getNumTrainingSets(); i++ )
	{
		nT.trainNetwork( d.getTrainingDataSet() );
	}	

	//save the weights
	nn.saveWeights(outputFile);
    
    printf("\n Average feed forward      time: %fsec (run count: %llu)", nn.averageFeedForwardTime(), nn.feedForwardCount());
    printf("\n Average input layer load  time: %fsec (run count: %llu)", nn.averageFeedForwardTime(), nn.feedForwardCount());

}
