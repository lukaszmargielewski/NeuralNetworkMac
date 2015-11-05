//standard includes
#include <iostream>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetworkTrainer.h"

using namespace std;

/*******************************************************************
* constructor
********************************************************************/
neuralNetworkTrainer::neuralNetworkTrainer( neuralNetwork *nn )	:	NN(nn),
																	epoch(0),
																	learningRate(LEARNING_RATE),
																	momentum(MOMENTUM),
																	maxEpochs(MAX_EPOCHS),
																	desiredAccuracy(DESIRED_ACCURACY),																	
																	useBatch(false),
																	trainingSetAccuracy(0),
																	validationSetAccuracy(0),
																	generalizationSetAccuracy(0),
																	trainingSetMSE(0),
																	validationSetMSE(0),
																	generalizationSetMSE(0)																	
{
    
    outputLayer = NN->getOutputLayer(&outputLayerCount);
    
	//create delta lists
	//--------------------------------------------------------------------------------------------------------
	deltaInputHidden = new double*[NN->_neuronsPerLayer[0] + 1] ;
	for ( int i=0; i <= NN->_neuronsPerLayer[0]; i++ ) 
	{
		deltaInputHidden[i] = new double[NN->_neuronsPerLayer[1]];
		for ( int j=0; j < NN->_neuronsPerLayer[1]; j++ ) deltaInputHidden[i][j] = 0;		
	}

	deltaHiddenOutput = new double*[NN->_neuronsPerLayer[1] + 1] ;
	for ( int i=0; i <= NN->_neuronsPerLayer[1]; i++ ) 
	{
		deltaHiddenOutput[i] = new double[NN->_neuronsPerLayer[2]];			
		for ( int j=0; j < NN->_neuronsPerLayer[2]; j++ ) deltaHiddenOutput[i][j] = 0;		
	}

	//create error gradient storage
	//--------------------------------------------------------------------------------------------------------
	hiddenErrorGradients = new double[NN->_neuronsPerLayer[1] + 1];
	outputErrorGradients = new double[NN->_neuronsPerLayer[2] + 1];
    initializeWeights();
    
}


/*******************************************************************
* Set training parameters
********************************************************************/
void neuralNetworkTrainer::setTrainingParameters( double lR, double m, bool batch )
{
	learningRate = lR;
	momentum = m;
	useBatch = batch;
}
/*******************************************************************
* Set stopping parameters
********************************************************************/
void neuralNetworkTrainer::setStoppingConditions( int mEpochs, double dAccuracy )
{
	maxEpochs = mEpochs;
	desiredAccuracy = dAccuracy;	
}




/*******************************************************************
* Train the NN using gradient descent
********************************************************************/
void neuralNetworkTrainer::trainNetwork( trainingDataSet* tSet )
{
	cout	<< endl << " Neural Network Training Starting: " << endl
			<< "==========================================================================" << endl
			<< " LR: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << endl
			<< " " << NN->_neuronsPerLayer[0] << " Input Neurons, " << NN->_neuronsPerLayer[1] << " Hidden Neurons, " << NN->_neuronsPerLayer[2] << " Output Neurons" << endl
			<< "==========================================================================" << endl << endl;

	//reset epoch and log counters
	epoch = 0;
	lastEpochLogged = -logResolution;
		
	//train network using training dataset for training and generalization dataset for testing
	//--------------------------------------------------------------------------------------------------------
	while (	( trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy ) && epoch < maxEpochs )				
	{			
		//store previous accuracy
		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		runTrainingEpoch( tSet->trainingSet );

		//get generalization set accuracy and MSE
		generalizationSetAccuracy = getSetAccuracy( tSet->generalizationSet );
		generalizationSetMSE = getSetMSE( tSet->generalizationSet );

		//Log Training results
		if ( loggingEnabled && logFile.is_open() && ( epoch - lastEpochLogged == logResolution ) ) 
		{
			logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl;
			lastEpochLogged = epoch;
		}
		
		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		if ( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) ) 
		{	
			cout << "Epoch :" << epoch;
			cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
			cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;				
		}
		
		//once training set is complete increment epoch
		epoch++;

	}//end while

	//get validation set accuracy and MSE
	validationSetAccuracy = getSetAccuracy(tSet->validationSet);
	validationSetMSE = getSetMSE(tSet->validationSet);

	//log end
	logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl << endl;
	logFile << "Training Complete!!! - > Elapsed Epochs: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << " Validation Set MSE: " << validationSetMSE << endl;
			
	//out validation accuracy and MSE
	cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
	cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
	cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
}


/*******************************************************************
* Run a single training epoch
********************************************************************/
void neuralNetworkTrainer::runTrainingEpoch( vector<dataEntry*> trainingSet )
{
	//incorrect patterns
	double incorrectPatterns = 0;
	double mse = 0;
		
	//for every training pattern
	for ( int tp = 0; tp < (int) trainingSet.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		NN->feedForward( trainingSet[tp]->pattern );
		backpropagate( trainingSet[tp]->target );	

		//pattern correct flag
		bool patternCorrect = true;

		//check all outputs from neural network against desired values
		for ( int k = 0; k < outputLayerCount; k++ )
		{					
			//pattern incorrect if desired and output differ
			if (clampOutput( outputLayer[k] ) != trainingSet[tp]->target[k] ) patternCorrect = false;
			
			//calculate MSE
			mse += pow(( outputLayer[k] - trainingSet[tp]->target[k] ), 2);
		}
		
		//if pattern is incorrect add to incorrect count
		if ( !patternCorrect ) incorrectPatterns++;	
		
	}//end for

    
	//if using batch learning - update the weights
	if ( useBatch )
        updateWeights();
	
	//update training accuracy and MSE
	trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
	trainingSetMSE = mse / ( NN->_neuronsPerLayer[2] * trainingSet.size() );
}




/*******************************************************************
* Propagate errors back through NN and calculate delta values
********************************************************************/
void neuralNetworkTrainer::backpropagate( double* desiredOutputs )
{		
	//modify deltas between hidden and output layers
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < NN->_neuronsPerLayer[2]; k++)
	{
		//get error gradient for every output node
		outputErrorGradients[k] = getOutputErrorGradient( desiredOutputs[k], NN->neurons[2][k] );
		
		//for all nodes in hidden layer and bias neuron
		for (int j = 0; j <= NN->_neuronsPerLayer[1]; j++) 
		{
            double ddd = learningRate * NN->neurons[1][j] * outputErrorGradients[k];
            
			//calculate change in weight
            if ( !useBatch ){
            
                deltaHiddenOutput[j][k] = ddd  + momentum * deltaHiddenOutput[j][k];
            }
            else {
            
                deltaHiddenOutput[j][k] += ddd;
            }
		}
	}

	//modify deltas between input and hidden layers
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < NN->_neuronsPerLayer[1]; j++)
	{
		//get error gradient for every hidden node
		hiddenErrorGradients[j] = getHiddenErrorGradient( j );

		//for all nodes in input layer and bias neuron
		for (int i = 0; i <= NN->_neuronsPerLayer[0]; i++)
		{
            
            double ddd = learningRate * NN->neurons[0][i] * hiddenErrorGradients[j];
            
			//calculate change in weight 
			if ( !useBatch )
                deltaInputHidden[i][j] = ddd + momentum * deltaInputHidden[i][j];
			else
                deltaInputHidden[i][j] += ddd;
		}
	}
	
	//if using stochastic learning update the weights immediately
	if ( !useBatch ) updateWeights();
}


/*******************************************************************
 * calculate output error gradient
 ********************************************************************/
inline double neuralNetworkTrainer::getOutputErrorGradient( double desiredValue, double outputValue)
{
    //return error gradient
    return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
}

/*******************************************************************
 * calculate input error gradient
 ********************************************************************/
double neuralNetworkTrainer::getHiddenErrorGradient( int j )
{
    //get sum of hidden->output weights * output error gradients
    double weightedSum = 0;
    
    for( int k = 0; k < NN->_neuronsPerLayer[2]; k++ )
        weightedSum += NN->weights[1][j][k] * outputErrorGradients[k];
    
    //return error gradient
    return NN->neurons[1][j] * ( 1 - NN->neurons[1][j] ) * weightedSum;
}


/*******************************************************************
* Update weights using delta values
********************************************************************/
void neuralNetworkTrainer::updateWeights()
{
	//input -> hidden weights
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= NN->_neuronsPerLayer[0]; i++)
	{
		for (int j = 0; j < NN->_neuronsPerLayer[1]; j++) 
		{
			//update weight
			NN->weights[0][i][j] += deltaInputHidden[i][j];	
			
			//clear delta only if using batch (previous delta is needed for momentum
			if (useBatch)
                deltaInputHidden[i][j] = 0;
		}
	}
	
	//hidden -> output weights
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j <= NN->_neuronsPerLayer[1]; j++)
	{
		for (int k = 0; k < NN->_neuronsPerLayer[2]; k++) 
		{					
			//update weight
			NN->weights[1][j][k] += deltaHiddenOutput[j][k];
			
			//clear delta only if using batch (previous delta is needed for momentum)
			if (useBatch)
                deltaHiddenOutput[j][k] = 0;
		}
	}
}


/*******************************************************************
 * Return the NN accuracy on the set
 ********************************************************************/
double neuralNetworkTrainer::getSetAccuracy( std::vector<dataEntry*>& set )
{
    double incorrectResults = 0;
    
    //for every training input array
    for ( int tp = 0; tp < (int) set.size(); tp++)
    {
        //feed inputs through network and backpropagate errors
        NN->feedForward( set[tp]->pattern );
        
        //correct pattern flag
        bool correctResult = true;
        
        //check all outputs against desired output values
        for ( int k = 0; k < outputLayerCount; k++ )
        {
            //set flag to false if desired and output differ
            if ( clampOutput(outputLayer[k]) != set[tp]->target[k] ) correctResult = false;
        }
        
        //inc training error for a incorrect result
        if ( !correctResult ) incorrectResults++;
        
    }//end for
    
    //calculate error and return as percentage
    return 100 - (incorrectResults/set.size() * 100);
}


/*******************************************************************
 * Return the NN mean squared error on the set
 ********************************************************************/
double neuralNetworkTrainer::getSetMSE( std::vector<dataEntry*>& set )
{
    double mse = 0;
    
    //for every training input array
    for ( int tp = 0; tp < (int) set.size(); tp++)
    {
        //feed inputs through network and backpropagate errors
        NN->feedForward( set[tp]->pattern );
        
        //check all outputs against desired output values
        for ( int k = 0; k < outputLayerCount; k++ )
        {
            //sum all the MSEs together
            mse += pow((outputLayer[k] - set[tp]->target[k]), 2);
        }
        
    }//end for
    
    //calculate error and return as percentage
    return mse/(outputLayerCount * set.size());
}


/*******************************************************************
 * Initialize Neuron Weights
 ********************************************************************/
void neuralNetworkTrainer::initializeWeights()
{
    
    for (int iL0 = 0; iL0 < NN->_layerCount - 1; iL0++) {
        int iL1 = iL0 + 1;

        //set range
        double r = 1/sqrt( (double) NN->_neuronsPerLayer[iL0]);
        
        //set weights between input and hidden
        //--------------------------------------------------------------------------------------------------------
        for(int i = 0; i <= NN->_neuronsPerLayer[iL0]; i++)
        {
            for(int j = 0; j < NN->_neuronsPerLayer[iL1]; j++)
            {
                //set weights to random values
                NN->weights[iL0][i][j] = ( ( (double)(rand()%100)+1)/100  * 2 * r ) - r;
            }
        }
    }
}


/*******************************************************************
 * Output Clamping
 ********************************************************************/
inline int neuralNetworkTrainer::clampOutput( double x )
{
    if ( x < 0.1 ) return 0;
    else if ( x > 0.9 ) return 1;
    else return -1;
}

/*******************************************************************
 * Enable training logging
 ********************************************************************/
void neuralNetworkTrainer::enableLogging(const char* filename, int resolution = 1)
{
    //create log file
    if ( ! logFile.is_open() )
    {
        logFile.open(filename, ios::out);
        
        if ( logFile.is_open() )
        {
            //write log file header
            logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;
            
            //enable logging
            loggingEnabled = true;
            
            //resolution setting;
            logResolution = resolution;
            lastEpochLogged = -resolution;
        }
    }
}

