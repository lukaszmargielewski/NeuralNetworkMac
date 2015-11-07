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
    uint lC = NN->_layerCount - 1;
    uint *nPl = NN->_neuronsPerLayer;
    
    deltas      = (NNType ***)malloc(MEMORY_ALIGNED_BYTES(lC * sizeof(NNType **)));
    gradients   = (NNType **)malloc(MEMORY_ALIGNED_BYTES(lC * sizeof(NNType *)));
    
    
    for (uint l = 0; l < lC; l++) {
        
        uint nC = nPl[l];
        uint nCn = nPl[l+1];
        
        gradients[l]    = (NNType *)malloc(MEMORY_ALIGNED_BYTES((nCn + 1) * sizeof(NNType)));
        deltas[l]       = (NNType **)malloc(MEMORY_ALIGNED_BYTES((nC + 1) * sizeof(NNType *)));
        
        for ( int i=0; i <= nC; i++ )
        {
            deltas[l][i] = (NNType *)malloc(MEMORY_ALIGNED_BYTES(nCn * sizeof(NNType)));
            
            for ( int j=0; j < nCn; j++ ) deltas[l][i][j] = 0;
        }
    }

    initializeWeights();
    
}

neuralNetworkTrainer::~neuralNetworkTrainer(){

    uint *nPl = NN->_neuronsPerLayer;
    
    uint lC = NN->_layerCount - 1;
    
    for (int l=0; l < lC; l++){
        
        free(gradients[l]);
    }
    
    free(gradients);
    
    for (int l=0; l < lC; l++){
        
        NNType** ddd = deltas[l];
        uint nC = nPl[l];
        
        for (int j=0; j <= nC; j++)
        {
            free(ddd[j]);
        }
        
        free(ddd);
    }
    
    free(deltas);
}


/*******************************************************************
* Set training parameters
********************************************************************/
void neuralNetworkTrainer::setTrainingParameters( NNType lR, NNType m, bool batch )
{
	learningRate = lR;
	momentum = m;
	useBatch = batch;
}
/*******************************************************************
* Set stopping parameters
********************************************************************/
void neuralNetworkTrainer::setStoppingConditions( int mEpochs, NNType dAccuracy )
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
		NNType previousTAccuracy = trainingSetAccuracy;
		NNType previousGAccuracy = generalizationSetAccuracy;

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
	NNType incorrectPatterns = 0;
	NNType mse = 0;
		
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
void neuralNetworkTrainer::backpropagate( NNType* desiredOutputs )
{		
	//modify deltas between hidden and output layers
	//--------------------------------------------------------------------------------------------------------
    
    uint lastLayerIndex = NN->_layerCount-1;
    
    for (uint l = lastLayerIndex; l > 0; l--) {
        
        NNType * neu            = NN->neurons[l];
        NNType *_neu            = NN->neurons[l-1];
        
        NNType *_grad           = gradients[l-1];
        NNType **_delt          = deltas[l-1];
        
        uint  c   = NN->_neuronsPerLayer[l];
        uint _c   = NN->_neuronsPerLayer[l-1];
        
        for (uint n = 0; n < c; n++) {
            
            NNType value            = neu[n];
            NNType sigmoid_dt       = value * ( 1 - value );
            NNType error            = 0;
            
            if (l == lastLayerIndex) { // Computing error for last layer is very simple:
            
                error  = desiredOutputs[n] - value;
                
            }else{
            
                NNType *w       = NN->weights[l][n];
                uint c_         = NN->_neuronsPerLayer[l+1];
                NNType *grad    = gradients[l];
                
                for( int n_ = 0; n_ < c_; n_++ ){
                    
                    error += w[n_] * grad[n_];
                }
            }
            
            
            NNType g = sigmoid_dt * error;
            _grad[n] = g;
            
            //for all nodes in input layer and bias neuron
            for (int _n = 0; _n <= _c; _n++)
            {
                
                NNType ddd = learningRate * _neu[_n] * g;
                
                //calculate change in weight
                if ( !useBatch ){
                
                    NNType dt = _delt[_n][n];
                    _delt[_n][n] = ddd + momentum * dt;
                }
                else{
                
                    _delt[_n][n] += ddd;
                }
                
            }
        }
    }
    //if using stochastic learning update the weights immediately
    if ( !useBatch ) updateWeights();
}


/*******************************************************************
* Update weights using delta values
********************************************************************/
void neuralNetworkTrainer::updateWeights()
{
	//input -> hidden weights
	//--------------------------------------------------------------------------------------------------------

    
    for (int l = 0; l < NN->_layerCount - 1; l++) {
    
        int c  = NN->_neuronsPerLayer[l];
        int c_ = NN->_neuronsPerLayer[l+1];
        
        for (int n = 0; n <= c; n++)
        {
            NNType *w = NN->weights[l][n];
            
            for (int n_ = 0; n_ < c_; n_++)
            {
                //update weight
                w[n_] += deltas[l][n][n_];
                
                //clear delta only if using batch (previous delta is needed for momentum
                if (useBatch)
                    deltas[l][n][n_] = 0;
            }
        }
        
    }
}


/*******************************************************************
 * Return the NN accuracy on the set
 ********************************************************************/
NNType neuralNetworkTrainer::getSetAccuracy( std::vector<dataEntry*>& set )
{
    NNType incorrectResults = 0;
    
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
NNType neuralNetworkTrainer::getSetMSE( std::vector<dataEntry*>& set )
{
    NNType mse = 0;
    
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
        NNType r = 1/sqrt( (NNType) NN->_neuronsPerLayer[iL0]);
        
        //set weights between input and hidden
        //--------------------------------------------------------------------------------------------------------
        for(int i = 0; i <= NN->_neuronsPerLayer[iL0]; i++)
        {
            NNType *w = NN->weights[iL0][i];
            
            for(int j = 0; j < NN->_neuronsPerLayer[iL1]; j++)
            {
                //set weights to random values
                w[j] = ( ( (NNType)(rand()%100)+1)/100  * 2 * r ) - r;
            }
        }
    }
}


/*******************************************************************
 * Output Clamping
 ********************************************************************/
inline int neuralNetworkTrainer::clampOutput( NNType x )
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

