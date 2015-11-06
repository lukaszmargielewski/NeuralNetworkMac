/*******************************************************************
* Neural Network Training Class
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
*********************************************************************/

#ifndef NNetworkTrainer
#define NNetworkTrainer

//standard includes
#include <fstream>
#include <vector>

//neural network header
#include "neuralNetwork.h"
#include "dataReader.h"

//Constant Defaults!
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 90  
#define DESIRED_MSE 0.001 

/*******************************************************************
* Basic Gradient Descent Trainer with Momentum and Batch Learning
********************************************************************/
class neuralNetworkTrainer
{
	//class members
	//--------------------------------------------------------------------------------------------

private:

	//network to be trained
	neuralNetwork* NN;

	//learning parameters
	NNType learningRate;					// adjusts the step size of the weight update	
	NNType momentum;						// improves performance of stochastic learning (don't use for batch)

	//epoch counter
	long epoch;
	long maxEpochs;
	
	//accuracy/MSE required
	NNType desiredAccuracy;
	
	//change to weights
	NNType*** deltas;
	//error gradients
	NNType** gradients;

	//accuracy stats per epoch
	NNType trainingSetAccuracy;
	NNType validationSetAccuracy;
	NNType generalizationSetAccuracy;
	NNType trainingSetMSE;
	NNType validationSetMSE;
	NNType generalizationSetMSE;

	//batch learning flag
	bool useBatch;

	//log file handle
	bool loggingEnabled;
	std::fstream logFile;
	int logResolution;
	int lastEpochLogged;
    
    NNType *outputLayer;
    uint   outputLayerCount;
    

	//public methods
	//--------------------------------------------------------------------------------------------
public:	
	
	neuralNetworkTrainer( neuralNetwork* untrainedNetwork );
    ~neuralNetworkTrainer();
    void setTrainingParameters( NNType lR, NNType m, bool batch );
	void setStoppingConditions( int mEpochs, NNType dAccuracy);
	void useBatchLearning( bool flag ){ useBatch = flag; }
	void enableLogging( const char* filename, int resolution );

	void trainNetwork( trainingDataSet* tSet );
	//private methods
	//--------------------------------------------------------------------------------------------
private:

	void runTrainingEpoch( std::vector<dataEntry*> trainingSet );
	void backpropagate(NNType* desiredOutputs);
	void updateWeights();
    
    void initializeWeights();
    NNType getSetAccuracy( std::vector<dataEntry*>& set );
    NNType getSetMSE( std::vector<dataEntry*>& set );
    inline int clampOutput( NNType x );
    
    
};


#endif