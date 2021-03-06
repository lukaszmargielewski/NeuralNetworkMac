/*******************************************************************
* Basic Feed Forward Neural Network Class
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
********************************************************************/

#ifndef NNetwork
#define NNetwork

#include <vector>


#include "neuralNetworkCommon.h"

using namespace std;


class neuralNetworkTrainer;
class neuralNetwork
;

typedef void (* neuralNetworkCallback)(neuralNetwork*network, NNType *pattern, NNType *result);

class neuralNetwork
{
	//class members
	//--------------------------------------------------------------------------------------------
private:

	//number of neurons
    uint _layerCount;
    uint *_neuronsPerLayer;
    
    NNType** neurons;
    NNType*** weights;
    
    //stats:
    uint64_t runCount;
    uint64_t totalFeedForwardTime;
    uint64_t totalInputLayerLoadTime;
    
	//Friends
	//--------------------------------------------------------------------------------------------
	friend neuralNetworkTrainer;
	
	//public methods
	//--------------------------------------------------------------------------------------------

public:

	//constructor & destructor
    neuralNetwork(uint numberOfLayers, uint* neuronsInLayer);
    ~neuralNetwork();
    
    NNType* getInputLayer(uint *count);
    NNType* getOutputLayer(uint *count);
    
    NNType* feedForwardPattern( NNType* pattern );
    
    vector<NNType> weightsVector();
        
    neuralNetworkCallback callback;
    
    //weight operations
    bool loadWeights(const char* inputFilename);
    bool saveWeights(const char* outputFilename);

    NNType averageFeedForwardTime();
    NNType averageInputLayerLoadTime();
    uint64_t feedForwardCount();
    
	//private methods
	//--------------------------------------------------------------------------------------------

private: 

    void feedForward( NNType* pattern );
	
};

#endif
