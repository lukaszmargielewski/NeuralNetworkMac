/*******************************************************************
* Basic Feed Forward Neural Network Class
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
********************************************************************/

#ifndef NNetwork
#define NNetwork

class neuralNetworkTrainer;

class neuralNetwork
{
	//class members
	//--------------------------------------------------------------------------------------------
private:

	//number of neurons
    uint _layerCount;
    uint *_neuronsPerLayer;
    
    double** neurons;
    double*** weights;
    
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
    
    double* getInputLayer(uint *count);
    double* getOutputLayer(uint *count);
    
    double* feedForwardPattern( double* pattern );
    
    //weight operations
    bool loadWeights(const char* inputFilename);
    bool saveWeights(const char* outputFilename);

    double averageFeedForwardTime();
    double averageInputLayerLoadTime();
    uint64_t feedForwardCount();
    
	//private methods
	//--------------------------------------------------------------------------------------------

private: 

    void feedForward( double* pattern );
	
};

#endif
