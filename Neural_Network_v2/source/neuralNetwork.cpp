//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetwork.h"
#include <mach/mach_time.h>

using namespace std;

#define MEMORY_ALIGNMENT 0x4000
//#define MEMORY_ALIGNED_BYTES(x) ceil((double)x / (double)MEMORY_ALIGNMENT) * MEMORY_ALIGNMENT
#define MEMORY_ALIGNED_BYTES(x) x


static double MachTimeToSecs(uint64_t time);

/*******************************************************************
* Constructor
********************************************************************/
neuralNetwork::neuralNetwork(uint numberOfLayers, uint* neuronsInLayer) : _layerCount(numberOfLayers)
{
    
    _neuronsPerLayer = (uint *)malloc(sizeof(uint) * _layerCount);
    
    neurons = (double **)malloc(MEMORY_ALIGNED_BYTES(sizeof(double *) * _layerCount));
    weights = (double ***)malloc(MEMORY_ALIGNED_BYTES(sizeof(double **) * _layerCount - 1));
    
    
    for (uint iLayer = 0; iLayer < _layerCount; iLayer++ ){
    
        uint layerSize = neuronsInLayer[iLayer];
        _neuronsPerLayer[iLayer] = layerSize;
        bool isLastLayer = (iLayer == (_layerCount - 1)) ? true : false;
        
        
        // Neurons & Weights:
        // Note: last layer (output) does not have a bias!
        // All, except last layer have biases and weights:
        
        uint neuronsCount = isLastLayer ? layerSize : layerSize + 1; // + 1 for bias neuron & bias:
        uint layerBytes = MEMORY_ALIGNED_BYTES((neuronsCount) * sizeof(double));

        neurons[iLayer] = (double *)malloc(layerBytes);

        printf("\n %i / %i layer. size: %i, neurons count %i, is last: %i", iLayer, _layerCount, layerSize, neuronsCount, isLastLayer);
        
        // All, except last layer have biases and weights:
        
        if (isLastLayer == false){
            
            neurons[iLayer][neuronsCount] = -1;
            weights[iLayer] = (double **)malloc(layerBytes);
            
            for ( int i = 0; i < neuronsCount; i++ ){
                
                uint layerSizeNext = neuronsInLayer[iLayer+1];
                uint layerBytesNext = MEMORY_ALIGNED_BYTES((layerSizeNext+1) * sizeof(double));
                
                weights[iLayer][i] = (double *)malloc(layerBytesNext);
            }
        }
        
    }

    
    runCount = 0;
    totalInputLayerLoadTime = totalFeedForwardTime = 0;
    
}

/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork(){
	//delete neurons
    for (int iLayer=0; iLayer < _layerCount; iLayer++){
    
        free(neurons[iLayer]);
    }
    
    free(neurons);

    for (int iLayer=0; iLayer < _layerCount-1; iLayer++){
    
        double** wi = weights[iLayer];
        
        uint weightsCount = _neuronsPerLayer[iLayer] + 1;
        
        for (int j=0; j < weightsCount; j++)
        {
            free(wi[j]);
        }
        
        free(wi);
    }
	
    free(weights);
}

/*******************************************************************
 * Input & Output Layer getters:
 ********************************************************************/

double* neuralNetwork::getInputLayer(uint *count){

    *count = _neuronsPerLayer[0];
    return neurons[0];
    
}
double* neuralNetwork::getOutputLayer(uint *count){

    *count = _neuronsPerLayer[2];
    return neurons[_layerCount-1];

}


/*******************************************************************
 * Feed Forward Operation
 ********************************************************************/
void neuralNetwork::feedForward(double *pattern){
    uint64_t ts = mach_absolute_time();
    //set input neurons to input values
    memcpy(neurons[0], pattern, sizeof(double) * _neuronsPerLayer[0]);
    uint64_t ts1 = mach_absolute_time();
    
    
    for (uint iLDst = 1; iLDst < _layerCount; iLDst++) {
    
        uint iLSrc = iLDst - 1;
        uint cDst = _neuronsPerLayer[iLDst];
        uint cSrc = _neuronsPerLayer[iLSrc];
        
        double *neuronsDst = neurons[iLDst];
        double *neuronsSrc = neurons[iLSrc];
        
        double **weightsSrcDst = weights[iLSrc];
        
        
        //Calculate Hidden Layer values - include bias neuron
        //--------------------------------------------------------------------------------------------------------
        for(int rowDst = 0; rowDst < cDst; rowDst++)
        {
            //clear value
            double sum = 0;
            
            //get weighted sum of pattern and bias neuron
            for( int rowSrc = 0; rowSrc <= cSrc; rowSrc++ ){
                
                sum += neuronsSrc[rowSrc] * weightsSrcDst[rowSrc][rowDst];
            }
            
            //sigmoid
            neuronsDst[rowDst] = ( 1 / (1 + exp(-sum) ) );
        }
        
    }
    
    
    uint64_t te = mach_absolute_time();
    
    totalFeedForwardTime += (te - ts1);
    totalInputLayerLoadTime += (ts1 - ts);
    runCount++;
}

/*******************************************************************
* Load Neuron Weights
********************************************************************/
bool neuralNetwork::loadWeights(const char* filename){
	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);

	if ( inputFile.is_open() )
	{
		vector<double> weightsVector;
		string line = "";
		
		//read data
		while ( !inputFile.eof() )
		{
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) 
			{				
				//store inputs		
				char* cstr = new char[line.size()+1];
				strcpy(cstr, line.c_str());
                weightsVector.push_back( atof(cstr) );
				//free memory
				delete[] cstr;
			}
		}	
		
		//check if sufficient weights were loaded
		if ( weightsVector.size() != ( (_neuronsPerLayer[0] + 1) * _neuronsPerLayer[1] + (_neuronsPerLayer[1] +  1) * _neuronsPerLayer[2] ) )
		{
			cout << endl << "Error - Incorrect number of weights in input file: " << filename << endl;
			
			//close file
			inputFile.close();

			return false;
		}
		else
		{
			//set weights
			int pos = 0;
            
            for (int l = 0; l < _layerCount-1; l++) {
               
                for ( int i = 0; i <= _neuronsPerLayer[l]; i++ )
                {
                    for ( int j = 0; j < _neuronsPerLayer[l+1]; j++ )
                    {
                        weights[l][i][j] = weightsVector[pos++];
                    }
                }
                
            }

			//print success
			cout << endl << "Neuron weights loaded successfuly from '" << filename << "'" << endl;

			//close file
			inputFile.close();
			
			return true;
		}		
	}
	else 
	{
		cout << endl << "Error - Weight input file '" << filename << "' could not be opened: " << endl;
		return false;
	}
}


/*******************************************************************
* Save Neuron Weights
********************************************************************/
bool neuralNetwork::saveWeights(const char* filename){
	//open file for reading
	fstream outputFile;
	outputFile.open(filename, ios::out);

	if ( outputFile.is_open() )
	{
		outputFile.precision(50);		

		//output weights
        int ccc = 1;
        
        for (int l = 0; l < _layerCount - 1; l++) {
         
            for ( int i = 0; i <= _neuronsPerLayer[l]; i++ )
            {
                for ( int j = 0; j < _neuronsPerLayer[l+1]; j++ )
                {
                    double value = weights[l][i][j];
                    
                    outputFile << value << "\n";
                    //printf("\n %i. [%i][%i][%i] = %.3f", ccc, l, i, j, value);
                    
                    ccc++;
                }
            }
        }

		//print success
		cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

		//close file
		outputFile.close();
		
		return true;
	}
	else 
	{
		cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
		return false;
	}
}

/*******************************************************************
 * Profiling Stats:
 ********************************************************************/


double neuralNetwork::averageFeedForwardTime(){

    return MachTimeToSecs((double)totalFeedForwardTime / (double)runCount);
    
}
double neuralNetwork::averageInputLayerLoadTime(){

    return MachTimeToSecs((double)totalInputLayerLoadTime / (double)runCount);
}

uint64_t neuralNetwork::feedForwardCount(){
    return runCount;
}
static double MachTimeToSecs(uint64_t time){
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    return (double)time * (double)timebase.numer /
    (double)timebase.denom / 1e9;
}


