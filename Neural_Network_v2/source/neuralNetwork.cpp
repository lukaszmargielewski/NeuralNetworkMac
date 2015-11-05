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
neuralNetwork::neuralNetwork(uint numberOfLayers, uint* neuronsInLayer) : nLayers(numberOfLayers)
{
    
    nNeuronsPerLayer = (uint *)malloc(sizeof(uint) * nLayers);
    
    neurons = (double **)malloc(MEMORY_ALIGNED_BYTES(sizeof(double *) * nLayers));
    weights = (double ***)malloc(MEMORY_ALIGNED_BYTES(sizeof(double **) * nLayers - 1));
    
    
    for (uint iLayer = 0; iLayer < nLayers; iLayer++ ){
    
        uint layerSize = neuronsInLayer[iLayer];
        nNeuronsPerLayer[iLayer] = layerSize;
        bool isLastLayer = (iLayer == (nLayers - 1)) ? true : false;
        
        
        // Neurons & Weights:
        // Note: last layer (output) does not have a bias!
        
        uint neuronsCount = isLastLayer ? layerSize : layerSize + 1;
        uint layerBytes = MEMORY_ALIGNED_BYTES((neuronsCount) * sizeof(double));

        neurons[iLayer] = (double *)malloc(layerBytes);

        printf("\n %i / %i layer. size: %i, neurons count %i, is last: %i", iLayer, nLayers, layerSize, neuronsCount, isLastLayer);
        // Bias neuron:
        if (isLastLayer == false){
        
            neurons[iLayer][neuronsCount] = -1;
            
            weights[iLayer] = (double **)malloc(layerBytes);
            
            for ( int i = 0; i < neuronsCount; i++ ){
                
                uint layerSizeNext = neuronsInLayer[iLayer+1];
                uint layerBytesNext = MEMORY_ALIGNED_BYTES((layerSizeNext+1) * sizeof(double));
                
                weights[iLayer][i] = (double *)malloc(layerBytesNext);
            }
         
            /*
             Replacement for:
             
             wInputHidden  = (double **)malloc(sizeof(double *) * (nInput + 1));
             wHiddenOutput = (double **)malloc(sizeof(double *) * (nHidden + 1));

             for ( int i=0; i <= nInput; i++ )
             wInputHidden[i] = (double *)malloc((nHidden) * sizeof(double));
             
             
             
             for ( int i=0; i <= nHidden; i++ )
             wHiddenOutput[i] = (double *)malloc((nOutput) * sizeof(double));
             
             */
        }
        
    }

    
    runCount = 0;
    totalInputLayerLoadTime = totalFeedForwardTime = 0;

    
    nInput  = nNeuronsPerLayer[0];
    nHidden = nNeuronsPerLayer[1];
    nOutput = nNeuronsPerLayer[2];
    
    inputNeurons = neurons[0];
    hiddenNeurons = neurons[1];
    outputNeurons = neurons[2];
    
    wInputHidden = weights[0];
    wHiddenOutput = weights[1];
    
}

/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork()
{
	//delete neurons
    for (int iLayer=0; iLayer < nLayers; iLayer++){
    
        free(neurons[iLayer]);
    }
    
    free(neurons);

    for (int iLayer=0; iLayer < nLayers-1; iLayer++){
    
        double** wHiddenOutput = weights[iLayer];
        
        uint weightsCount = nNeuronsPerLayer[iLayer] + 1;
        
        for (int j=0; j < weightsCount; j++)
        {
            free(wHiddenOutput[j]);
        }
        
        free(wHiddenOutput);
    }
	
    free(weights);
}

/*******************************************************************
 * Feed pattern through network and return results
 ********************************************************************/
double* neuralNetwork::feedForwardPattern(double *pattern)
{
    feedForward(pattern);
    return neurons[nLayers-1];
}


/*******************************************************************
 * Feed Forward Operation
 ********************************************************************/
void neuralNetwork::feedForward(double* pattern)
{
    uint64_t ts = mach_absolute_time();
    //set input neurons to input values
    memcpy(inputNeurons, pattern, sizeof(double) * nInput);
    uint64_t ts1 = mach_absolute_time();
    
    
    //Calculate Hidden Layer values - include bias neuron
    //--------------------------------------------------------------------------------------------------------
    for(int i1 = 0; i1 < nHidden; i1++)
    {
        //clear value
        hiddenNeurons[i1] = 0;
        
        //get weighted sum of pattern and bias neuron
        
        for( int i0 = 0; i0 <= nInput; i0++ ){
        
            double nI0 = inputNeurons[i0];
            double wI0i1 = wInputHidden[i0][i1];
            
            hiddenNeurons[i1] += nI0 * wI0i1;
        }
        
        //set to result of sigmoid
        hiddenNeurons[i1] = activationFunction( hiddenNeurons[i1] );
    }
    
    //Calculating Output Layer values - include bias neuron
    //--------------------------------------------------------------------------------------------------------
    for(int i2 = 0; i2 < nOutput; i2++)
    {
        //clear value
        outputNeurons[i2] = 0;
        
        //get weighted sum of pattern and bias neuron
        for( int i1 = 0; i1 <= nHidden; i1++ ){
        
            double nI1 = hiddenNeurons[i1];
            double wI1i2 = wHiddenOutput[i1][i2];
            
            outputNeurons[i2] += nI1 * wI1i2;

        }
        
        //set to result of sigmoid
        outputNeurons[i2] = activationFunction( outputNeurons[i2] );
    }
    
    uint64_t te = mach_absolute_time();
    
    totalFeedForwardTime += (te - ts1);
    totalInputLayerLoadTime += (ts1 - ts);
    runCount++;
}

/*******************************************************************
 * Activation Function
 ********************************************************************/
inline double neuralNetwork::activationFunction( double x )
{
    //sigmoid function
    return 1/(1+exp(-x));
}



/*******************************************************************
* Load Neuron Weights
********************************************************************/
bool neuralNetwork::loadWeights(const char* filename)
{
	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);

	if ( inputFile.is_open() )
	{
		vector<double> weights;
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
				char* t;
				strcpy(cstr, line.c_str());

				//tokenise
				int i = 0;
				t=strtok (cstr,",");
				
				while ( t!=NULL )
				{	
					weights.push_back( atof(t) );
				
					//move token onwards
					t = strtok(NULL,",");
					i++;			
				}

				//free memory
				delete[] cstr;
			}
		}	
		
		//check if sufficient weights were loaded
		if ( weights.size() != ( (nInput + 1) * nHidden + (nHidden +  1) * nOutput ) ) 
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

			for ( int i=0; i <= nInput; i++ ) 
			{
				for ( int j=0; j < nHidden; j++ ) 
				{
					wInputHidden[i][j] = weights[pos++];					
				}
			}
			
			for ( int i=0; i <= nHidden; i++ ) 
			{		
				for ( int j=0; j < nOutput; j++ ) 
				{
					wHiddenOutput[i][j] = weights[pos++];						
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
bool neuralNetwork::saveWeights(const char* filename)
{
	//open file for reading
	fstream outputFile;
	outputFile.open(filename, ios::out);

	if ( outputFile.is_open() )
	{
		outputFile.precision(50);		

		//output weights
		for ( int i=0; i <= nInput; i++ ) 
		{
			for ( int j=0; j < nHidden; j++ ) 
			{
				outputFile << wInputHidden[i][j] << ",";				
			}
		}
		
		for ( int i=0; i <= nHidden; i++ ) 
		{		
			for ( int j=0; j < nOutput; j++ ) 
			{
				outputFile << wHiddenOutput[i][j];					
				if ( i * nOutput + j + 1 != (nHidden + 1) * nOutput ) outputFile << ",";
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


double neuralNetwork::averageFeedForwardTime()
{

    return MachTimeToSecs((double)totalFeedForwardTime / (double)runCount);
    
}
double neuralNetwork::averageInputLayerLoadTime(){

    return MachTimeToSecs((double)totalInputLayerLoadTime / (double)runCount);
}

uint64_t neuralNetwork::feedForwardCount()
{
    return runCount;
}
static double MachTimeToSecs(uint64_t time)
{
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    return (double)time * (double)timebase.numer /
    (double)timebase.denom / 1e9;
}


