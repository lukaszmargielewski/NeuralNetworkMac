//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetwork.h"
#include <mach/mach.h>
#include <stdlib.h>

using namespace std;

#define MEMORY_ALIGNMENT 0x4000
#define MEMORY_ALIGNED_BYTES(x) ceil((double)x / (double)MEMORY_ALIGNMENT) * MEMORY_ALIGNMENT

/*******************************************************************
* Constructor
********************************************************************/
neuralNetwork::neuralNetwork(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO)
{				
	//create neuron lists
	//--------------------------------------------------------------------------------------------------------

    uint nInputBytes = MEMORY_ALIGNED_BYTES((nInput + 1) * sizeof(double));
    uint nHiddenBytes = MEMORY_ALIGNED_BYTES((nHidden + 1) * sizeof(double));
    uint nOutputBytes = MEMORY_ALIGNED_BYTES((nOutput + 1) * sizeof(double));
    
    /*
    posix_memalign((void *)inputNeurons, MEMORY_ALIGNMENT, bytes);
    */
    
    inputNeurons    = (double *)malloc(nInputBytes);
    hiddenNeurons   = (double *)malloc(nHiddenBytes);
    outputNeurons   = (double *)malloc(nOutputBytes);;
    
	//create input bias neuron
	inputNeurons[nInput] = -1;
    //create hidden bias neuron
	hiddenNeurons[nHidden] = -1;

    
	//create weight lists (include bias neuron weights)
	//--------------------------------------------------------------------------------------------------------
	
    wInputHidden = (double **)malloc(sizeof(double *) * (nInput + 1));
    
    for ( int i=0; i <= nInput; i++ )
		wInputHidden[i] = (double *)malloc((nHidden) * sizeof(double));

    
    wHiddenOutput = (double **)malloc(sizeof(double *) * (nHidden + 1));
    
	for ( int i=0; i <= nHidden; i++ )
        wHiddenOutput[i] = (double *)malloc((nOutput) * sizeof(double));

}

/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork()
{
	//delete neurons
	free(inputNeurons);
	free(hiddenNeurons);
	free(outputNeurons);

	//delete weight storage
	for (int i=0; i <= nInput; i++)free(wInputHidden[i]);
    
	free(wInputHidden);

	for (int j=0; j <= nHidden; j++)free(wHiddenOutput[j]);
	free(wHiddenOutput);
}

/*******************************************************************
 * Feed pattern through network and return results
 ********************************************************************/
double* neuralNetwork::feedForwardPattern(double *pattern)
{
    feedForward(pattern);
    return outputNeurons;
}


/*******************************************************************
 * Feed Forward Operation
 ********************************************************************/
void neuralNetwork::feedForward(double* pattern)
{
    //set input neurons to input values
    memcpy(inputNeurons, pattern, sizeof(double) * nInput);
    
    //Calculate Hidden Layer values - include bias neuron
    //--------------------------------------------------------------------------------------------------------
    for(int j=0; j < nHidden; j++)
    {
        //clear value
        hiddenNeurons[j] = 0;
        
        //get weighted sum of pattern and bias neuron
        for( int i=0; i <= nInput; i++ )
            hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];
        
        //set to result of sigmoid
        hiddenNeurons[j] = activationFunction( hiddenNeurons[j] );
    }
    
    //Calculating Output Layer values - include bias neuron
    //--------------------------------------------------------------------------------------------------------
    for(int k=0; k < nOutput; k++)
    {
        //clear value
        outputNeurons[k] = 0;
        
        //get weighted sum of pattern and bias neuron
        for( int j=0; j <= nHidden; j++ )
            outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];
        
        //set to result of sigmoid
        outputNeurons[k] = activationFunction( outputNeurons[k] );
    }
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



