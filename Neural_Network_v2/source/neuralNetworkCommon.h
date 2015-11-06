//
//  neuralNetworkCommon_h.h
//  NN
//
//  Created by Lukasz Margielewski on 07/11/15.
//  Copyright © 2015 Lukasz Margielewski. All rights reserved.
//

#ifndef neuralNetworkCommon_h
#define neuralNetworkCommon_h

#ifdef NN_DOUBLE_SUPPORT_ENABLED
    typedef double NNType;
#else
    typedef float NNType;
#endif

#define MEMORY_ALIGNMENT 0x4000
#define MEMORY_ALIGNED_BYTES(x) ceil((NNType)x / (NNType)MEMORY_ALIGNMENT) * MEMORY_ALIGNMENT
//#define MEMORY_ALIGNED_BYTES(x) x

#endif /* neuralNetworkCommon_h */
