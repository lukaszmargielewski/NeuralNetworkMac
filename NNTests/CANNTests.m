//
//  CANNTests.m
//  NN
//
//  Created by Lukasz Margielewski on 13/11/15.
//  Copyright © 2015 Lukasz Margielewski. All rights reserved.
//

#import <XCTest/XCTest.h>

#import "cann_train.h"

@interface CANNTests : XCTestCase{

    struct Cann *cann;
    
}

@end

@implementation CANNTests

- (void)setUp {
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}
- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
    
    CannDestroy(cann);
}


#pragma mark - Tests:

- (void)testCANNlayersIntegrity{
    
    cann = CannCreate(3,10,10,10);
    
    CannLayer *inputLayer   = cann->inputLayer;
    CannLayer *middleLayer  = cann->outputLayer->prevLayer;
    CannLayer *outputLayer  = cann->outputLayer;
    
    XCTAssert(inputLayer->gradients == NULL);
    XCTAssert(middleLayer->gradients == NULL);
    XCTAssert(outputLayer->gradients == NULL);
    
    [self basicNonTrainingModeCannIntegrityTests];
    
    CannEnableTrainingMode(cann);
    [self basicTrainingModeCannIntegrityTests];
    
    CannDisableTrainingMode(cann);
    [self basicNonTrainingModeCannIntegrityTests];
 
    XCTAssert(inputLayer->gradients == NULL);
    XCTAssert(middleLayer->gradients == NULL);
    XCTAssert(outputLayer->gradients == NULL);
    
    CannDestroy(cann);
    
    
}
- (void)testCANNSaveAndLoad{

    cann = CannCreate(3, 3, 3, 3);
    
    CannType weights[] = {
                            1,2,3,
                            4,5,6,
                            7,8,9,
                            10,11,12,
        
                            13,14,15,
                            16,17,18,
                            19,20,21,
                            22,23,24,
                        };
    
    
    long weightsCount = sizeof(weights) / sizeof(CannType);
    CannLoadWeights(cann, weights, weightsCount);
    
    NSString *dir = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
    NSString *filePath = [dir stringByAppendingPathComponent:@"test_cann.cann"];
    
    const char *fp = [filePath UTF8String];
    
    bool result = CannSaveToFilePath(cann, fp);
    XCTAssert(result == true, @"Save failed");
    
    CannDestroy(cann);
    cann = CannCreateFromFilePath(fp);
    
    XCTAssert(cann->layersCount == 3);
    XCTAssert(cann->inputLayer->neuronsCount == 3);
    XCTAssert(cann->inputLayer->nextLayer->neuronsCount == 3);
    XCTAssert(cann->outputLayer->neuronsCount == 3);
    
    
    CannLayer *layer = cann->inputLayer;
    
    long toLoadLeftCount = weightsCount;
    
    bool everythingOK = true;
    
    long j = 0;
    
    while (layer != NULL && layer->nextLayer != NULL) {
        
        CannTypeArray2D *www = layer->weights;
        
        long toLoadThisLayer = www->count;
        toLoadLeftCount -= toLoadThisLayer;
        
        if (toLoadLeftCount < 0) {
            
            everythingOK = false;
            break;
        }
        
        CannType *pointer = www->pointer;
        
        for (long i = 0; i < toLoadThisLayer; ++i, ++j) {
            
            CannType wi = weights[j];
            CannType wf = pointer[i];
            NSLog(@"%li weight: %f (suppose to be: %f)", j + 1, wf, wi);
            
            XCTAssert(wf == wi);
        }
        
        
        layer = layer->nextLayer;
    }
    
    
    
}

#pragma mark - functions:

- (void)basicTrainingModeCannIntegrityTests{

    CannLayer *inputLayer   = cann->inputLayer;
    CannLayer *middleLayer  = cann->outputLayer->prevLayer;
    CannLayer *outputLayer  = cann->outputLayer;
    
    XCTAssert(inputLayer->deltas->size.rows == 11);
    XCTAssert(inputLayer->deltas->size.columns == 10);
    
    XCTAssert(middleLayer->deltas->size.rows == 11);
    XCTAssert(middleLayer->deltas->size.columns == 10);
    
    XCTAssert(outputLayer->deltas == NULL);
    XCTAssert(outputLayer->deltas == NULL);
    
    XCTAssert(inputLayer->gradients->count == 10);
    XCTAssert(middleLayer->gradients->count == 10);
    XCTAssert(outputLayer->gradients == NULL);
    
}
- (void)basicNonTrainingModeCannIntegrityTests{

    CannLayer *inputLayer   = cann->inputLayer;
    CannLayer *middleLayer  = cann->outputLayer->prevLayer;
    CannLayer *outputLayer  = cann->outputLayer;
    
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
    XCTAssert(cann->inputLayer->neuronsCount == 10);
    XCTAssert(cann->inputLayer->neurons->count == 11);
    
    XCTAssert(cann->outputLayer->neuronsCount == 10);
    XCTAssert(cann->outputLayer->neurons->count == 11);
    
    XCTAssert(middleLayer->nextLayer == cann->outputLayer);
    
    XCTAssert(cann->outputLayer->prevLayer == middleLayer);
    XCTAssert(cann->inputLayer->nextLayer == middleLayer);
    
    // Biases:
    XCTAssert(cann->inputLayer->neurons->pointer[10] == -1);
    XCTAssert(middleLayer->neurons->pointer[10] == -1);
    XCTAssert(cann->outputLayer->neurons->pointer[10] == -1);

}


@end
