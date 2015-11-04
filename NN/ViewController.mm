//
//  ViewController.m
//  NN
//
//  Created by Lukasz Margielewski on 04/11/15.
//  Copyright © 2015 Lukasz Margielewski. All rights reserved.
//

#import "ViewController.h"
#include "letter_recognition_training_example.h"

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    // Do any additional setup after loading the view.
}

- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}

- (IBAction)startAction:(id)sender {
    

    NSString *inputFile        = [[NSBundle mainBundle] pathForResource:@"letter-recognition-2" ofType:@"csv"];
    
    NSString *outputFile       = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
    outputFile = [outputFile stringByAppendingPathComponent:@"weights.csv"];
    
    
    
    example_nn([inputFile UTF8String], [outputFile UTF8String]);
    
}
@end
