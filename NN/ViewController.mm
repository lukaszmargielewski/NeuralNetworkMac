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
    
    
    self.progressLabel.stringValue = @"Training...";
    self.progressIndicator.doubleValue = 0;
    
    self.progressIndicator.hidden = self.progressLabel.hidden = NO;
    self.startButton.enabled = NO;
    self.startButton.title = @"Working...";
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
    
        example_nn([inputFile UTF8String], [outputFile UTF8String]);
        
        dispatch_sync(dispatch_get_main_queue(), ^{
        
            self.startButton.enabled = YES;
            self.startButton.title = @"Start";
            self.progressIndicator.hidden = self.progressLabel.hidden = YES;
            
        });
    });
    
    
    
}
@end
