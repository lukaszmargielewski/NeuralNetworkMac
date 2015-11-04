//
//  ViewController.h
//  NN
//
//  Created by Lukasz Margielewski on 04/11/15.
//  Copyright © 2015 Lukasz Margielewski. All rights reserved.
//

#import <Cocoa/Cocoa.h>


@interface ViewController : NSViewController

@property (weak) IBOutlet NSTextField *progressLabel;
@property (weak) IBOutlet NSProgressIndicator *progressIndicator;

@property (weak) IBOutlet NSButton *startButton;
- (IBAction)startAction:(id)sender;

@end

