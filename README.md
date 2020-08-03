# GoogLeNet
implement of GoogLeNet and add pairwise confusion

# dataset 
    make folder named "dataset" in currect dir , which includes training dataset and testing dataset

# GoogLeNet models
    the googlenet in model.py is litter different from googelnet in torch.nn, whose InceptionAux uses
    GlobalAvgPool instead of Liner in the end of the branch. Just a attempt, it's referring to the 
    paper <<network in network>>. with the struct, net can obtain approximate accuracy with less parameters.

# Loss Function 
    I added the PCLoss on the CrossEntropyLoss referring to  
    paper<<Pairwise Confusionfor Fine-Grained Visual Classification>>
    and the result increased 82.5% from 81.9% on cubbirds compared with origin GoogLeNet net