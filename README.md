# GoogLeNet
implement of GoogLeNet and add pairwise confusion

# dataset 
    make folder in currect dir named "dataset", which include training dataset and test dataset

# GoogLeNet models
    the googlemet in model.py is litter different from googelnet in torch.nn, whose InceptionAux uses
    GlobalAvgPool instead of Liner. Just a attempt, it's caused by the paper <<network in network>>.
    with the struct, net can obtain approximate accuracy with less parameter.

# Loss Function 
    I added the PCLoss on the CrossEntropyLoss after referring to  
    paper<<Pairwise Confusionfor Fine-Grained Visual Classification>>
    and the result increased 82.5% to 81.9% on cubbirds compared with origin GoogLeNet net