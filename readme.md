## Repository for a research for person-following in real robot based on state tracking

### Overview

soft-state-tracker is a robust target following framework including state tracking and navigation based on CNNs and MLP that can deal with challenging human following tasks with 
safety and efficiency.

### directory structure

* models  ------ state tracker networks and convolutional auto-encoder network
* tracker ------ state tracker server for evaluate and deploy 
* train   ------ train scripts to training all networks
* AE_train.py ------- 
* record.csv  ------- record the outputs of two different state-tracker networks in the same test environment for evaluation
* train.ipynb ------- jupyter notebook file to train the convolutional auto-encoder network
* utils.py    ------- tools for sorting the samples and rename the every item to corrent sequence
* visualization.py ------ some funtions for visualization

### tracker network framework



