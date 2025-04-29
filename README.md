# Cultural Classification
The task of this homework consists in developing a non LM-based and an LM-based method for cultural classification among three classes: C.A. (cultural agnostic), C.R. (cultural representative) and C.E. (cultural exclusive). 

<img src="./images/image_1.png" alt="Description" width="300" height = "300" />

# Graph-based Method


<img src="./images/image_2.png" alt="Description" width="300" height = "300" />

# LM-based Method
In the LM-based method we have used a Transformer Encoder (like BERT) to extract features from text, a Pooling layer to aggregate all the hidden tensors and a Classifier layer (like a Linear or MLP layer) at the end for the final classification.

<img src="./images/image_3.png" alt="Description" width="300" height = "300" />

# Results
