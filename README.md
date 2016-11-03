# ByteNet -- TensorFlow Implementation

This is a TensorFlow implementation of the [ByteNet generative neural
network architecture](https://arxiv.org/pdf/1610.10099v1.pdf) for seq2seq text generation. This is a fork from [ibab's tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet).


##Work In Progress

In the bytenet folder you will find the model and its associated ops. Will be working on this throughout the next week.

###Implemented:

* Source Network
* Residual Block (Fig 3 left)
* Regular Batch Normalization 

###Need To Implement

* Target Network
* Masking Causal Convolutions
* Framework for Training and Decoding
* Sub Batch Normalization

##Structure

The structure of the model will be in two parts:

- One ByteNet model is initialized as a source network
- A Second ByteNet model is initialized as a target network

To implement network:

```python
from bytenet import bytenet_model

source_network = bytenet_model.ByteNetModel(args)
source_output = source_network.create_source_network(inputs)

target_network = bytenet_model.ByteNetModel(args) 
output = target_network.create_target_network(source_output, conditional_inputs) #this has not been implemented

```


This is done this way so that ByteNet is modular. This means that you can use a convolution encoder and a RNN decoder. Parts should be interchangeable. 


##Other Notes

*Contributions are welcome!*

List of contributions that would help me out while I work on other parts of the network:

* Masking Causal Convolutions
* Sub Batch Normalization

I apologize for all the comments but it helps me make progress, debug, and understand what is going on specifically. I do many pushes as I experiment heavily with different combinations of models.
