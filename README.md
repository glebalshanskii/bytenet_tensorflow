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

##Results

With the source network, it does perform pretty well for language tasks, but not as well as a standard vanilla LSTM 3 layer stack. The bytenet source network certainly consumes your gpus. I running it on three oc Titan X's and all three at 100% gpu usage most of the time.

In terms of actual wall-time, the 1024 dilation (25 layers repeating 1,2,4,8,16 rates) is slightly faster than 1024 unit LSTM. Not exactly comparing them appropriately because the source network has way more parameters, and there are far more computations being done. However, given the fact that the wall time is almost the same due to parrallelization, I feel the source network has promise.

##Other Notes

*Contributions are welcome!*

List of contributions that would help me out while I work on other parts of the network:

* Masking Causal Convolutions
* Sub Batch Normalization
* Multiple Integration Advanced Block Network (fig 3 right)

I apologize for all the comments but it helps me make progress, debug, and understand what is going on specifically. I do many pushes as I experiment heavily with different combinations of models.
