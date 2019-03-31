# End_to_End_Code_Generation
The goal of this projects is to be able to generate code snippets in Python from natural language decriptions following an End-to-End approach (parallel text corpora: intentions (NL) Vs. Snippets (Python)).

## Dataset

https://conala-corpus.github.io/

Code/Natural Language Challenge, a joint project of the Carnegie Mellon University NeuLab and STRUDEL Lab! This challenge was designed to test systems for generating program snippets from natural language. For example, if the input is:

> sort list x in reverse order

Then the system would be required to output

> x.sort(reverse=True)

The data is available in [conala-train.json](conala-train.json) and [conala-test.json](conala-test.json). Split into 2,379 training and 500 test examples. The json file format is shown as [follows](json.PNG):

## Framework
Tensorflow - Keras

## Description

Adaptation of [Machine Translation](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/21_Machine_Translation.ipynb) neural network. I am considering sequences of words and not of single characters. An encoder which maps the source-text to a "thought vector" that summarizes the text's contents, which is then input to the second part of the neural network that decodes the "thought vector" to the destination-text.

### Encoder
Neural Networks cannot work directly on text-data. We use a two-step process to convert text into numbers that can be used in a neural network.
#### Tokenizer
Convert each word to an integer-token. It also reverses the order of the words because this could increase, said in 
##### [Output]

#### Embedding
Convert each integer-token to a vector of floating-point values. The embedding is trained alongside the rest of the neural network to map words with similar semantic meaning to similar vectors of floating-point values.
##### [Output]

**These embedding-vectors can then be input to the Recurrent Neural Network, which has 3 GRU-layers.**

The last GRU-layer outputs a single vector - the "thought vector" that summarizes the contents of the source-text - which is then used as the initial state of the GRU-units in the decoder-part. 

### Decoder
The destination-text is padded with special markers to indicate its beginning and end.
##### [Output]

## Results
### Parameters
  - Number of words: 
  - Embedding size: 
  - Number of epochs: 
  - Number of layers: 
  - Number of ....: 

[Training the model]

### Intention-to-Snippet

## Improvements
The results are always the same and are not correct. This could be improved considering some adjustments:
  - Don't work with words but character to character
  - Mind the rewritten intent of json file for better matching
  - Change the dataset
  
