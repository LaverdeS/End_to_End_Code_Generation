# End_to_End_Code_Generation
The goal of this projects is to be able to generate code snippets in Python from natural language decriptions following an End-to-End approach (parallel text corpora: intentions (NL) Vs. Snippets (Python)).

## Dataset

https://conala-corpus.github.io/

Code/Natural Language Challenge, a joint project of the Carnegie Mellon University NeuLab and STRUDEL Lab! This challenge was designed to test systems for generating program snippets from natural language. For example, if the input is:

> sort list x in reverse order

Then the system would be required to output

> x.sort(reverse=True)

The data is available in [conala-train.json](conala-train.json) and [conala-test.json](conala-test.json). Split into 2,379 training and 500 test examples. The json file format is shown as [follows](json.PNG):

## Requisites
Tensorflow - Keras

## Description

Adaptation of Neural Network for [Machine Translation](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/21_Machine_Translation.ipynb). I am considering sequences of words and not of single characters. An encoder which maps the source-text to a "thought vector" that summarizes the text's contents, which is then input to the second part of the neural network that decodes the "thought vector" to the destination-text.

### Encoder
Neural Networks cannot work directly on text-data. We use a two-step process to convert text into numbers that can be used in a neural network.
#### Tokenizer
  - Convert each word to an integer-token. 
  - Reverses the order of the words because this could increase, said in . 
  - Set the maximum **number of words** in our vocabulary as well. Pad and truncate sequences to the given length.
  
  > Number of words: 5000
  > Shape of tokenization of intents:  (2379, 15)
  > Shape of tokenization of snippets:  (2379, 17)
  
  Example for intent case 5: 
  
  intent: **Prepend the same string to all items in a list**
  > snippet: start ['hello{0}'.format(i) for i in a] end
  
  intent:  [  0   0   0   0   0   7   1   2  78  28   4   8 102   9 929]
  snippet:  [  1 290   9   3  63  14   6  14   4   8   2   0   0   0   0   0   0]

#### Embedding
Convert each integer-token to a vector of floating-point values. The embedding is trained alongside the rest of the neural network to map words with similar semantic meaning to similar vectors of floating-point values.

> embedding_size = 128

**These embedding-vectors can then be input to the Recurrent Neural Network, which has 3 GRU-layers.**

The last GRU-layer outputs a single vector - the "thought vector" that summarizes the contents of the source-text - which is then used as the initial state of the GRU-units in the decoder-part. 

### Decoder
The destination-text is padded with special markers to indicate its beginning and end.

  > decoder input shape:  (2379, 16)
  > decoder output shape:  (2379, 16)
[  1 290   9   3  63  14   6  14   4   8   2   0   0   0   0   0]
[290   9   3  63  14   6  14   4   8   2   0   0   0   0   0   0]

## Results
### Parameters
  - Number of words: 5000
  - Embedding Size: 128
  - Number of States: 512
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

