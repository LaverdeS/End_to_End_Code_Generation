# End_to_End_Code_Generation
The goal of this projects is to be able to generate code snippets in Python from natural language decriptions following an End-to-End approach (parallel text corpora: intentions (NL) Vs. Snippets (Python)).

## Dataset

https://conala-corpus.github.io/

Code/Natural Language Challenge, a joint project of the Carnegie Mellon University NeuLab and STRUDEL Lab! This challenge was designed to test systems for generating program snippets from natural language. For example, if the input is:

> sort list x in reverse order

Then the system would be required to output

> x.sort(reverse=True)

The data is available in [conala-train.json](conala-train.json) and [conala-test.json](conala-test.json). Split into 2,379 training and 500 test examples. The json file format is shown as [follows](json.PNG):

## Requirements
Tensorflow - Keras

## Description

Adaptation of Neural Network for [Machine Translation](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/21_Machine_Translation.ipynb). I am considering sequences of words and not of single characters. An encoder which maps the source-text to a "thought vector" that summarizes the text's contents, which is then input to the second part of the neural network that decodes the "thought vector" to the destination-text.

### Encoder
Neural Networks cannot work directly on text-data. We use a two-step process to convert text into numbers that can be used in a neural network.

#### Tokenizer
  - Convert each word to an integer-token. 
  - Reverses the sequences of the words in the source text to make it closer to the destination text (for exmaple the first words in source and destination texts) so the model learns better the words dependencies.
  - Set the maximum **number of words** in our vocabulary as well. Pad and truncate sequences to the given length.
  
  > Number of words: 5000
  
  > Shape of tokenization of intents:  (2379, 15)
  
  > Shape of tokenization of snippets:  (2379, 17)
  
  Example for intent case 5: 
  
  intent: **Prepend the same string to all items in a list**
  
  > snippet: start ['hello{0}'.format(i) for i in a] end
  
  intent:  [  0   0   0   0   0   7   1   2  78  28   4   8 102   9 929]
  
  snippet:  [  1 290   9   3  63  14   6  14   4   8   2   0   0   0   0   0   0]
  
  **Markers 'start' and 'end' are tokenized as 1 and 2.**

#### Embedding
Convert each integer-token to a vector of floating-point values. The embedding is trained alongside the rest of the neural network to map words with similar semantic meaning to similar vectors of floating-point values.

> embedding_size = 128

This means that an integer token for words will be converted to a vector that has 128 floating-point numbers. Words with similar semantic meaning will have a similiar vector.

We need embedding for both encoder and decoder.

**These embedding-vectors can then be input to the Recurrent Neural Network, which has 3 GRU-layers.**

The last GRU-layer outputs a single vector - the "thought vector" that summarizes the contents of the source-text - which is then used as the initial state of the GRU-units in the decoder-part. 

### **GATE RECURRENT UNITS (GRUs)**

The GRU is like a long short-term memory (LSTM) with forget gate but has fewer parameters than LSTM, as it lacks an output gate. GRU's performance on certain tasks of polyphonic music modeling and speech signal modeling was found to be similar to that of LSTM. GRUs have been shown to exhibit even better performance on certain smaller datasets.

> State size: 512

Creates the 3 GRU layers that will map from a sequence of embedding-vectors to a single "thought vector" which summarizes the contents of the input-text.

The GRU layers output a tensor with shape [batch_size, sequence_length, state_size], where each "word" is encoded as a vector of length state_size. We need to convert this into sequences of integer-tokens that can be interpreted as words from our vocabulary.

### Decoder
The destination-text is padded with special markers to indicate its beginning and end.

  > decoder input shape:  (2379, 16)
  
  > decoder output shape:  (2379, 16)

Example with intent [5]:

[  1 290   9   3  63  14   6  14   4   8   2   0   0   0   0   0]

[290   9   3  63  14   6  14   4   8   2   0   0   0   0   0   0]

**token to text decoder output:** start 'hello 0 ' format i for i in a end

The decoder is built using the functional API of Keras, which allows more flexibility in connecting the layers e.g. to route different inputs to the decoder. This is useful because we have to connect the decoder directly to the encoder, but we will also connect the decoder to another input to run it separately.

This function connects all the layers of the decoder to some input of the initial-state values for the GRU layers.

The decoder outputs a tensor with shape [batch_size, sequence_length, num_words] which contains batches of sequences of one-hot encoded arrays of length num_words. We will compare this to a tensor with shape [batch_size, sequence_length] containing sequences of integer-tokens.

This comparison is done with a sparse-cross-entropy function directly from TensorFlow.

## Results
### Parameters
  - Number of words: 5000
  - Embedding Size: 128
  - Number of States: 512
  - Number of epochs: 20
  - Number of layers: 3

### Intention-to-Snippet
#### Loss Reduction
2379/2379 [==============================] - 57s 24ms/step - loss: 6.6142
Epoch 2/20
2379/2379 [==============================] - 48s 20ms/step - loss: 3.9104
Epoch 3/20
2379/2379 [==============================] - 48s 20ms/step - loss: 3.6865
Epoch 4/20
2379/2379 [==============================] - 51s 21ms/step - loss: 3.5967
Epoch 5/20
2379/2379 [==============================] - 52s 22ms/step - loss: 6.5992
Epoch 6/20
2379/2379 [==============================] - 51s 21ms/step - loss: 3.6097
Epoch 7/20
2379/2379 [==============================] - 50s 21ms/step - loss: 3.4128
Epoch 8/20
2379/2379 [==============================] - 48s 20ms/step - loss: 3.4724
Epoch 9/20
2379/2379 [==============================] - 48s 20ms/step - loss: 3.3675
Epoch 10/20
2379/2379 [==============================] - 49s 21ms/step - loss: 3.3892
Epoch 11/20
2379/2379 [==============================] - 51s 21ms/step - loss: 3.3543
Epoch 12/20
2379/2379 [==============================] - 48s 20ms/step - loss: 3.2484
Epoch 13/20
2379/2379 [==============================] - 50s 21ms/step - loss: 3.2142
Epoch 14/20
2379/2379 [==============================] - 49s 20ms/step - loss: 3.1691
Epoch 15/20
2379/2379 [==============================] - 49s 20ms/step - loss: 2.9812
Epoch 16/20
2379/2379 [==============================] - 49s 20ms/step - loss: 2.9265
Epoch 17/20
2379/2379 [==============================] - 54s 23ms/step - loss: 2.9321
Epoch 18/20
2379/2379 [==============================] - 53s 22ms/step - loss: 2.8677
Epoch 19/20
2379/2379 [==============================] - 53s 22ms/step - loss: 2.8758
Epoch 20/20
2379/2379 [==============================] - 53s 22ms/step - loss: 2.8350

Input text:
how to convert a datetime string back to datetime object?

Translated text:
 re ' ' ' ' ' ' ' end

True output text:
start datetime.strptime('2010-11-13 10:33:54.227806', '%Y-%m-%d %H:%M:%S.%f') end

Input text:
Averaging the values in a dictionary based on the key

Translated text:
 re ' ' ' ' ' ' ' end

True output text:
start [(i, sum(j) / len(j)) for i, j in list(d.items())] end

TOTAL TIME OF EXECUTION: 20.25 min

## Improvements
The results are always the same and are not correct. This could be improved considering some adjustments:
  - Don't work with words but character to character: The network could be getting the snippet as one word and trying to learn that
  - Mind the rewritten intent of json file for better matching
  - Change the dataset to another one with a reduced number of intentions with different versions of code outputs so the algorithm could be trained to learn how to solve specific problems(expressed in natural language)
  - Don't use GRUs but LSTMs

## License (MIT)
Copyright (c) 2018 by Magnus Erik Hvass Pedersen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
