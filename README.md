# End_to_End_Code_Generation
The goal of this projects is to be able to generate code snippets in Python from natural language decriptions following an End-to-End approach (parallel text corpora: intentions (NL) Vs. Snippets (Python)).

## Dataset

https://conala-corpus.github.io/

Code/Natural Language Challenge, a joint project of the Carnegie Mellon University NeuLab and STRUDEL Lab! This challenge was designed to test systems for generating program snippets from natural language. For example, if the input is:

> sort list x in reverse order

Then the system would be required to output

> x.sort(reverse=True)

Split into 2,379 training and 500 test examples. The json file format is shown as [follows](diagram.png):

>{
  
>  "question_id": 36875258,
  
> "intent": "copying one file's contents to another in python", 

> "rewritten_intent": "copy the content of file 'file.txt' to file 'file2.txt'", 

> "snippet": "shutil.copy('file.txt', 'file2.txt')", 

> }


## Framework
Tensorflow - Keras

## Description

  - Same implementation as Machine Translation. I am considering sequences of words and not of single characters. Following the structure shown in: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/21_Machine_Translation.ipynb based on

### Encoder
#### Tokenizer
##### Output

#### Embedding
#### Output

### Decoder
##### Output

## Results
### Parameters

### Intention-to-Snippet

## Improvements
  - Don't work with words but character to character
  - Mind the rewritten intent for better matching
  
