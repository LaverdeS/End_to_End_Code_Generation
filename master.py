"""Author: Sebastian Laverde Alfonso
    Purpose: End-to-End Code Generation"""

import os
import time
import json
import numpy as np
import tensorflow as tf

from pprint import pprint
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.models import Model

start = time.time()

with open('c:/Users/SEBASTIAN LAVERDE/Documents/GitHub/End_to_End_Code_Generation/conala-train.json') as f:
    data = json.load(f)

intents = []
snippets = []
j = 1

for i in data:
    intents.append(i['intent'])
    snippets.append(i['snippet'])

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality.
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/21_Machine_Translation.ipynb"""
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(), self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        # Note that the sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            # Reverse the token-sequences.
            self.tokens = [list(reversed(x)) for x in self.tokens]
        
            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]

        # Max number of tokens to use in all sequences.
        # We will pad / truncate all sequences to this length.
        # This is a compromise so we save a lot of memory and
        # only have to truncate maybe 5% of all the sequences.
        self.max_tokens = np.mean(self.num_tokens) + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)

        # Pad / truncate all token-sequences to the given length.
        # This creates a 2-dim numpy matrix that is easier to use.
        self.tokens_padded = pad_sequences(self.tokens, maxlen=self.max_tokens, padding=padding, truncating=truncating)

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token] for token in tokens if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """

        # Convert to tokens. Note that we assume there is only
        # a single text-string so we wrap it in a list.
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens, maxlen=self.max_tokens, padding='pre', truncating=truncating)

        return tokens

num_words = 10000
mark_start = "start "
mark_end = " end"
j = 0

for i in snippets:    
    i = mark_start + i + mark_end
    snippets[j] = i
    j = j + 1

print(snippets[0])

tokenizer_intents = TokenizerWrap(texts=intents, padding='pre', reverse=True, num_words=num_words)
tokenizer_snippets = TokenizerWrap(texts=snippets, padding='post', reverse=False, num_words=num_words)
tokens_intents = tokenizer_intents.tokens_padded
tokens_snippets = tokenizer_snippets.tokens_padded

print("Shape of tokenization of intents: ", tokens_intents.shape)
print("Shape of tokenization of snippets: ",tokens_snippets.shape)

token_start = tokenizer_snippets.word_index[mark_start.strip()]
print("token for start: ", token_start)

token_end = tokenizer_snippets.word_index[mark_end.strip()]
print("token for start: ", token_end)

print("intent: ", tokens_intents[5])
print("snippet: ", tokens_snippets[5])
print(tokenizer_snippets.tokens_to_string(tokens_snippets[5]))
print(snippets[5])

encoder_input_data = tokens_intents
decoder_input_data = tokens_snippets[:, :-1]
print("decoder input shape: ", decoder_input_data.shape)

decoder_output_data = tokens_snippets[:, 1:]
print("decoder output shape: ", decoder_output_data.shape)

print(decoder_input_data[5])
print(decoder_output_data[5])

print(tokenizer_snippets.tokens_to_string(decoder_input_data[5]))
print(tokenizer_snippets.tokens_to_string(decoder_output_data[5]))

encoder_input = Input(shape=(None, ), name='encoder_input')
embedding_size = 128
encoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_size, name='encoder_embedding')
state_size = 512

encoder_gru1 = GRU(state_size, name='encoder_gru1', return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2', return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3', return_sequences=False)

def connect_encoder():
    # Start the neural network with its input-layer.
    net = encoder_input
    
    # Connect the embedding-layer.
    net = encoder_embedding(net)

    # Connect all the GRU-layers.
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)

    # This is the output of the encoder.
    encoder_output = net
    
    return encoder_output

encoder_output = connect_encoder()

decoder_initial_state = Input(shape=(state_size,), name='decoder_initial_state')
decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_size, name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1', return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2', return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3', return_sequences=True)

decoder_dense = Dense(num_words, activation='linear', name='decoder_output')


def connect_decoder(initial_state):
    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU-layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output

decoder_output = connect_decoder(initial_state=encoder_output)
model_train = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

end = time.time()
print(end - start)