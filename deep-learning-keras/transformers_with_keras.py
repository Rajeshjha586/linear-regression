# !pip install tensorflow_cpu==2.17.1
# !pip install matplotlib==3.9.2

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from keras.layers import Layer
import warnings
warnings.simplefilter('ignore', FutureWarning)

# Step 1: Data Preparation
# Sample parallel sentences (English -> Spanish)
input_texts = [
    "Hello.", "How are you?", "I am learning machine translation.", "What is your name?", "I love programming."
]
target_texts = [
    "Hola.", "¿Cómo estás?", "Estoy aprendiendo traducción automática.", "¿Cuál es tu nombre?", "Me encanta programar."
]

target_texts = ["startseq " + x + " endseq" for x in target_texts]

# Next, we convert the text from the sentences to tokens and create a vocabulary
# Tokenization
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

output_tokenizer = Tokenizer()
output_tokenizer.fit_on_texts(target_texts)
output_sequences = output_tokenizer.texts_to_sequences(target_texts)

input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

# Now pad the corresponding sentences
# Padding: Ensures all sequences have the same length.
max_input_length = max([len(seq) for seq in input_sequences])
max_output_length = max([len(seq) for seq in output_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')

# Prepare the target data for training
decoder_input_data = output_sequences[:, :-1]
decoder_output_data = output_sequences[:, 1:]

# Convert to one-hot
decoder_output_data = np.array([np.eye(output_vocab_size)[seq] for seq in decoder_output_data])


# ============================================================
# Step 2: Self-Attention Layer
# ============================================================

# Self-attention allows the model to determine which words
# in a sequence are most important when processing a given word.
#
# Instead of looking at words independently, each word can
# "attend" to (focus on) other relevant words in the sentence.
#
# Example:
# Sentence: "The animal didn't cross the street because it was tired."
#
# To understand what "it" refers to, the model needs to pay
# attention to the word "animal". Self-attention helps learn
# these relationships automatically.


# ------------------------------------------------------------
# Where Self-Attention is Used
# ------------------------------------------------------------

# - Machine Translation
# - Text Summarization
# - Speech Recognition
# - Question Answering
# - Vision Transformers (ViT)
#
# In this project, self-attention is used for
# sequence-to-sequence text translation.


# ------------------------------------------------------------
# Core Idea
# ------------------------------------------------------------

# For every word in the input sequence:
#
# 1. Compare it with all other words.
# 2. Determine which words are important.
# 3. Create a new representation that contains
#    information from the most relevant words.
#
# This helps the model capture context and
# long-range dependencies in a sentence.


# ------------------------------------------------------------
# Query (Q), Key (K), and Value (V)
# ------------------------------------------------------------

# Self-attention creates three vectors for each word:
#
# Query (Q)
# -> What information this word is looking for.
#
# Key (K)
# -> What information this word can provide.
#
# Value (V)
# -> The actual information/content of the word.
#
# Example:
#
# Word: "it"
#
# Query:
#   "Which word does 'it' refer to?"
#
# Other words provide Keys and Values.
#
# The model compares the Query with all Keys
# to find the most relevant words.


# ------------------------------------------------------------
# Compute Attention Scores
# ------------------------------------------------------------

# Calculate similarity between:
#
# Query (Q)
# and
# Key (K)
#
# using dot-product attention.
#
# Formula:
#
# score = Q × Kᵀ
#
# Larger score:
# -> stronger relationship
#
# Smaller score:
# -> weaker relationship
#
# This allows every word to attend to every
# other word in the sequence.


# ------------------------------------------------------------
# Scale Attention Scores
# ------------------------------------------------------------

# Dot-product values can become very large,
# especially when vector dimensions increase.
#
# To stabilize training, scores are scaled by:
#
# 1 / sqrt(d_k)
#
# where:
# d_k = dimension of the Key vectors
#
# Formula:
#
# scaled_score = (Q × Kᵀ) / sqrt(d_k)


# ------------------------------------------------------------
# Apply Softmax
# ------------------------------------------------------------

# Softmax converts attention scores into probabilities.
#
# Properties:
# - Values range from 0 to 1
# - Sum of all values equals 1
#
# The resulting values are called:
#
# Attention Weights
#
# These weights indicate how much attention
# should be given to each word.


# ------------------------------------------------------------
# Generate Final Attention Output
# ------------------------------------------------------------

# Attention weights are multiplied by the
# Value (V) vectors.
#
# This creates a new representation for each word
# that contains information from the most relevant
# words in the sequence.
#
# As a result, the model gains contextual understanding
# rather than treating words independently.


# ============================================================
# Self-Attention Pipeline
# ============================================================

# Input Words
#       ↓
# Create Q, K, V vectors
#       ↓
# Compute Attention Scores (Q × Kᵀ)
#       ↓
# Scale Scores
#       ↓
# Apply Softmax
#       ↓
# Generate Attention Weights
#       ↓
# Weighted Sum of Values (V)
#       ↓
# Context-Aware Word Representations



# ============================================================
# Self-Attention Layer
# ============================================================

# This custom Self-Attention layer allows the model to learn
# which words/tokens in a sequence are most relevant when
# processing a particular word.
#
# The layer computes attention using three trainable matrices:
#
# Query (Q) -> What information a token is looking for
# Key (K)   -> What information a token contains
# Value (V) -> The actual information carried by the token
#
# The attention mechanism helps the model capture relationships
# between words, even when they are far apart in a sequence.


# ------------------------------------------------------------
# build() Method
# ------------------------------------------------------------

# The build() method is called automatically when the layer
# receives its first input.
#
# Here we initialize three trainable weight matrices:
#
# self.Wq -> Query weight matrix
# self.Wk -> Key weight matrix
# self.Wv -> Value weight matrix
#
# Shape:
#
# (feature_dim, feature_dim)
#
# Example:
#
# Input shape:
# (batch_size, seq_len, 128)
#
# Then:
#
# Wq = (128, 128)
# Wk = (128, 128)
# Wv = (128, 128)
#
# These matrices transform input features into
# Query, Key, and Value representations.


# ------------------------------------------------------------
# call() Method
# ------------------------------------------------------------

# The call() method defines the forward pass of the layer.
#
# Every time input data passes through the layer,
# the following steps are performed.


# ------------------------------------------------------------
# Step 1: Compute Query, Key, and Value
# ------------------------------------------------------------

# Input tensor:
#
# inputs
#
# Shape:
#
# (batch_size, seq_len, feature_dim)
#
# Generate:
#
# Q = inputs × Wq
# K = inputs × Wk
# V = inputs × Wv
#
# These transformed representations are used
# to compute attention.


# ------------------------------------------------------------
# Step 2: Compute Attention Scores
# ------------------------------------------------------------

# Calculate similarity between Query and Key vectors.
#
# Example:
#
# score = Q × Kᵀ
#
# Implemented using:
#
# K.batch_dot(Q, K, axes=[2, 2])
#
# Output shape:
#
# (batch_size, seq_len, seq_len)
#
# Meaning:
#
# Each token receives a score representing
# how strongly it should attend to every
# other token in the sequence.


# ------------------------------------------------------------
# Step 3: Scale Attention Scores
# ------------------------------------------------------------

# Dot-product values can become very large.
#
# To stabilize training, divide scores by:
#
# sqrt(feature_dim)
#
# Formula:
#
# scaled_scores = scores / sqrt(d_k)
#
# where:
#
# d_k = dimension of Key vectors


# ------------------------------------------------------------
# Step 4: Apply Softmax
# ------------------------------------------------------------

# Softmax converts attention scores into probabilities.
#
# Properties:
#
# - Values range between 0 and 1
# - Values sum to 1
#
# Result:
#
# Attention Weights
#
# These weights determine how much attention
# should be given to each token.


# ------------------------------------------------------------
# Step 5: Compute Final Attention Output
# ------------------------------------------------------------

# Multiply attention weights by Value vectors.
#
# Formula:
#
# output = Attention_Weights × V
#
# This creates a new representation for each token
# containing information from the most relevant tokens
# in the sequence.
#
# The resulting output becomes context-aware.


# ------------------------------------------------------------
# compute_output_shape() Method
# ------------------------------------------------------------

# This method defines the output tensor shape
# produced by the Self-Attention layer.
#
# For this implementation:
#
# Input Shape:
# (batch_size, seq_len, feature_dim)
#
# Output Shape:
# (batch_size, seq_len, feature_dim)
#
# The attention layer enriches the representation
# of each token but does NOT change the dimensions.


# ------------------------------------------------------------
# Why Output Shape Stays the Same
# ------------------------------------------------------------

# Self-attention changes the content of the vectors,
# not their size.
#
# Example:
#
# Before attention:
# (32, 50, 128)
#
# After attention:
# (32, 50, 128)
#
# Same dimensions
# Better contextual information


# ------------------------------------------------------------
# If Shape Needed to Change
# ------------------------------------------------------------

# If your attention layer performs:
#
# - Projection
# - Dimension reduction
# - Feature expansion
#
# then compute_output_shape() should be modified
# to return the new tensor dimensions.


# ============================================================
# Self-Attention Workflow
# ============================================================

# Input Sequence
#        ↓
# Create Query (Q)
# Create Key (K)
# Create Value (V)
#        ↓
# Compute Attention Scores (Q × Kᵀ)
#        ↓
# Scale Scores
#        ↓
# Apply Softmax
#        ↓
# Generate Attention Weights
#        ↓
# Multiply Weights with Values (V)
#        ↓
# Context-Aware Output
#        ↓
# Output Shape Same as Input Shape
#
# (batch_size, seq_len, feature_dim)


from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is a list: [q_shape, k_shape, v_shape]
        feature_dim = input_shape[0][-1]

        self.Wq = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wq'
        )

        self.Wk = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wk'
        )

        self.Wv = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wv'
        )

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Expect list: [query, key, value]
        q, k, v = inputs

        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)

        # Scaled dot-product attention
        scores = K.batch_dot(q, k, axes=[2, 2])
        dk = K.cast(K.shape(k)[-1], dtype=K.floatx())
        scores = scores / K.sqrt(dk)

        attention_weights = K.softmax(scores, axis=-1)
        output = K.batch_dot(attention_weights, v)

        return output

# Step 3: Model Architecture
# The model follows an Encoder-Decoder structure:
#
# Encoder:
# Takes input sentences (padded and tokenized).
# Uses an Embedding layer (word representations) + LSTM (to process sequences).
# The LSTMs are used as the help process variable-length input sentences and generate meaningful translations.
# Outputs context vectors (hidden & cell states).
#
# Attention Layer
# Applied to both the encoder and decoder outputs.
# Helps the decoder focus on relevant words during translation.
#
# Decoder
# Receives target sequences (shifted one step ahead).
# Uses an LSTM with encoder states as initial states.
# Applies self-attention for better learning.
# Uses a Dense layer (Softmax) to predict the next word.

from tensorflow.keras.layers import AdditiveAttention, Concatenate, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model

# Encoder
encoder_inputs = Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_output_length - 1,))
decoder_embedding = Embedding(output_vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention: decoder attends to encoder outputs
self_attention = SelfAttention()
attention_output = self_attention(
    [decoder_outputs, encoder_outputs, encoder_outputs]
)

# Combine decoder outputs with attention context
decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention_output])

# Final Dense layer
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat)

# Full Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Step 6: Train the Model
history_glorot_adam = model.fit([input_sequences, decoder_input_data], decoder_output_data, epochs=100, batch_size=16)

# Plotting training loss
import matplotlib.pyplot as plt
plt.plot(history_glorot_adam.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# let's train the model using "he_uniform" initializer instead of "glorot_uniform".
# Then, compare the training loss between model using "glorot_uniform" vs "he_uniform" initializers
# by plotting them using matplotlib

# Define the Self-Attention Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is a list: [q_shape, k_shape, v_shape]
        feature_dim = input_shape[0][-1]

        self.Wq = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='he_uniform',
            trainable=True,
            name='Wq'
        )

        self.Wk = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='he_uniform',
            trainable=True,
            name='Wk'
        )

        self.Wv = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='he_uniform',
            trainable=True,
            name='Wv'
        )

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Expect list: [query, key, value]
        q, k, v = inputs

        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)

        # Scaled dot-product attention
        scores = K.batch_dot(q, k, axes=[2, 2])
        dk = K.cast(K.shape(k)[-1], dtype=K.floatx())
        scores = scores / K.sqrt(dk)

        attention_weights = K.softmax(scores, axis=-1)
        output = K.batch_dot(attention_weights, v)

        return output


from tensorflow.keras.layers import AdditiveAttention, Concatenate, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model

# Encoder
encoder_inputs = Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_output_length - 1,))
decoder_embedding = Embedding(output_vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention: decoder attends to encoder outputs
self_attention = SelfAttention()
attention_output = self_attention(
    [decoder_outputs, encoder_outputs, encoder_outputs]
)

# Combine decoder outputs with attention context
decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention_output])

# Final Dense layer
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat)

# Full Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
# Step 6: Train the Model
history_he = model.fit([input_sequences, decoder_input_data], decoder_output_data, epochs=100, batch_size=16)

# Plotting training losses for glorot_uniform and he_uniform inititalizers
import matplotlib.pyplot as plt

plt.plot(history_glorot_adam.history['loss'], label="glorot_uniform", color='red')
plt.plot(history_he.history['loss'], label="he_uniform", color='blue')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# try to use adaptive gradient optimizer instead of adam.
# Then, plot and compare the results between adam and adaptive gradient optimizers
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])


#Step 6: Train the Model
history_adagrad = model.fit([input_sequences, decoder_input_data], decoder_output_data, epochs=100, batch_size=16)

#Plotting training losses for glorot_uniform and he_uniform inititalizers
import matplotlib.pyplot as plt
plt.plot(history_glorot_adam.history['loss'], label="adam", color='red')
plt.plot(history_adagrad.history['loss'], label="adagrad", color='blue')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()