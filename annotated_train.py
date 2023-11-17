#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import absl.app
import numpy as np
import progressbar
import shutil
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import time

tfv1.logging.set_verbosity({
    '0': tfv1.logging.DEBUG,
    '1': tfv1.logging.INFO,
    '2': tfv1.logging.WARN,
    '3': tfv1.logging.ERROR
}.get(DESIRED_LOG_LEVEL))

from datetime import datetime
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from .evaluate import evaluate
from six.moves import zip, range
from .util.config import Config, initialize_globals
from .util.checkpoints import load_or_init_graph_for_training, load_graph_for_evaluation, reload_best_checkpoint
from .util.evaluate_tools import save_samples_json
from .util.feeding import create_dataset, audio_to_features, audiofile_to_features
from .util.flags import create_flags, FLAGS
from .util.helpers import check_ctcdecoder_version, ExceptionBox
from .util.logging import create_progressbar, log_debug, log_error, log_info, log_progress, log_warn
from .util.io import open_remote, remove_remote, listdir_remote, is_remote_path, isdir_remote

check_ctcdecoder_version()

# Graph Creation
# ==============


# In this kitchen, some ingredients (data) are better handled by the main chef (CPU), 
# while others are faster to prepare with the help of the sous-chefs (GPUs). 
# This is because the main chef (CPU) is really good at doing lots of different tasks, 
# but not all at once, while the sous-chefs (GPUs) are great at chopping up a huge pile of vegetables 
# (processing lots of data) really fast, but only certain kinds of vegetables.

# Now, the variable_on_cpu function is like a rule in the kitchen that says, 
# “This specific ingredient (a variable in your program) should always be 
# prepared by the main chef (CPU), no matter what.”

def variable_on_cpu(name, shape, initializer):
    r"""
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_cpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device(Config.cpu_device):
        # Create or get apropos variable
        var = tfv1.get_variable(name=name, shape=shape, initializer=initializer)
    return var


# The create_overlapping_windows function is like a special 
# tool that helps the computer not just look at one data point at a 
# time but also pay attention to what's happening before and after that point. 
# It takes a chunk of sound data and creates these "windows" where each 
# window includes a bit of what came before and after each moment in the sound.

def create_overlapping_windows(batch_x):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * Config.n_context + 1
    num_channels = Config.n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                               .reshape(window_width, num_channels, window_width * num_channels), tf.float32) # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x


# dense function  is used to build a layer in a neural network. Here's what each part does:

# Setting Up: The function starts by preparing to add a new layer. 
# It uses things called 'weights' and 'biases', which are like the settings you adjust to add new features to your video game character.

# Combining Inputs: It then takes the data coming in (like the experiences your character has had so far in the game) and mixes it with the weights and biases. 
# This mixing is done using a math operation called a matrix multiplication (tf.matmul).

# Adding Non-Linearity (ReLU): If relu is true, it applies something called a ReLU (Rectified Linear Unit). 
# It's like setting a rule that if the character's energy goes below zero, just set it to zero 
# (since negative energy doesn’t make sense). This makes the model's learning process more effective.

# Layer Normalization: If layer_norm is true, it applies layer normalization, which is like making sure your character's features aren’t 
# too extreme in any direction. This helps keep the training of the AI model stable.

# Dropout: If there’s a dropout_rate, it randomly leaves out some of the data points. 
# This might sound bad, but it’s actually like giving your character random challenges 
# to make it more adaptable and prevent it from relying too much on certain features or experiences.

# Return the Output: Finally, the function returns the output, which is the new set of features (data) your character (AI model) has now, 
# after going through this layer.
def dense(name, x, units, dropout_rate=None, relu=True, layer_norm=False):
    with tfv1.variable_scope(name):
        bias = variable_on_cpu('bias', [units], tfv1.zeros_initializer())
        weights = variable_on_cpu('weights', [x.shape[-1], units], tfv1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

    output = tf.nn.bias_add(tf.matmul(x, weights), bias)

    if relu:
        output = tf.minimum(tf.nn.relu(output), FLAGS.relu_clip)

    if layer_norm:
        with tfv1.variable_scope(name):
            output = tf.contrib.layers.layer_norm(output)

    if dropout_rate is not None:
        output = tf.nn.dropout(output, rate=dropout_rate)

    return output


# It's specifically for a type of memory and learning 
# system called LSTM (Long Short-Term Memory). Here's how it works:

# 1. **Setting the Stage**: The function starts by setting up an LSTM cell. This cell is like a tiny brain 
# that remembers things and makes decisions based on both new information 
# (what's happening now) and what it remembers (past experiences).

# 2. **Forget Bias**: The `forget_bias` is set to 0, which is a bit like tuning how much the cell should 
# prioritize new information over old information. A forget bias of 0 means it doesn’t lean towards either.

# 3. **Reuse**: If `reuse` is true, the function will use an already existing LSTM cell if available, 
# kind of like reusing a tool you already have instead of getting a new one.

# 4. **Processing Data**: The cell takes in the current data (`x`, like the current scene in the game), the 
# length of the sequences it's looking at (`seq_length`, like how far back in the game's history it should consider), 
# and its previous state (`previous_state`, the memory of what happened before).

# 5. **Outputting Decisions**: The cell then processes all this information and comes up with two things: the `output` 
# (like the decision or action your character should take next) and `output_state` 
# (the updated memory after considering the new information).

def rnn_impl_lstmblockfusedcell(x, seq_length, previous_state, reuse):
    with tfv1.variable_scope('cudnn_lstm/rnn/multi_rnn_cell/cell_0'):
        fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(Config.n_cell_dim,
                                                    forget_bias=0,
                                                    reuse=reuse,
                                                    name='cudnn_compatible_lstm_cell')

        output, output_state = fw_cell(inputs=x,
                                       dtype=tf.float32,
                                       sequence_length=seq_length,
                                       initial_state=previous_state)

    return output, output_state


# In summary, `rnn_impl_cudnn_rnn` is like a high-tech, efficient memory tool for an AI model, 
# helping it to process sequences of data (like steps in a maze) and use that information to make decisions 
# or predictions. It's particularly optimized for running fast on specific types of computer hardware (NVIDIA GPUs).

# How it works:
# 1. **No Previous State**: First, the function makes sure it doesn't use any previous memory or state. 
# It's like starting fresh each time you run the function, without remembering past runs.

# 2. **Setting Up a Specialized Memory System**: The function uses a special type of memory system 
# called `CudnnLSTM` (a kind of Long Short-Term Memory system optimized to run on NVIDIA GPUs). 
# This system is really good at remembering and processing sequences of data.

# 3. **Singleton Instance**: The function has a quirky way of setting up this memory system. 
# Instead of creating a new one each time, it creates a single instance and reuses it. 
# Think of it like having a special notebook where you write down important notes to remember; 
# instead of getting a new notebook each time, you keep using the same one.

# 4. **Processing Data**: The memory system takes in the current data (`x`, like the current position in the maze), 
# and the sequence length (`seq_length`, which tells it how much of the past it should consider).

# 5. **Outputting Decisions**: After processing the data, it gives two things: the `output` 
# (like the next move in the maze) and `output_state` (updated memory based on the new information).

def rnn_impl_cudnn_rnn(x, seq_length, previous_state, _):
    assert previous_state is None # 'Passing previous state not supported with CuDNN backend'

    # Hack: CudnnLSTM works similarly to Keras layers in that when you instantiate
    # the object it creates the variables, and then you just call it several times
    # to enable variable re-use. Because all of our code is structure in an old
    # school TensorFlow structure where you can just call tf.get_variable again with
    # reuse=True to reuse variables, we can't easily make use of the object oriented
    # way CudnnLSTM is implemented, so we save a singleton instance in the function,
    # emulating a static function variable.
    if not rnn_impl_cudnn_rnn.cell:
        # Forward direction cell:
        fw_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                 num_units=Config.n_cell_dim,
                                                 input_mode='linear_input',
                                                 direction='unidirectional',
                                                 dtype=tf.float32)
        rnn_impl_cudnn_rnn.cell = fw_cell

    output, output_state = rnn_impl_cudnn_rnn.cell(inputs=x,
                                                   sequence_lengths=seq_length)

    return output, output_state

rnn_impl_cudnn_rnn.cell = None


# the `rnn_impl_static_rnn` function is like a guide for the AI model, 
# helping it to remember and process a sequence of events (like levels in a game) 
# and use that information to make decisions or predictions. It's particularly useful 
# for tasks where understanding the order and context of events is important, 
# like speech recognition or predicting the next note in a melody.

# Here's how it works:

# 1. **Setting Up a Brain Cell for Memory**: First, the function sets up a special type 
# of brain cell called an `LSTMCell` (Long Short-Term Memory Cell). This cell is like a 
# mini-brain that's really good at remembering sequences of events, which is crucial for 
# understanding things that happen over time (like a conversation or a melody).

# 2. **Preparing the Data**: The input data (`x`) is like the levels in your video game. T
# he function breaks this data into smaller pieces, just like breaking the game into individual levels. 
# This makes it easier for the LSTM cell to process each piece one at a time.

# 3. **Remembering and Processing**: The `static_rnn` part of the function is where the actual 
# processing happens. The LSTM cell takes in the data (the levels), along with any memories it 
# has from previous runs (`previous_state`), and processes them. It's like playing through each 
# level of the game, remembering what happened before to make better decisions.

# 4. **Outputting Decisions and Updated Memory**: After processing, the function gives two things: 
# `output` (the decisions or actions to take, based on the levels it has seen) and `output_state` 
# (the updated memories, after seeing the new levels).

# 5. **Combining the Outputs**: The `output` from each piece of data (each level) is then combined 
# together using `tf.concat(output, 0)`. This is like putting together all the experiences from 
# each level of the game into one complete story.

def rnn_impl_static_rnn(x, seq_length, previous_state, reuse):
    with tfv1.variable_scope('cudnn_lstm/rnn/multi_rnn_cell'):
        # Forward direction cell:
        fw_cell = tfv1.nn.rnn_cell.LSTMCell(Config.n_cell_dim,
                                            forget_bias=0,
                                            reuse=reuse,
                                            name='cudnn_compatible_lstm_cell')

        # Split rank N tensor into list of rank N-1 tensors
        x = [x[l] for l in range(x.shape[0])]

        output, output_state = tfv1.nn.static_rnn(cell=fw_cell,
                                                  inputs=x,
                                                  sequence_length=seq_length,
                                                  initial_state=previous_state,
                                                  dtype=tf.float32,
                                                  scope='cell_0')

        output = tf.concat(output, 0)

    return output, output_state


# create_model is a function for building a neural network suitable for speech recognition. 
# It integrates overlapping window creation for contextual information, multiple dense layers 
# for feature extraction and transformation, an RNN layer for capturing sequence patterns, 
# and final processing layers for output generation. 
def create_model(batch_x, seq_length, dropout, reuse=False, batch_size=None, previous_state=None, overlap=True, rnn_impl=rnn_impl_lstmblockfusedcell):
    layers = {}

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    if not batch_size:
        batch_size = tf.shape(input=batch_x)[0]

    # Create overlapping feature windows if needed
    if overlap:
        batch_x = create_overlapping_windows(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(a=batch_x, perm=[1, 0, 2, 3])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, Config.n_input + 2*Config.n_input*Config.n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    layers['input_reshaped'] = batch_x

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.
    layers['layer_1'] = layer_1 = dense('layer_1', batch_x, Config.n_hidden_1, dropout_rate=dropout[0], layer_norm=FLAGS.layer_norm)
    layers['layer_2'] = layer_2 = dense('layer_2', layer_1, Config.n_hidden_2, dropout_rate=dropout[1], layer_norm=FLAGS.layer_norm)
    layers['layer_3'] = layer_3 = dense('layer_3', layer_2, Config.n_hidden_3, dropout_rate=dropout[2], layer_norm=FLAGS.layer_norm)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_size, Config.n_hidden_3])

    # Run through parametrized RNN implementation, as we use different RNNs
    # for training and inference
    output, output_state = rnn_impl(layer_3, seq_length, previous_state, reuse)

    # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
    # to a tensor of shape [n_steps*batch_size, n_cell_dim]
    output = tf.reshape(output, [-1, Config.n_cell_dim])
    layers['rnn_output'] = output
    layers['rnn_output_state'] = output_state

    # Now we feed `output` to the fifth hidden layer with clipped RELU activation
    layers['layer_5'] = layer_5 = dense('layer_5', output, Config.n_hidden_5, dropout_rate=dropout[5], layer_norm=FLAGS.layer_norm)

    # Now we apply a final linear layer creating `n_classes` dimensional vectors, the logits.
    layers['layer_6'] = layer_6 = dense('layer_6', layer_5, Config.n_hidden_6, relu=False)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_size, Config.n_hidden_6], name='raw_logits')
    layers['raw_logits'] = layer_6

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, layers


# Accuracy and Loss
# =================

# In accord with 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# the loss function used by our network should be the CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).
# Conveniently, this loss function is implemented in TensorFlow.
# Thus, we can simply make use of this implementation to define our loss.
# This function's, `calculate_mean_edit_distance_and_loss()`, purpose is to check how well a model is performing 
# on a batch of data 
# by calculating the average "loss". The less the loss, the better the model's predictions. The function also 
# identifies any problematic files that lead to non-finite losses.

# 1. The function uses the input "iterator" to grab a batch of data, which includes file names, 
# inputs (`batch_x` and `batch_seq_len`), and expected outputs (`batch_y`).

# 2. Depending on a flag in the software, it chooses one of two types of RNNs (recurrent neural networks). 
# RNNs are a type of artificial neural network designed for recognizing patterns in sequences of data, such 
# as time series or text.

# 3. The function feeds the batch data into a model, resulting in predicted outputs called "logits".

# 4. Using a specific type of machine learning loss function called `ctc_loss`, the function then 
# calculates the "total loss," which measures how far off the model's predictions (the logits) are 
# from the expected outputs. A lower loss means the model's predictions are closer to what we expected, 
# and therefore, the model is better.

# 5. In case any of the files lead to non-finite losses (for example, if they are invalid in some way), 
# it pulls out those file names.

# 6. Finally, it averages the loss over all the data in the batch to get the "average loss". 

def calculate_mean_edit_distance_and_loss(iterator, dropout, reuse):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_filenames, (batch_x, batch_seq_len), batch_y = iterator.get_next()

    if FLAGS.train_cudnn:
        rnn_impl = rnn_impl_cudnn_rnn
    else:
        rnn_impl = rnn_impl_lstmblockfusedcell

    # Calculate the logits of the batch
    logits, _ = create_model(batch_x, batch_seq_len, dropout, reuse=reuse, rnn_impl=rnn_impl)

    # Compute the CTC loss using TensorFlow's `ctc_loss`
    total_loss = tfv1.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Check if any files lead to non finite loss
    non_finite_files = tf.gather(batch_filenames, tfv1.where(~tf.math.is_finite(total_loss)))

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(input_tensor=total_loss)

    # Finally we return the average loss
    return avg_loss, non_finite_files


# Adam Optimization
# =================

# In contrast to 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# in which 'Nesterov's Accelerated Gradient Descent'
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.


# This function is for creating an optimizer using the Adam Optimization algorithm, 
# which is a type of algorithm used in machine learning to handle data. 
# Let's break it down:

# - "AdamOptimizer" is a method from TensorFlow, a widely-used library in machine learning.
# - The parameters like learning_rate_var, beta1, beta2, epsilon are all used to tune the 
#   AdamOptimizer. Each one contributes to the behaviour of the optimizer and will affect 
#   how well it learns from the data.
# - Learning rate, the parameter that you provide when calling this function, determines how 
#   quickly or slowly the optimizer will adjust its internal values. If you set it too high, 
#   it might miss the best solution. If you set it too low, it could take forever to find it.
# - Beta1, Beta2 and Epsilon are numbers that control how the optimizer reacts to errors it 
#   encounters during training.
# - This function returns the newly created optimizer (which has your specified learning rate), 
#   ready to be used to train a machine learning model.

# Saying it in a less technical way, imagine you're playing a game where you're trying to find 
# a treasure by guessing its location. The learning rate is like how big your steps are. The 
# other parameters are like guides that help you adapt your steps based on the slopes or humps 
# you encounter on the land. This whole function is like equipping you with a strategy for how 
# you explore the area.
def create_optimizer(learning_rate_var):
    optimizer = tfv1.train.AdamOptimizer(learning_rate=learning_rate_var,
                                         beta1=FLAGS.beta1,
                                         beta2=FLAGS.beta2,
                                         epsilon=FLAGS.epsilon)
    return optimizer


# Towers
# ======

# In order to properly make use of multiple GPU's, one must introduce new abstractions,
# not present when using a single GPU, that facilitate the multi-GPU use case.
# In particular, one must introduce a means to isolate the inference and gradient
# calculations on the various GPU's.
# The abstraction we intoduce for this purpose is called a 'tower'.
# A tower is specified by two properties:
# * **Scope** - A scope, as provided by `tf.name_scope()`,
# is a means to isolate the operations within a tower.
# For example, all operations within 'tower 0' could have their name prefixed with `tower_0/`.
# * **Device** - A hardware device, as provided by `tf.device()`,
# on which all operations within the tower execute.
# For example, all operations of 'tower 0' could execute on the first GPU `tf.device('/gpu:0')`.





# GPUs are working together to speed up the processing time of training a ML model. 
# Each GPU works on a different, smaller batch of the overall training data, 
# which is called a "tower". 
# 
# This function's job is to calculate and keep track of 
# how well the ML model is doing on each tower, then combine it all to get an overall error measurement.

# Let's break down the main tasks this function performs:

# 1. It iterates through each GPU available (or each "tower") using a for loop.

# 2. On each tower, the function calculates two important measures: the 'avg_loss' and 'non_finite_files'. 
#    - 'avg_loss' is the average error rate of the ML model on that tower's data batch. 
#       Lower loss is better - it means the model is doing a better job predicting the output.
#    - 'non_finite_files' is a way to keep track of problematic data files that might cause errors 
#       because they contain 'non-finite' values (like infinity).

# 3. The function retains the average losses and gradients on each tower. These are important because they 
#   are used for adjusting the ML model to improve its performance.

# 4. Another important metrics calculated is the gradients. Gradients show the direction and rate of change 
#   in the loss - understanding gradients helps in adjusting the model to reduce loss. 

# 5. Finally, the function averages the loss across all towers to get the overall performance of the model 
#   (avg_loss_across_towers), and it concatenates root all 'non_finite_files' to keep track of all problematic files.

# To sum it up, this function is like a manager that oversees how well the ML model is performing on each worksite or 
# GPU (tower), then combines performance stats to give an overall view of the model's training progress. It also keeps 
# track of any problematic data that could affect the accuracy of the model's training.
def get_tower_results(iterator, optimizer, dropout_rates):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate and return the optimization gradients
    and the average loss across towers.
    '''
    # To calculate the mean of the losses
    tower_avg_losses = []

    # Tower gradients to return
    tower_gradients = []

    # Aggregate any non finite files in the batches
    tower_non_finite_files = []

    with tfv1.variable_scope(tfv1.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            device = Config.available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i):
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    avg_loss, non_finite_files = calculate_mean_edit_distance_and_loss(iterator, dropout_rates, reuse=i > 0)

                    # Allow for variables to be re-used by the next tower
                    tfv1.get_variable_scope().reuse_variables()

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)

                    tower_non_finite_files.append(non_finite_files)

    avg_loss_across_towers = tf.reduce_mean(input_tensor=tower_avg_losses, axis=0)
    tfv1.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

    all_non_finite_files = tf.concat(tower_non_finite_files, axis=0)

    # Return gradients and the average loss
    return tower_gradients, avg_loss_across_towers, all_non_finite_files


# This function's purpose is to average the gradients obtained from multiple GPUs when training a deep learning model. 
# Here's why this is needed:

# In deep learning, we usually use a concept called 'gradient descent' to learn from the data. 
# Gradients in this context might be imagined as small stepladders that the model follows to "descend" 
# to a solution that minimizes error. 

# When using multiple GPUs to compute these gradients from different portions of the data - 
# commonly referred to as mini-batches - we get several sets of gradients. This function works to 
# average out these received gradients to get one common gradient, which brings us to a more general 
# approximation of our model's error, instead of depending on specific GPUs' computations.

# Think of it this way: Imagine each GPU is a student working on a part of a group project. 
# Each student will make their own notes and insights. After all students are done, they get 
# together and average their findings. The outcome will be a more comprehensive understanding of the project. 
# This function facilitates this "group study" for GPUs.
def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a synchronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(Config.cpu_device):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []

            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(input_tensor=grad, axis=0)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads



# Logging
# =======

# This function, "log_variable", is used to log, or record, information about a "variable". 
# In the context of how this code would be used, a "variable" can be thought of as a number 
# that changes values as a computer program runs.
# To sum it up, this function logs vital statistics about a variable: its average value, 
# how much it varies, its min and max values, and how these values distribute. 

# Let's break down what the function is doing step by step:

# 1. It takes in a variable as an input and optionally a gradient. 
# The gradient is a math term that is often used in optimization processes to figure out 
# which way to change a variable to make the output of a function smaller or larger.

# 2. It cleans up the variable name a bit so it doesn't include characters like ':' 
# that could mess with the logging.

# 3. It calculates and logs the mean (the "average" value), the standard deviation 
# (a measure of how much the value of the variable is expected to vary), and the 
# minimum and maximum values of the variable.

# 4. It also creates and logs a histogram, which shows how frequently different values 
# of the variable are appearing. 

# 5. If a gradient was provided, it also creates and logs a histogram for this.
def log_variable(variable, gradient=None):
    r'''
    We introduce a function for logging a tensor variable's current state.
    It logs scalar values for the mean, standard deviation, minimum and maximum.
    Furthermore it logs a histogram of its state and (if given) of an optimization gradient.
    '''
    name = variable.name.replace(':', '_')
    mean = tf.reduce_mean(input_tensor=variable)
    tfv1.summary.scalar(name='%s/mean'   % name, tensor=mean)
    tfv1.summary.scalar(name='%s/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(input_tensor=tf.square(variable - mean))))
    tfv1.summary.scalar(name='%s/max'    % name, tensor=tf.reduce_max(input_tensor=variable))
    tfv1.summary.scalar(name='%s/min'    % name, tensor=tf.reduce_min(input_tensor=variable))
    tfv1.summary.histogram(name=name, values=variable)
    if gradient is not None:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is not None:
            tfv1.summary.histogram(name='%s/gradients' % name, values=grad_values)




# the purpose of log_grads_and_vars function is to loop through each gradient-variable pair and log them 
# for future analysis or debugging. It calls another function log_variable(), which presumably logs the 
# variable and its corresponding gradient somewhere (maybe a file or console). 

# By logging gradients and variables, we are essentially keeping track of these adjustments over time. 
# This can be useful to check if the learning process is going as expected, or if there are any issues 
# like the model not learning or the gradients becoming too large or too small (common issues in training 
# deep learning models). 

def log_grads_and_vars(grads_and_vars):
    r'''
    Let's also introduce a helper function for logging collections of gradient/variable tuples.
    '''
    for gradient, variable in grads_and_vars:
        log_variable(variable, gradient=gradient)


# This function, `train()`, is used to train a machine learning model.

# Here's a step-by-step breakdown of what it does:

# 1. A new training session begins and several variables and datasets are set up.
# 2. The input data is divided into two parts - one for training the model and the 
#   other to validate how well the model is learning. The latter part is also known 
#   as a "development set".
# 3. The function also sets up a system where it will drop out some neurons during 
#    training to prevent overfitting. This essentially means that not all neurons 
#    in the neural network are trained so that the model doesn't become too specialized 
#   towards the training data, hence it's still flexible for future, unseen data.
# 4. The model then begins training through various epochs. An epoch is one complete pass 
#   through the entire training data. Within each epoch, it calculates loss (the discrepancy 
#   between the predicted output and the actual output) and tries to minimize it, adjusting 
#   the weights of the neurons in the process.
# 5. Each time the model trains, it saves the current state of the machine learning model. 
#   This is known as a checkpoint. It does this in case something goes wrong, we can always 
#   go back to this checkpoint and start over without losing too much progress.
# 6. If the model's performance on the development set does not improve for several epochs, 
#   the model will either stop training altogether or reduce its learning rate and try again. 
#   This is to prevent wasteful computation for negligible or no gains.
# 7. In the end, the function provides the best validation loss (measurement of errors made 
#   for predictions on the development set) as a measure of how well the model has been trained. 
#   This metric guides further training or deployment of the model.

# This is an iterative process and this function iterates over the training dataset in chunks or 'batches'. 
def train():
    exception_box = ExceptionBox()

    if FLAGS.horovod:
        import horovod.tensorflow as hvd

    # Create training and validation datasets
    split_dataset = FLAGS.horovod

    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=FLAGS.train_batch_size,
                               epochs=FLAGS.epochs,
                               augmentations=Config.augmentations,
                               cache_path=FLAGS.feature_cache,
                               train_phase=True,
                               exception_box=exception_box,
                               process_ahead=Config.num_devices * FLAGS.train_batch_size * 2,
                               reverse=FLAGS.reverse_train,
                               limit=FLAGS.limit_train,
                               buffering=FLAGS.read_buffer,
                               split_dataset=split_dataset)

    iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(train_set),
                                                 tfv1.data.get_output_shapes(train_set),
                                                 output_classes=tfv1.data.get_output_classes(train_set))

    # Make initialization ops for switching between the two sets
    train_init_op = iterator.make_initializer(train_set)

    if FLAGS.dev_files:
        dev_sources = FLAGS.dev_files.split(',')
        dev_sets = [create_dataset([source],
                                   batch_size=FLAGS.dev_batch_size,
                                   train_phase=False,
                                   exception_box=exception_box,
                                   process_ahead=Config.num_devices * FLAGS.dev_batch_size * 2,
                                   reverse=FLAGS.reverse_dev,
                                   limit=FLAGS.limit_dev,
                                   buffering=FLAGS.read_buffer,
                                   split_dataset=split_dataset) for source in dev_sources]
        dev_init_ops = [iterator.make_initializer(dev_set) for dev_set in dev_sets]

    if FLAGS.metrics_files:
        metrics_sources = FLAGS.metrics_files.split(',')
        metrics_sets = [create_dataset([source],
                                       batch_size=FLAGS.dev_batch_size,
                                       train_phase=False,
                                       exception_box=exception_box,
                                       process_ahead=Config.num_devices * FLAGS.dev_batch_size * 2,
                                       reverse=FLAGS.reverse_dev,
                                       limit=FLAGS.limit_dev,
                                       buffering=FLAGS.read_buffer,
                                       split_dataset=split_dataset) for source in metrics_sources]
        metrics_init_ops = [iterator.make_initializer(metrics_set) for metrics_set in metrics_sets]

    # Dropout
    dropout_rates = [tfv1.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]
    dropout_feed_dict = {
        dropout_rates[0]: FLAGS.dropout_rate,
        dropout_rates[1]: FLAGS.dropout_rate2,
        dropout_rates[2]: FLAGS.dropout_rate3,
        dropout_rates[3]: FLAGS.dropout_rate4,
        dropout_rates[4]: FLAGS.dropout_rate5,
        dropout_rates[5]: FLAGS.dropout_rate6,
    }
    no_dropout_feed_dict = {
        rate: 0. for rate in dropout_rates
    }

    # Building the graph
    learning_rate_var = tfv1.get_variable('learning_rate', initializer=FLAGS.learning_rate, trainable=False)
    reduce_learning_rate_op = learning_rate_var.assign(tf.multiply(learning_rate_var, FLAGS.plateau_reduction))
    if FLAGS.horovod:
        # Effective batch size in synchronous distributed training is scaled by the number of workers. An increase in learning rate compensates for the increased batch size.
        optimizer = create_optimizer(learning_rate_var * hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer)
    else:
        optimizer = create_optimizer(learning_rate_var)

    # Enable mixed precision training
    if FLAGS.automatic_mixed_precision:
        log_info('Enabling automatic mixed precision training.')
        optimizer = tfv1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if FLAGS.horovod:
        loss, non_finite_files = calculate_mean_edit_distance_and_loss(iterator, dropout_rates, reuse=False)
        gradients = optimizer.compute_gradients(loss)

        tfv1.summary.scalar(name='step_loss', tensor=loss, collections=['step_summaries'])
        log_grads_and_vars(gradients)

        # global_step is automagically incremented by the optimizer
        global_step = tfv1.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step=global_step)
    else:
        gradients, loss, non_finite_files = get_tower_results(iterator, optimizer, dropout_rates)

        # Average tower gradients across GPUs
        avg_tower_gradients = average_gradients(gradients)
        log_grads_and_vars(avg_tower_gradients)

        # global_step is automagically incremented by the optimizer
        global_step = tfv1.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)

    # Summaries
    step_summaries_op = tfv1.summary.merge_all('step_summaries')
    step_summary_writers = {
        'train': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), max_queue=120),
        'dev': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'dev'), max_queue=120),
        'metrics': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'metrics'), max_queue=120),
    }

    human_readable_set_names = {
        'train': 'Training',
        'dev': 'Validation',
        'metrics': 'Metrics',
    }

    # Checkpointing
    if Config.is_master_process:
        checkpoint_saver = tfv1.train.Saver(max_to_keep=FLAGS.max_to_keep)
        checkpoint_path = os.path.join(FLAGS.save_checkpoint_dir, 'train')

        best_dev_saver = tfv1.train.Saver(max_to_keep=1)
        best_dev_path = os.path.join(FLAGS.save_checkpoint_dir, 'best_dev')

        # Save flags next to checkpoints
        if not is_remote_path(FLAGS.save_checkpoint_dir):
            os.makedirs(FLAGS.save_checkpoint_dir, exist_ok=True)
        flags_file = os.path.join(FLAGS.save_checkpoint_dir, 'flags.txt')
        with open_remote(flags_file, 'w') as fout:
            fout.write(FLAGS.flags_into_string())

    if FLAGS.horovod:
        bcast = hvd.broadcast_global_variables(0)

    with tfv1.Session(config=Config.session_config) as session:
        log_debug('Session opened.')

        # Prevent further graph changes
        tfv1.get_default_graph().finalize()

        # Load checkpoint or initialize variables
        load_or_init_graph_for_training(session)
        if FLAGS.horovod:
            bcast.run()










# In short, this function tests a model with certain files and saves the results in a 
# specified file. If you don't specify an output file, it will execute the tests but 
# won't save the results anywhere.

# Let's break it down: 
# - `samples = evaluate(FLAGS.test_files.split(','), create_model)`
# essentially saying: "evaluate these test files using the model we've created".

# - `if FLAGS.test_output_file:`
# Next we have an if-statement that asks: "do we have a specified output file?" 
# The `FLAGS.test_output_file` is probably a string of the output file path 
# where you want to write the test results.

# - `save_samples_json(samples, FLAGS.test_output_file)`
# If we do have a specified output file, save the samples we've made into the specified test output file in JSON format.

def test():
    samples = evaluate(FLAGS.test_files.split(','), create_model)
    if FLAGS.test_output_file:
        save_samples_json(samples, FLAGS.test_output_file)



# This function, `create_inference_graph`, is used to set up a deep learning model for 
# making predictions, factoring in a number of different potential 
# configurations and outputs.

# Here's a simple breakdown of what it's doing.
# 1. It first sets up inputs called placeholders. These are actually inputs 
# for your model when it starts running. There are placeholders for raw audio input, 
# the feature-extracted performance of the audio (MFCC: Mel Frequency Cepstral Coefficients), 
# the sequence length of the audio, and the previous states of the model's 
# long short-term memory (LSTM) cell (if the batch size is greater than zero).

# 2. It then determines the type of Recurrent Neural Network (RNN) implementation to use
#  based on whether the TensorFlow Lite flag is set or not — TensorFlow Lite being a more 
# lightweight version of TensorFlow designed for mobile and edge devices.

# 3. Next, it builds the neural network model using `create_model` function 

# 4. After the model is created, it runs the output through a softmax function to convert 
# it into a probability distribution. This is specifically useful in getting the final 
# predicted output from our model. 

# 5. Depending on the batch size and whether the model will be used on TensorFlow Lite, 
# it sets up additional inputs and outputs and adjusts the shape of the tensors.

# 6. Finally, it returns a dictionary with all necessary inputs and outputs for further processing.

def create_inference_graph(batch_size=1, n_steps=16, tflite=False):
    batch_size = batch_size if batch_size > 0 else None

    # Create feature computation graph
    input_samples = tfv1.placeholder(tf.float32, [Config.audio_window_samples], 'input_samples')
    samples = tf.expand_dims(input_samples, -1)
    mfccs, _ = audio_to_features(samples, FLAGS.audio_sample_rate)
    mfccs = tf.identity(mfccs, name='mfccs')

    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    # This shape is read by the native_client in DS_CreateModel to know the
    # value of n_steps, n_context and n_input. Make sure you update the code
    # there if this shape is changed.
    input_tensor = tfv1.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2 * Config.n_context + 1, Config.n_input], name='input_node')
    seq_length = tfv1.placeholder(tf.int32, [batch_size], name='input_lengths')

    if batch_size <= 0:
        # no state management since n_step is expected to be dynamic too (see below)
        previous_state = None
    else:
        previous_state_c = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_c')
        previous_state_h = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_h')

        previous_state = tf.nn.rnn_cell.LSTMStateTuple(previous_state_c, previous_state_h)

    # One rate per layer
    no_dropout = [None] * 6

    if tflite:
        rnn_impl = rnn_impl_static_rnn
    else:
        rnn_impl = rnn_impl_lstmblockfusedcell

    logits, layers = create_model(batch_x=input_tensor,
                                  batch_size=batch_size,
                                  seq_length=seq_length if not FLAGS.export_tflite else None,
                                  dropout=no_dropout,
                                  previous_state=previous_state,
                                  overlap=False,
                                  rnn_impl=rnn_impl)

    # TF Lite runtime will check that input dimensions are 1, 2 or 4
    # by default we get 3, the middle one being batch_size which is forced to
    # one on inference graph, so remove that dimension
    if tflite:
        logits = tf.squeeze(logits, [1])

    # Apply softmax for CTC decoder
    probs = tf.nn.softmax(logits, name='logits')

    if batch_size <= 0:
        if tflite:
            raise NotImplementedError('dynamic batch_size does not support tflite nor streaming')
        if n_steps > 0:
            raise NotImplementedError('dynamic batch_size expect n_steps to be dynamic too')
        return (
            {
                'input': input_tensor,
                'input_lengths': seq_length,
            },
            {
                'outputs': probs,
            },
            layers
        )

    new_state_c, new_state_h = layers['rnn_output_state']
    new_state_c = tf.identity(new_state_c, name='new_state_c')
    new_state_h = tf.identity(new_state_h, name='new_state_h')

    inputs = {
        'input': input_tensor,
        'previous_state_c': previous_state_c,
        'previous_state_h': previous_state_h,
        'input_samples': input_samples,
    }

    if not FLAGS.export_tflite:
        inputs['input_lengths'] = seq_length

    outputs = {
        'outputs': probs,
        'new_state_c': new_state_c,
        'new_state_h': new_state_h,
        'mfccs': mfccs,

        # Expose internal layers for downstream applications
        'layer_3': layers['layer_3'],
        'layer_5': layers['layer_5']
    }

    return inputs, outputs, layers


# This function is used to open and read a file in Python
def file_relative_read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()



# The `export()` function  takes an advanced model, simplifies it, 
# freezes it so it can't be changed and neatly packs it away with a tag 
# explaining what's inside.

# Here's a quick breakdown: 
# 1. The function first restores the trained variables from an existing model and 
#   builds an inference graph, which is a graph that lets us make predictions.

# 2. Then it assigns some of the model's metadata (like version and sample rate) to 
#   variables. Metadata is like a model's 'about' details, that explains what the model 
#   is, versions, additional functionalities and so on. 

# 3. It also finalizes the current state of the graph so that no further changes can be made. 
#   This sets our packing in stone, to make sure nothing gets mixed up.

# 4. After some preprocessing, the function 'freezes' the model, which like literal freezing, 
#   puts the model into a state where it can't be further modified. This is done by converting 
#   the model's variables into constants.

# 5. The function then writes the frozen model to a local file so we can access it later.

# 6. Finally, the function saves all important metadata (the model's about details we mentioned earlier) 
#   in a file and prints where these files are located.

def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')

    inputs, outputs, _ = create_inference_graph(batch_size=FLAGS.export_batch_size, n_steps=FLAGS.n_steps, tflite=FLAGS.export_tflite)

    graph_version = int(file_relative_read('GRAPH_VERSION').strip())
    assert graph_version > 0

    outputs['metadata_version'] = tf.constant([graph_version], name='metadata_version')
    outputs['metadata_sample_rate'] = tf.constant([FLAGS.audio_sample_rate], name='metadata_sample_rate')
    outputs['metadata_feature_win_len'] = tf.constant([FLAGS.feature_win_len], name='metadata_feature_win_len')
    outputs['metadata_feature_win_step'] = tf.constant([FLAGS.feature_win_step], name='metadata_feature_win_step')
    outputs['metadata_beam_width'] = tf.constant([FLAGS.export_beam_width], name='metadata_beam_width')
    outputs['metadata_alphabet'] = tf.constant([Config.alphabet.Serialize()], name='metadata_alphabet')

    if FLAGS.export_language:
        outputs['metadata_language'] = tf.constant([FLAGS.export_language.encode('utf-8')], name='metadata_language')

    # Prevent further graph changes
    tfv1.get_default_graph().finalize()

    output_names_tensors = [tensor.op.name for tensor in outputs.values() if isinstance(tensor, tf.Tensor)]
    output_names_ops = [op.name for op in outputs.values() if isinstance(op, tf.Operation)]
    output_names = output_names_tensors + output_names_ops

    with tf.Session() as session:
        # Restore variables from checkpoint
        load_graph_for_evaluation(session)

        output_filename = FLAGS.export_file_name + '.pb'
        if FLAGS.remove_export:
            if isdir_remote(FLAGS.export_dir):
                log_info('Removing old export')
                remove_remote(FLAGS.export_dir)

        output_graph_path = os.path.join(FLAGS.export_dir, output_filename)

        if not is_remote_path(FLAGS.export_dir) and not os.path.isdir(FLAGS.export_dir):
            os.makedirs(FLAGS.export_dir)

        frozen_graph = tfv1.graph_util.convert_variables_to_constants(
            sess=session,
            input_graph_def=tfv1.get_default_graph().as_graph_def(),
            output_node_names=output_names)

        frozen_graph = tfv1.graph_util.extract_sub_graph(
            graph_def=frozen_graph,
            dest_nodes=output_names)

        if not FLAGS.export_tflite:
            with open_remote(output_graph_path, 'wb') as fout:
                fout.write(frozen_graph.SerializeToString())
        else:
            output_tflite_path = os.path.join(FLAGS.export_dir, output_filename.replace('.pb', '.tflite'))

            converter = tf.lite.TFLiteConverter(frozen_graph, input_tensors=inputs.values(), output_tensors=outputs.values())
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # AudioSpectrogram and Mfcc ops are custom but have built-in kernels in TFLite
            converter.allow_custom_ops = True
            tflite_model = converter.convert()

            with open_remote(output_tflite_path, 'wb') as fout:
                fout.write(tflite_model)

        log_info('Models exported at %s' % (FLAGS.export_dir))

    metadata_fname = os.path.join(FLAGS.export_dir, '{}_{}_{}.md'.format(
        FLAGS.export_author_id,
        FLAGS.export_model_name,
        FLAGS.export_model_version))

    model_runtime = 'tflite' if FLAGS.export_tflite else 'tensorflow'
    with open_remote(metadata_fname, 'w') as f:
        f.write('---\n')
        f.write('author: {}\n'.format(FLAGS.export_author_id))
        f.write('model_name: {}\n'.format(FLAGS.export_model_name))
        f.write('model_version: {}\n'.format(FLAGS.export_model_version))
        f.write('contact_info: {}\n'.format(FLAGS.export_contact_info))
        f.write('license: {}\n'.format(FLAGS.export_license))
        f.write('language: {}\n'.format(FLAGS.export_language))
        f.write('runtime: {}\n'.format(model_runtime))
        f.write('min_ds_version: {}\n'.format(FLAGS.export_min_ds_version))
        f.write('max_ds_version: {}\n'.format(FLAGS.export_max_ds_version))
        f.write('acoustic_model_url: <replace this with a publicly available URL of the acoustic model>\n')
        f.write('scorer_url: <replace this with a publicly available URL of the scorer, if present>\n')
        f.write('---\n')
        f.write('{}\n'.format(FLAGS.export_description))

    log_info('Model metadata file saved to {}. Before submitting the exported model for publishing make sure all information in the metadata file is correct, and complete the URL fields.'.format(metadata_fname))



# This function creates a zip file of the specified directory.
def package_zip():
    # --export_dir path/to/export/LANG_CODE/ => path/to/export/LANG_CODE.zip
    export_dir = os.path.join(os.path.abspath(FLAGS.export_dir), '') # Force ending '/'
    if is_remote_path(export_dir):
        log_error("Cannot package remote path zip %s. Please do this manually." % export_dir)
        return

    zip_filename = os.path.dirname(export_dir)
    
    shutil.copy(FLAGS.scorer_path, export_dir)

    archive = shutil.make_archive(zip_filename, 'zip', export_dir)
    log_info('Exported packaged model {}'.format(archive))


# This function `do_single_file_inference` is used to run inference on a single audio file. 
# Basically, it means it processes the audio file to guess or predict the spoken words in it. 
# Here's how it works:

# 1. It first sets up a TensorFlow session. TensorFlow is a framework developed by Google for 
# creating machine learning models. A session allows running TensorFlow operations.

# 2. It creates an inference graph, which is a computational representation of the model 
# used for predicting outcomes.

# 3. Then it loads the previously trained model (learning from past data) to make predictions on new data.

# 4. It converts the audio file to features. This means it changes the audio file 
# into a format or representation that the machine learning model can understand and process.

# 5. It creates some zero-filled arrays of certain dimensions and adds additional dimensions 
# (using `tf.expand_dims`) to the features.

# 6. It evaluates these features by running the model using the TensorFlow session.

# 7. Then it gets probabilities for every possible prediction. Each probability represents 
# how likely the model thinks each possible outcome is.

# 8. If a scorer (another model used to refine or adjust the probabilities or predictions) 
# is provided, it uses the scorer to get the final outcome probabilities.

# 9. It then uses a decoder to convert the probabilities to the final predictions, 
# the words that the model thinks were spoken in the input audio file.

# 10. The function finally prints out the most likely prediction or the best guess of 
# the spoken words in the audio file.

# So, in short, this function feeds an audio file into a trained model, which then predicts 
# the spoken words in the audio file.
def do_single_file_inference(input_file_path):
    with tfv1.Session(config=Config.session_config) as session:
        inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=-1)

        # Restore variables from training checkpoint
        load_graph_for_evaluation(session)

        features, features_len = audiofile_to_features(input_file_path)
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])

        # Add batch dimension
        features = tf.expand_dims(features, 0)
        features_len = tf.expand_dims(features_len, 0)

        # Evaluate
        features = create_overlapping_windows(features).eval(session=session)
        features_len = features_len.eval(session=session)

        probs = outputs['outputs'].eval(feed_dict={
            inputs['input']: features,
            inputs['input_lengths']: features_len,
            inputs['previous_state_c']: previous_state_c,
            inputs['previous_state_h']: previous_state_h,
        }, session=session)

        probs = np.squeeze(probs)

        if FLAGS.scorer_path:
            scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            FLAGS.scorer_path, Config.alphabet)
        else:
            scorer = None
        decoded = ctc_beam_search_decoder(probs, Config.alphabet, FLAGS.beam_width,
                                          scorer=scorer, cutoff_prob=FLAGS.cutoff_prob,
                                          cutoff_top_n=FLAGS.cutoff_top_n)
        # Print highest probability result
        print(decoded[0][1])



# This function, `early_training_checks`, has two main tasks:

# 1. The first part of the function checks a scorer file. This is 
#   likely part of a language model for a speech recognition application. 
#   The check ensures that a scorer file was provided by looking for the 
#   specified path (`FLAGS.scorer_path`). It creates a `Scorer` object 
#   containing the lambda (`lm_alpha`), beta (`lm_beta`), scorer path and the 
#   alphabet from the configuration file (`Config.alphabet`), then immediately 
#   deletes it. Essentially, it's checking if the scorer file exists and is usable.

# 2. The second part issues a warning if the user made a potential mistake in 
# command line instructions. If the user specified different directories for 
# loading a previous checkpoint (starting point for training the model) and 
# saving the newly trained model, but asked the process to train and test 
# within the same run, a warning message is issued. This could lead to 
# inconsistencies between the trained and tested models, because even if the 
# model is trained in this run, the testing phase would still use the old model 
# from `load_checkpoint_dir` instead of the newly trained one from `save_checkpoint_dir`. 
# The warning message advises the user to either train and test separately while 
# specifying the correct directories, or to use the same location for both loading
#  and saving if they wish to train and test within a single run.
def early_training_checks():
    # Check for proper scorer early
    if FLAGS.scorer_path:
        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        FLAGS.scorer_path, Config.alphabet)
        del scorer

    if FLAGS.train_files and FLAGS.test_files and FLAGS.load_checkpoint_dir != FLAGS.save_checkpoint_dir:
        log_warn('WARNING: You specified different values for --load_checkpoint_dir '
                 'and --save_checkpoint_dir, but you are running training and testing '
                 'in a single invocation. The testing step will respect --load_checkpoint_dir, '
                 'and thus WILL NOT TEST THE CHECKPOINT CREATED BY THE TRAINING STEP. '
                 'Train and test in two separate invocations, specifying the correct '
                 '--load_checkpoint_dir in both cases, or use the same location '
                 'for loading and saving.')



# main function that initiates machine learning model 
# training and testing process for a certain project. 

# At first, it initializes some global variables and conducts some preliminary checks. 

# Then, it verifies if there are training files available. If so, it resets the default 
# graph that TensorFlow uses for computations. It sets a random seed for ensuring the 
# consistency of the results across multiple runs and initiates the training.

# After training, if the current process is the master process, it checks if there are 
# test files available. If so, it again resets the graph and starts the testing.

# Next, if an export directory is specified and the export is not supposed to be packed 
# into a zip file, it exports the trained model. If the export is supposed to be in a zip 
# format, it sets an additional flag, checks whether the export directory is empty. If 
# the directory is not empty, it sends an error message and terminates the program. If 
# the directory is empty, it exports the model and packs it into a zip file.

# Finally, if a command line flag "one_shot_infer" is turned on, it again resets the graph 
# and performs a single file inference i.e., it uses the trained model to make predictions 
# on a single input file.
def main(_):
    initialize_globals()
    early_training_checks()

    if FLAGS.train_files:
        tfv1.reset_default_graph()
        tfv1.set_random_seed(FLAGS.random_seed)

        train()

    if Config.is_master_process:
        if FLAGS.test_files:
            tfv1.reset_default_graph()
            test()

        if FLAGS.export_dir and not FLAGS.export_zip:
            tfv1.reset_default_graph()
            export()

        if FLAGS.export_zip:
            tfv1.reset_default_graph()
            FLAGS.export_tflite = True

            if listdir_remote(FLAGS.export_dir):
                log_error('Directory {} is not empty, please fix this.'.format(FLAGS.export_dir))
                sys.exit(1)

            export()
            package_zip()

        if FLAGS.one_shot_infer:
            tfv1.reset_default_graph()
            do_single_file_inference(FLAGS.one_shot_infer)



# This function, named "run_script," is essentially a simple form of program control. 
# It's performing two main tasks:

# 1. `create_flags()` looks like it is setting up some configurations or settings, 
# like the options or modes that a program can run under. If we compare it to a video game, 
# flags would be the settings such as difficulty level or sound on/off that you set before 
# you start playing.

# 2. `absl.app.run(main)` is the main functionality of your code that performs the main 
# action in your program. Using the video game example again, this would be like actually 
# playing the game once you've set up your preferences.

# `run_script()` is the function that starts all this process, like the "play" button once you 
# are ready to start the game.
def run_script():
    create_flags()
    absl.app.run(main)

if __name__ == '__main__':
    run_script()
