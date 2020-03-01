# ------------------------------------------------------------------------------
# MIT License
# 
# Copyright (c) 2020 Marcin Sielski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Initialize IPython
from IPython import get_ipython

# %% [markdown]
# # Hazyair Forecast
#
# Ths notebook implements simple neural network that will 
# do **time-series prediction**. Given some set of input data, it should be
# able to generate a prediction for the next time steps.

# %% [markdown]
# ## 1. Import resources.

import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
#from concurrent.futures import ThreadPoolExecutor
#from concurrent.futures import as_completed

get_ipython().magic('matplotlib inline')

# %% [markdown]
# ## 2. Read data from a file and show its visualization.

class DATA:

    """
    DATA class implements tools to read the data from a specified file,
    visualize data and rescale data.

    Attributes:
        filename: Name of the file.
    """

    def __init__(self, filename):

        """
        Reads data from the specified file.

        Args:
            filename: Name of the file.
        """

        # Read CSV data from file
        dataset = pd.read_csv(filename)
        # Drop empty rows where there is no temperature measurement
        dataset = dataset.dropna(subset=['DUST'])
        # Reset index
        dataset = dataset.reset_index(drop=True)
        # Extract temperature column
        self.data = dataset.iloc[:, 2:3].values
        self.time_steps = dataset.iloc[:, 0:1].values
        

    def rescale(self):

        """
        Rescale data to [0, 1].

        """

        # Create scaler
        self.sc = MinMaxScaler(feature_range=(0, 1))
        # Calculate mean and std and scale values
        data_scaled = self.sc.fit_transform(self.data)
        self.data = data_scaled.reshape((data_scaled.size, 1))
        

    def show(self):

        """
        Visualize data.
        """

        # Prepare figure
        plt.figure(figsize=(8, 5))

        # Display the data
        plt.plot(self.time_steps, self.data, 'g.', label='data') 
        
        # Display legend
        plt.legend(loc='best')

        # Show the figure
        plt.show()


    def generate_sequences(self, sequence_length):

        """
        Generate train, validation and test sequences from data of specified
        length.

        Args:
            sequence_length: Sequence length.

        Returns:
            train_sequences: Train sequences.
            valid_sequences: Validation sequences.
            test_sequences: Test sequences. 
        """

        # Skip missing data
        sequences = []
        for id_start in range(self.time_steps.size-sequence_length+1):
            if abs(self.time_steps[id_start+sequence_length-1] - \
                self.time_steps[id_start]) > 60*60*1000*(sequence_length-1) + \
                    60*1000:
                continue
            else:
                sequences.append((id_start, id_start + sequence_length))

        # Divide sequences into train, validation and test sequences 
        train_sequences, valid_sequences, _, _ = train_test_split(
            sequences, sequences, test_size=0.2, shuffle=False)
        valid_sequences, test_sequences, _, _ = train_test_split(
            valid_sequences, valid_sequences, test_size=0.2, shuffle=False)

        # Show sequences statistics
        print('Train sequences: '+str(len(train_sequences)) +
            ' ('+str(round(len(train_sequences)*100/len(sequences)))+'%)')
        print('Valid sequences: '+str(len(valid_sequences)) +
            ' ('+str(round(len(valid_sequences)*100/len(sequences)))+'%)')
        print('Tests sequences: '+str(len(test_sequences)) +
            ' ('+str(round(len(test_sequences)*100/len(sequences)))+'%)')
        print('Total sequences: '+str(len(sequences))+' (100%)')

        # Return sequences
        return train_sequences, valid_sequences, test_sequences


    #def __input_batch(self, sequence, sequence_length, input_size, i):
    #    return [self.data[j]+(self.data[j+1]-self.data[j])*i/sequence_length \
    #        for j in range(sequence[0],sequence[0]+input_size)]


    #def __output_batch(self, sequence, sequence_length, input_size, i):
    #    return [self.data[j]+(self.data[j+1]-self.data[j])*i/sequence_length \
    # for j in range(sequence[0]+input_size,sequence[1])]


    def get_batches(self, sequences, batch_size=1, sequence_length=1, \
        input_size=1):

        """
        Generates batches from sequences of defined input size and batch size.

        Args:
            sequences: Sequences used to generate batches.
            batch_size: Size of batch.
            sequence_length: Length of the sequence.
            input_size: Size of input part of the batch.


        Yields:
            input_batch: Batch for input.
            output_batch: Batch for output.
            time_steps: Corresponding time steps.
        """

        # Initialize variables
        counter = 1
        input_batch = []
        output_batch = []
        time_steps = []

        # Loop through sequences
        for sequence in sequences:
            if counter % batch_size == 0:
                for i in range(sequence_length):
                    if batch_size == 1:
                        input_batch.append([])
                        output_batch.append([])
                #with ThreadPoolExecutor(max_workers=4) as executor:
                #    futures_to_i = {executor.submit(self.__input_batch, sequence, sequence_length, input_size, i): i for i in range(sequence_length)}
                #    for future in as_completed(futures_to_i):
                #        i = futures_to_i[future]
                #        input_batch[i].append(future.result())

                #with ThreadPoolExecutor(max_workers=4) as executor:
                #    futures_to_i = {executor.submit(self.__output_batch, sequence, sequence_length, input_size, i): i for i in range(sequence_length)}
                #    for future in as_completed(futures_to_i):
                #        i = futures_to_i[future]
                #        output_batch[i].append(future.result())
               
                for i in range(sequence_length):
                    input_batch[i].append([self.data[j]+(self.data[j+1] \
                        -self.data[j])*i/sequence_length for j in \
                            range(sequence[0],sequence[0]+input_size)])
                    output_batch[i].append([self.data[j]+(self.data[j+1] \
                        -self.data[j])*i/sequence_length for j in \
                            range(sequence[0]+input_size,sequence[1])])
                time_steps.append(\
                    self.time_steps[sequence[0]:sequence[1]])
                
                # Yield input batch, output batch and corresponding time steps 
                yield input_batch, output_batch, time_steps

                # Empty batches
                input_batch = []
                output_batch = []
                time_steps = []
            else:
                for i in range(sequence_length):
                    if (counter - 1) % batch_size == 0:
                        input_batch.append([])
                        output_batch.append([])
                #with ThreadPoolExecutor(max_workers=4) as executor:
                #    futures_to_i = {executor.submit(self.__input_batch, sequence, sequence_length, input_size, i): i for i in range(sequence_length)}
                #    for future in as_completed(futures_to_i):
                #        i = futures_to_i[future]
                #        input_batch[i].append(future.result())

                #with ThreadPoolExecutor(max_workers=4) as executor:
                #    futures_to_i = {executor.submit(self.__output_batch, sequence, sequence_length, input_size, i): i for i in range(sequence_length)}
                #    for future in as_completed(futures_to_i):
                #        i = futures_to_i[future]
                #        output_batch[i].append(future.result())

                for i in range(sequence_length):
                    input_batch[i].append(\
                        [self.data[j]+(self.data[j+1]-self.data[j])*i/ \
                            sequence_length for j in \
                                range(sequence[0],sequence[0]+input_size)])
                    output_batch[i].append([self.data[j]+(self.data[j+1] \
                        -self.data[j])*i/sequence_length for j in \
                            range(sequence[0]+input_size,sequence[1])])
                time_steps.append(\
                    self.time_steps[sequence[0]:sequence[1]])

            # Increment counter
            counter += 1

# Read data from a file
data = DATA('data.csv')

# Visualize data
data.show()

# Rescale data
data.rescale()

# Visualize rescaled data
data.show()

# %% [markdown]
# ## 3. Define the model architecture and learn, validation and test processes.

from model import MODEL

# %% [markdown]
# ## 4. Create the model with hyperparameters and start start training process.

# Instantiate the model
network = \
    MODEL(input_size=12, output_size=4, hidden_size=32, num_layers=2, p=0.3)
print(network)

# Execute learning process
model_name = \
    network.learn(data=data, batch_size=16, sequence_length=16, lr=0.01, \
        num_epochs=100, validate_every=1, max_norm=4)

#%% [markdown]
# ## 5. Read best model from the file.

with open(model_name, 'rb') as f:

    # Load checkpoint
    checkpoint = torch.load(f)

# Instantiate the model
model = MODEL(checkpoint['input_size'], checkpoint['output_size'], \
    checkpoint['hidden_size'], checkpoint['num_layers'], checkpoint['p'])
model.load_state_dict(checkpoint['state_dict'])

#%% [markdown]
# ## 6. Test model on initial data.

model.test_first(data)

#%% [markdown]
# ## 6+X. Test model on next data.
model.test_next()


# %%
