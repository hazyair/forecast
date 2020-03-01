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
# # Hazyair Air Pollution Forecast
#
# Ths notebook implements simple neural network that will 
# do **time-series prediction**. Given some set of input data, it should be
# able to generate a prediction for the next time steps.

# %% [markdown]
# ## 1. Import resources.

import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import datetime
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

class MODEL(nn.Module):

    """
    MODEL class implements neural network architecture and learn, validation and
    test processes.

    Attributes:
        input_size: The number of expected features in the input.
        output_size: The number of expected features in the output.
        hidden_size: The number of features in the hidden state.
        num_layers: Number of recurrent layers.
        p: probability of an element to be zeroed.        
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers, p):
        
        """
        Defines neural network layers based on specified hyperparameters.

        Args:
            input_size: The number of expected features in the input.
            output_size: The number of expected features in the output.
            hidden_size: The number of features in the hidden state.
            num_layers: Number of recurrent layers.
            p: probability of an element to be zeroed.        
        """

        # Call constructor for superclass
        super(MODEL, self).__init__()

        # Initialize class members
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p = p
        self.cuda_available = torch.cuda.is_available()
        if (self.cuda_available):
            print('CUDA is available.') 
        self.data = None
        self.batches = None  
        self.test_sequences = None     

        # Define an network layers with specified parameters
        # batch_first means that the first dim of the input and output will be
        # the batch_size
        self.network = nn.LSTM(self.input_size, self.hidden_size, 
        self.num_layers, batch_first=True, dropout=p)
        self.dropout = nn.Dropout(p)

        # last, fully-connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

  
    def forward(self, x, hidden):

        """
        Overloads feed forward function from superclass and defines network
        architecture.

        Args:
            x: Input (batch_size, sequence_length, input_size).
            hidden: Hidden input (num_layers, batch_size, hidden_size).

        Returns:
            output: Output (batch_size, sequence_length, output_size).
            hidden: Hidden output (num_layers, batch_size, hidden_size).
        """

        batch_size = x.size(0)

        # Get network outputs
        # output (batch_size, sequence_length, hidden_size)
        # hidden (num_layers, batch_size, hidden_size)
        output, hidden = self.network(x, hidden)
        output = self.dropout(output)

        # Shape output to be (batch_size*sequence_length, hidden_size)
        #output = output.view(-1, self.hidden_size)

        # Get final output
        output = self.fc(output)

        # Return output and hidden
        return output, hidden


    def init_hidden(self, batch_size):

        '''
        Initializes hidden state

        Args:
            batch_size: Size of batch.

        Returns:
            hidden: Initial hidden layers.
        '''

        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (self.cuda_available):
            hidden = (weight.new(self.num_layers, batch_size,
            self.hidden_size).zero_().cuda(), weight.new(self.num_layers,
            batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size,
            self.hidden_size).zero_(), weight.new(self.num_layers, batch_size,
            self.hidden_size).zero_())
        
        # Return initial hidden layers
        return hidden


    def learn(self, data, batch_size, sequence_length, lr, num_epochs, \
        validate_every, max_norm):

        """
        Defines learning process for the model.

        Args:
            data: Input data.
            batch_size: Size of batch.
            sequence_length: Length of sequence.
            lr: Learning rate.
            num_epochs: Number of epochs.
            validate_every: Specifies when training process shall be validated.
            max_norm:

        Returns:
            model_name: Name of the model with the best validation results. 
        """

        # Initialize initial model name
        model_name = None

        # Generate train and validation sequences for the input data with
        # specified length
        train_sequences, valid_sequences, _ = \
            data.generate_sequences(self.input_size + self.output_size)

        # Define criterion and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Move computations to CUDA if available
        if(self.cuda_available):
            self.cuda()

        # Initialize the hidden state
        hidden = None

        # Initialize validation loss
        validation_loss = float('inf')

        # Set training mode
        self.train()

        # Train over defined number of epochs
        for epoch in range(num_epochs):

            # Initialize losses array
            losses = []

            # Initialize hidden layers
            hidden = self.init_hidden(batch_size)

            # Loop through the batches of data
            for x_train, y_train, _ in data.get_batches(\
                train_sequences, batch_size, sequence_length, self.input_size):

                # Reshape input and expected output data
                x = np.asarray(x_train).reshape((batch_size, \
                    sequence_length, self.input_size))
                y = np.asarray(y_train).reshape((batch_size, sequence_length, \
                    self.output_size))

                # Convert data into Tensors
                x_tensor = torch.Tensor(x)
                y_tensor = torch.Tensor(y)

                # Move Tensors to CUDA if available
                if (self.cuda_available):
                    x_tensor = x_tensor.cuda()
                    y_tensor = y_tensor.cuda()

                # Create new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                hidden = tuple([each.data for each in hidden])

                # Output from the network
                prediction, hidden = self(x_tensor, hidden)

                ## Representing Memory ##
                # make a new variable for hidden and detach the hidden state
                # from its history
                # this way, we don't backpropagate through the entire history
                #hidden = hidden.data

                # Calculate the loss
                loss = criterion(prediction, y_tensor)

                # Zero gradients
                optimizer.zero_grad()

                # Perform backprop and update weights
                loss.backward()

                # Clips gradient norm
                nn.utils.clip_grad_norm_(self.parameters(), max_norm)

                # Make step
                optimizer.step()

                # Accumulate losses
                losses.append(loss.item())

            # Validate 
            if epoch % validate_every == 0:

                # Initialize the model for inference. 
                self.eval()
                validation_losses = []

                #hidden = None
                hidden = self.init_hidden(batch_size)

                # Loop through validation sequences
                for x_train, y_train, _ in data.get_batches(valid_sequences, \
                    batch_size, sequence_length, self.input_size):

                    # Reshape input and expected output data
                    x = np.asarray(x_train).reshape((batch_size, \
                        sequence_length, self.input_size))
                    y = np.asarray(y_train).reshape((batch_size, \
                        sequence_length, self.output_size))

                    # convert data into Tensors
                    x_tensor = torch.Tensor(x)
                    y_tensor = torch.Tensor(y)

                    # Move Tensors to CUDA if available
                    if (self.cuda_available):
                        x_tensor = x_tensor.cuda()
                        y_tensor = y_tensor.cuda()

                    # Create new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    hidden = tuple([each.data for each in hidden])

                    # Output from the network
                    prediction, hidden = self(x_tensor, hidden)

                    #hidden = hidden.data

                    # Calculate the loss
                    loss = criterion(prediction, y_tensor)

                    # Accumulate losses
                    validation_losses.append(loss.item())

                # Show learning progress
                print("Epoch: {}/{}...".format(epoch + 1, num_epochs), \
                    "Loss: {:.4f}...".format(np.mean(losses)), 
                    "Val Loss: {:.4f}".format(np.mean(validation_losses)))

                # If current model is better then previous
                if (np.mean(validation_losses) < validation_loss):

                    # Store the smallest validation loss
                    validation_loss = np.mean(validation_losses)

                    # Create model name
                    model_name = 'forecast_' + str(epoch + 1) + '_' + \
                        str(validation_loss) + '_' + \
                            str(datetime.datetime.now()) + '.model'

                    # Prepare model with metadata
                    checkpoint = {'epoch': epoch + 1, \
                        'validation_loss': validation_loss, 
                        'input_size': self.input_size,
                        'output_size': self.output_size,
                        'hidden_size': self.hidden_size,
                        'num_layers': self.num_layers,
                        'p': self.p,
                        'state_dict': self.state_dict()
                        }

                    # Store model to the file
                    with open(model_name, 'wb') as f:
                        torch.save(checkpoint, f)

                # # Set training mode
                self.train()

        # Return the best model name
        return model_name

    
    def test_first(self, data):

        """
        Test model with initial data

        Args:
            data: Test data.
        """

        self.data = data
        self.batches = None
        _, _, self.test_sequences = \
            self.data.generate_sequences(self.input_size + self.output_size)
        self.test_next()


    def test_next(self):

        """
        Test model with next data.
        """

        # Set network to evaluation mode
        self.eval()

        # Check if batches were already generated
        if self.batches == None:

            # Generate batches from test sequences
            self.batches = self.data.get_batches(self.test_sequences, 1, 1, \
                self.input_size)
        try:

            # Get next batch of data
            x_test, y_test, time_step = next(self.batches)
        
        except:

            # If batches exhausted generate batches again 
            self.batches = self.data.get_batches(self.test_sequences, 1, 1, \
                self.input_size)

            # Get first batch of data
            x_test, y_test, time_step = next(self.batches)

        # Reshape input and expected output data
        x = np.asarray(x_test).reshape((1, 1, self.input_size))
        y = np.asarray(y_test).reshape((1, 1, self.output_size))

        # Move Tensor and model to CUDA if available
        x_tensor = torch.Tensor(x)
        if (self.cuda_available):
            x_tensor = x_tensor.cuda()
            self.cuda()

        # Output from the model
        prediction, _ = self(x_tensor, None)

        # Move predition Tensor to CPU if CUDA available
        if (self.cuda_available):
            prediction = prediction.cpu()

        # Display input data
        plt.plot(np.asarray(time_step).reshape((1, \
            self.input_size + self.output_size))[0][:self.input_size], \
                self.data.sc.inverse_transform(x[0].reshape(-1, 1)), 'g.', \
                    label='input data')

        # Display expected output data
        plt.plot(np.asarray(time_step).reshape((1, \
            self.input_size + self.output_size))[0][self.input_size:], \
                self.data.sc.inverse_transform(y[0].reshape(-1, 1)), 'r.', \
                    label='expected data')

        # Display prediction
        plt.plot(np.asarray(time_step).reshape((1, \
            self.input_size + self.output_size))[0][self.input_size:], \
                self.data.sc.inverse_transform(prediction.data.numpy() \
                    .flatten().reshape(-1, 1)), 'b.', \
                        label='predicted data')

        # Display legend
        plt.legend(loc='best')

        plt.show()

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
