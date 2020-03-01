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

import torch
from torch import nn
import numpy as np
import datetime
import matplotlib.pyplot as plt

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