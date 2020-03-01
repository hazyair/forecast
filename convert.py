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
# Import resources
import torch
from os import listdir
from model import MODEL
import onnx

#%%
# Find and load model

# Find the model
files = listdir('input')
model_name = files[0]

# Load the model
with open(model_name, 'rb') as f:

    # Load checkpoint
    checkpoint = torch.load(f)

# Instantiate the model
model = MODEL(checkpoint['input_size'], checkpoint['output_size'], \
    checkpoint['hidden_size'], checkpoint['num_layers'], checkpoint['p'])
model.load_state_dict(checkpoint['state_dict'])

# Move to CUDA if availablr
if torch.cuda.is_available():
    model.cuda()

# Configure evaluation mode
model.eval()

# %%
# Export model
torch.onnx.export(model, (torch.randn(1,1,12, device='cuda'), model.init_hidden(1)),'output/forecast.onnx', input_names=['input', 'h0', 'c0'], output_names=['output', 'hn', 'cn'])#, dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})

# %%
# Load model
onnx_model = onnx.load('output/forecast.onnx')

# Verify inputs and outputs
print(onnx_model.graph.input)
print(onnx_model.graph.output)


# %%
from openvino.inference_engine import IENetwork, IECore
import os

def load_to_IE(model_xml):
    global exec_net, input_shape
    ### Load the Inference Engine API
    plugin = IECore()

    ### Load IR files into their related class
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)


    ### Get the supported layers of the network
    supported_layers = plugin.query_network(network=net, device_name="MYRIAD")
    ### Check for any unsupported layers, and let the user
    ### know if anything is missing. Exit the program, if so.
    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        print("Unsupported layers found: {}".format(unsupported_layers))
        print("Check whether extensions are available to add to IECore.")
        return

    ### Load the network into the Inference Engine
    try:
        exec_net = plugin.load_network(net, "MYRIAD")
        print("Loading to VPU/CPU...")
    except:
        print('error')

    print("IR successfully loaded into Inference Engine.")


load_to_IE('output/forecast.xml')

# %%
