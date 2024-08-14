import os
from neural import Network
from dataset import Dataset, DataObject

SAVE_FILE = 'model.safetensors'

dataset = Dataset('Iris.csv')

alpha = 1e-5
network = Network([
    4, 8, 16, 32, 64, 3
])

train_dataset = [(o.network_input, o.network_output) for o in dataset.data_objects]

if os.path.exists(SAVE_FILE):
    network.load(SAVE_FILE)
else:
    network.train(train_dataset, alpha)
    # network.save(SAVE_FILE)