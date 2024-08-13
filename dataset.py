import torch
import pandas as pd

class DataObject:
    def __init__(self, id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, species):
        self.id = id
        self.sepal_length_cm = sepal_length_cm
        self.sepal_width_cm = sepal_width_cm
        self.petal_length_cm = petal_length_cm
        self.petal_width_cm = petal_width_cm
        self.species = species
        self.network_output = [0, 0, 0, 0]
        if species == 'Iris-setosa':
            self.network_output = [1, 0, 0, 0]
        if species == 'Iris-versicolor':
            self.network_output = [0, 1, 0, 0]
        if species == 'Iris-virginica':
            self.network_output = [0, 0, 1, 0]
        self.network_output = torch.tensor(self.network_output)
        self.network_input = torch.tensor([self.sepal_length_cm, self.sepal_width_cm, self.petal_length_cm, self.petal_width_cm])

    def __repr__(self):
        return (f"DataObject(Id: {self.id}, SepalLengthCm: {self.sepal_length_cm}, "
                f"SepalWidthCm: {self.sepal_width_cm}, PetalLengthCm: {self.petal_length_cm}, "
                f"PetalWidthCm: {self.petal_width_cm}, Species: {self.species})")

class Dataset:
    def __init__(self, file_path):
        self.data_objects = self.load_data(file_path)

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        data_objects = []
        for _, row in df.iterrows():
            data_object = DataObject(
                id=row['Id'],
                sepal_length_cm=row['SepalLengthCm'],
                sepal_width_cm=row['SepalWidthCm'],
                petal_length_cm=row['PetalLengthCm'],
                petal_width_cm=row['PetalWidthCm'],
                species=row['Species']
            )
            data_objects.append(data_object)
        return data_objects

    def __repr__(self):
        return f"Dataset containing {len(self.data_objects)} objects"