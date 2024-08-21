import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_graph(title: str, x_name: str, y_name: str, data: List[Tuple[str, Tuple[List[float], List[float]]]]):
    plt.figure()
    for item in data:
        plt.plot(item[1][0], item[1][1], marker='o', label=item[0])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.ylim(top=.5, bottom=0)
    plt.title(f"{x_name} vs {y_name}")
    plt.grid(True)
    plt.legend()
    if not os.path.exists('graphs'):
        os.mkdir('graphs')
    plt.savefig(f'graphs/{title} {x_name} vs {y_name}.png')
    plt.close()

def get_plot_data_from_file(file_name: str):
    with open(f'results/{file_name}', 'r', encoding='utf-8') as f:
        data = json.load(f)

    y = [i for i in data['losses']]
    x = [i for i in range(0, len(data['losses']))]
    return f'a{data['alpha']} e{data['epochs']}', x, y

def create_graphs():
    data = []
    for file in os.listdir('results'):
        if not '.json' in file:
            continue

        title, x, y = get_plot_data_from_file(file)
        data.append((title, (x, y)))

    plot_graph('training stats', 'epoch', 'loss', data)

create_graphs()