import csv

import matplotlib.pyplot as plt
import argparse


def print_loss_graph(input_file):
    with open(input_file, mode='r') as csv_file:
        training_history = list(csv_file.readline())
        val_history = list(csv_file.readline())

    plt.plot(training_history, '-bx')
    plt.plot(val_history, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig('loss.png', bbox_inches='tight')



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help="input file path")
    opt = parser.parse_args()
    print_loss_graph(opt.file)
