import ast
import csv
import os
import sys
from pickle import dump
import pandas as pd
import numpy as np


output_folder = 'processed'
os.makedirs(output_folder, exist_ok=True)


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
    if dataset == 'SMD':
        dataset_folder = 'ServerMachineDataset'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test_label', filename, filename.strip('.txt'), dataset_folder)
    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = 'data'
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, 'test_label')
        os.makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels)
        print(dataset, 'test_label', labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(labels, file)


        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ['train', 'test']:
            concatenate_and_save(c)
    elif dataset == "SWaT":
        train_df = pd.read_csv("./input/SWaT_Dataset_Normal_v1.csv", delimiter=",")
        train_df_ = train_df.drop(["Timestamp", "Normal/Attack"], axis=1)
        for i in list(train_df_):
            train_df_[i] = train_df_[i].apply(lambda x: str(x).replace(",", "."))
        train_df_ = train_df_.astype(float)
        X_train = train_df_.values
        with open(os.path.join(output_folder, dataset + "_" + "train" + ".pkl"), "wb") as file:
            dump(X_train, file)

        test_df = pd.read_csv("./input/SWaT_Dataset_Attack_v0.csv", delimiter=";")
        test_df_ = test_df.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        for i in list(test_df_):
            test_df_[i] = test_df_[i].apply(lambda x: str(x).replace(",", "."))
        test_df_ = test_df_.astype(float)
        X_test = test_df_.values
        with open(os.path.join(output_folder, dataset + "_" + "test" + ".pkl"), "wb") as file:
            dump(X_test, file)

        y_test = []
        for index in test_df['Normal/Attack'].index:
            label = test_df['Normal/Attack'].get(index)
            if label == "Normal":
                y_test.append(0)
            elif label == "Attack":
                y_test.append(1)
        y_test = np.asarray(y_test)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(y_test, file)








if __name__ == '__main__':
    datasets = ['SMD', 'SMAP', 'MSL','SWaT']
    commands = sys.argv[1:]
    load = []

    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d)
    else:
        print("""
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMD', 'SMAP', 'MSL']
        """)
