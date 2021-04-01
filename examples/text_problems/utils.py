import os
import uuid
import shutil
from copy import deepcopy
from os.path import join, exists, split

def select_indices(labels, size, ratio=0.5):
    positive_size = size * ratio
    negative_size = size * (1 - ratio)
    positive_indices = []
    negative_indices = []
    selected_indices = []
    remaining_indices = []
    
    for index, label in enumerate(labels):
        if label == 1 and len(positive_indices) < positive_size:
            positive_indices.append(index)
        elif label == 0 and len(negative_indices) < negative_size:
            negative_indices.append(index)

        if len(positive_indices) == positive_size and len(negative_indices) == negative_size:
            break

    selected_indices = positive_indices + negative_indices

    for index in range(len(labels)):
        if index not in selected_indices:
            remaining_indices.append(index)
    
    return selected_indices, remaining_indices


def dataframe_to_d3mtext(dataframe, source_path, destination_path, text_column):
    dataframe_copy = deepcopy(dataframe)
    destination_path = join(destination_path, 'TRAIN')
    if exists(destination_path):
        shutil.rmtree(destination_path)
    os.makedirs(destination_path)
    
    problem_path = join(source_path, 'problem_TRAIN')
    new_problem_path = join(destination_path, 'problem_TRAIN')
    dataset_path = join(destination_path, 'dataset_TRAIN')
    media_path = join(destination_path, 'dataset_TRAIN', 'media')
    tables_path = join(destination_path, 'dataset_TRAIN', 'tables')
    
    if exists(new_problem_path):
        shutil.rmtree(new_problem_path)
    shutil.copytree(problem_path, new_problem_path)
    
    os.makedirs(dataset_path)
    datasetdoc_path = join(source_path, 'dataset_TRAIN', 'datasetDoc.json')
    new_datasetdoc_path = join(dataset_path, 'datasetDoc.json')
    shutil.copyfile(datasetdoc_path, new_datasetdoc_path)
        
    os.makedirs(media_path)
    os.makedirs(tables_path)
    
    def save_text(text):
        file_name = '%s.txt' % str(uuid.uuid4())
        file_path = join(media_path, file_name)
        with open(file_path, 'w') as fout:
            fout.write(text)
        return file_name

    dataframe_copy[text_column] = dataframe_copy[text_column].apply(save_text)
    dataframe_copy.to_csv(join(tables_path, 'learningData.csv'), index=False)
    
    return destination_path

def copy_dataset(source_path, destination_path):
    sufix = split(source_path)[-1]
    destination_path = join(destination_path, sufix)
    
    if exists(destination_path):
        shutil.rmtree(destination_path)
    shutil.copytree(source_path, destination_path)
    
    return destination_path
