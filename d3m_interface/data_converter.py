import os
import json
import shutil
import datamart_profiler
import pandas as pd
from datamart_materialize.d3m import D3mWriter
from os.path import join, exists


def is_d3m_format(dataset, suffix):
    if isinstance(dataset, str) and exists(join(dataset, 'dataset_%s/datasetDoc.json' % suffix)):
        return True

    return False


def convert_d3m_format(dataset, output_folder, problem_config, suffix):
    print('Reiceving a raw dataset, converting to D3M format')
    dataset_folder = join(output_folder, 'temp', 'dataset_d3mformat', suffix, 'dataset_%s' % suffix)
    problem_folder = join(output_folder, 'temp', 'dataset_d3mformat', suffix, 'problem_%s' % suffix)
    create_d3m_problem(dataset, problem_folder, problem_config)
    create_d3m_dataset(dataset, dataset_folder)

    return join(output_folder, 'temp', 'dataset_d3mformat', suffix)


def create_d3m_dataset(dataset, destination_path, version='4.0.0'):
    metadata = datamart_profiler.process_dataset(dataset)

    if exists(destination_path):
        shutil.rmtree(destination_path)

    writer = D3mWriter(
        dataset_id='internal_dataset',
        destination=destination_path,
        metadata=metadata,
        format_options={'need_d3mindex': True, 'version': version},
    )
    with open(dataset, 'rb') as source:
        with writer.open_file('wb') as dest:
            shutil.copyfileobj(source, dest)
    writer.finish()


def create_d3m_problem(dataset, destination_path, problem_config):
    if isinstance(dataset, str):
        target_index = pd.read_csv(dataset).columns.get_loc(problem_config['target_name'])
    else:
        target_index = dataset.columns.get_loc(problem_config['target_name'])
    problem_config['target_index'] = target_index

    if exists(destination_path):
        shutil.rmtree(destination_path)
    os.makedirs(destination_path)

    problem_json = {
          "about": {
            "problemID": "",
            "problemName": "",
            "problemDescription": "",
            "problemVersion": "4.0.0",
            "problemSchemaVersion": "4.0.0",
            "taskKeywords": problem_config.get('task_keywords', ["classification", "multiClass"])
          },
          "inputs": {
            "data": [
              {
                "datasetID": "internal_dataset",
                "targets": [
                  {
                    "targetIndex": 0,
                    "resID": "learningData",
                    "colIndex": problem_config.get('target_index'),
                    "colName": problem_config.get('target_name')
                  }
                ]
              }
            ],
            "performanceMetrics": [
              {
                "metric": problem_config.get('metric', "accuracy")
              }
            ]
          },
          "expectedOutputs": {
            "predictionsFile": "predictions.csv"
          }
        }

    with open(join(destination_path, 'problemDoc.json'), 'w') as fout:
        json.dump(problem_json, fout, indent=4)
