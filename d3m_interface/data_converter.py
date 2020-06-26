import os
import json
import shutil
import logging
from os.path import join, exists
from d3m.container import Dataset
from d3m.utils import fix_uri
from d3m.container.utils import save_container

logger = logging.getLogger(__name__)
DATASET_ID = 'internal_dataset'


def is_d3m_format(dataset, suffix):
    if isinstance(dataset, str) and exists(join(dataset, 'dataset_%s/datasetDoc.json' % suffix)):
        return True

    return False


def convert_d3m_format(dataset_uri, output_folder, problem_config, suffix):
    logger.info('Reiceving a raw dataset, converting to D3M format')
    dataset_folder = join(output_folder, 'temp', 'dataset_d3mformat', suffix, 'dataset_%s' % suffix)
    problem_folder = join(output_folder, 'temp', 'dataset_d3mformat', suffix, 'problem_%s' % suffix)
    dataset = create_d3m_dataset(dataset_uri, dataset_folder)
    create_d3m_problem(dataset['learningData'], problem_folder, problem_config)

    return join(output_folder, 'temp', 'dataset_d3mformat', suffix)


def create_d3m_dataset(dataset_uri, destination_path):
    if exists(destination_path):
        shutil.rmtree(destination_path)

    dataset = Dataset.load(fix_uri(dataset_uri), dataset_id=DATASET_ID)
    save_container(dataset, destination_path)

    return dataset


def create_d3m_problem(dataset, destination_path, problem_config):
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
            "taskKeywords": problem_config.get('task_keywords')
          },
          "inputs": {
            "data": [
              {
                "datasetID": DATASET_ID,
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
