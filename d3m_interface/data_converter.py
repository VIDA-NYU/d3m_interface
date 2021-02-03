import os
import json
import shutil
import logging
import importlib
import pickle
import pandas as pd
from os.path import join, exists, split, dirname
from d3m.container import Dataset
from d3m.utils import fix_uri
from d3m.container.utils import save_container
from d3m.metadata.problem import PerformanceMetricBase, TaskKeywordBase
from d3m_interface.pipeline import PipelineParameter, PipelineConnection, PipelineModule

logger = logging.getLogger(__name__)
DATASET_ID = 'internal_dataset'


def is_d3m_format(dataset, suffix):
    if isinstance(dataset, str) and exists(join(dataset, 'dataset_%s/datasetDoc.json' % suffix)):
        return True

    return False


def dataset_to_d3m(dataset_uri, output_folder, problem_config, suffix):
    #logger.info('Reiceving a raw dataset, converting to D3M format')
    problem_config = check_problem_config(problem_config)
    dataset_folder = join(output_folder, 'temp', 'dataset_d3mformat', suffix, 'dataset_%s' % suffix)
    problem_folder = join(output_folder, 'temp', 'dataset_d3mformat', suffix, 'problem_%s' % suffix)
    dataset = create_d3m_dataset(dataset_uri, dataset_folder)
    create_d3m_problem(dataset['learningData'], problem_folder, problem_config)

    return join(output_folder, 'temp', 'dataset_d3mformat', suffix)


def check_problem_config(problem_config):
    if problem_config['target_column'] is None:
        raise ValueError('Parameter "target" not provided, but it is mandatory')

    valid_task_keywords = {keyword for keyword in TaskKeywordBase.get_map().keys() if keyword is not None}
    if problem_config['task_keywords'] is None:
        problem_config['task_keywords'] = ['classification', 'multiClass']
        logger.warning('Task keywords not defined, using: [%s]' % ', '.join(problem_config['task_keywords']))

    for task_keyword in problem_config['task_keywords']:
        if task_keyword not in valid_task_keywords:
            raise ValueError('Unknown "%s" task keyword, you should choose among [%s]' %
                             (task_keyword, ', '.join(valid_task_keywords)))

    valid_metrics = {metric for metric in PerformanceMetricBase.get_map()}
    if problem_config['metric'] is None:
        problem_config['metric'] = 'accuracy'
        if 'regression' in problem_config['task_keywords']:
            problem_config['metric'] = 'rootMeanSquaredError'
        logger.warning('Metric not defined, using: %s' % problem_config['metric'])
    elif problem_config['metric'] not in valid_metrics:
        raise ValueError('Unknown "%s" metric, you should choose among [%s]' %
                         (problem_config['metric'], ', '.join(valid_metrics)))

    #  Check special cases
    if problem_config['metric'] == 'f1' and 'binary' in problem_config['task_keywords'] and \
            'pos_label' not in problem_config['optional']:
        raise ValueError('pos_label parameter is mandatory for f1 and binary problems')

    return problem_config


def create_d3m_dataset(dataset_uri, destination_path):
    if callable(dataset_uri):
        dataset_uri = 'sklearn://' + dataset_uri.__name__.replace('load_', '')
    if exists(destination_path):
        shutil.rmtree(destination_path)

    dataset = Dataset.load(fix_uri(dataset_uri), dataset_id=DATASET_ID)
    save_container(dataset, destination_path)

    return dataset


def create_d3m_problem(dataset, destination_path, problem_config):
    target_index = dataset.columns.get_loc(problem_config['target_column'])
    problem_config['target_index'] = target_index

    if exists(destination_path):
        shutil.rmtree(destination_path)
    os.makedirs(destination_path)

    metric = {"metric": problem_config['metric']}
    if 'pos_label' in problem_config['optional']:
        metric['posLabel'] = str(problem_config['optional']['pos_label'])

    problem_json = {
        "about": {
            "problemID": "",
            "problemName": "",
            "problemDescription": "",
            "problemVersion": "4.0.0",
            "problemSchemaVersion": "4.0.0",
            "taskKeywords": problem_config['task_keywords']
        },
        "inputs": {
            "data": [
                {
                    "datasetID": DATASET_ID,
                    "targets": [
                        {
                            "targetIndex": 0,
                            "resID": "learningData",
                            "colIndex": problem_config['target_index'],
                            "colName": problem_config['target_column']
                        }
                    ]
                }
            ],
            "performanceMetrics": [metric]
        },
        "expectedOutputs": {
            "predictionsFile": "predictions.csv"
        }
    }

    with open(join(destination_path, 'problemDoc.json'), 'w') as fout:
        json.dump(problem_json, fout, indent=4)


def d3mtext_to_dataframe(folder_path, text_column):
    suffix = split(folder_path)[1]
    dataframe = pd.read_csv(join(folder_path, 'dataset_%s/tables/learningData.csv' % suffix))
    folder_files = join(folder_path, 'dataset_%s/media/' % suffix)

    def read_text(file_name):
        file_path = join(folder_files, file_name)
        with open(file_path, 'r') as fin:
            text = fin.read().replace('\n', ' ')
            return text

    dataframe[text_column] = dataframe[text_column].apply(read_text)

    return dataframe


def copy_folder(source_path, destination_path):
    if exists(destination_path):
        shutil.rmtree(destination_path)

    shutil.copytree(source_path, destination_path)


def make_pipeline_module(pipeline, name, package='d3m', version='2019.10.10'):
    pipeline_module = PipelineModule(package=package, version=version, name=name)
    pipeline.add_module(pipeline_module)
    return pipeline_module


def make_data_module(pipeline):
    input_data = make_pipeline_module(pipeline, 'dataset', 'data', '0.0')
    return input_data


def connect(pipeline, from_module, to_module, from_output='produce', to_input='inputs'):
    connection = PipelineConnection(from_module_id=from_module.id,
                                    from_output_name=from_output,
                                    to_module_id=to_module.id,
                                    to_input_name=to_input)
    pipeline.add_connection(connection)
    to_module.add_connection_to(connection)


def set_hyperparams(pipeline, module, **hyperparams):
    parameters = PipelineParameter(module_id=module.id,
                                   name='hyperparams',
                                   value=pickle.dumps(hyperparams))
    pipeline.add_parameters(parameters)


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _add_step(steps, modules, params, module_to_step, mod):
    if mod.id in module_to_step:
        return module_to_step[mod.id]

    # Special case: the "dataset" module
    if mod.package == 'data' and mod.name == 'dataset':
        module_to_step[mod.id] = 'inputs.0'
        return 'inputs.0'
    elif mod.package != 'd3m':
        raise ValueError("Got unknown module '%s:%s'", mod.package, mod.name)

    # Recursively walk upstream modules (to get `steps` in topological order)
    # Add inputs to a dictionary, in deterministic order
    inputs = {}

    for conn in sorted(mod.connections_to, key=lambda c: c.to_input_name):
        step = _add_step(steps, modules, params, module_to_step, modules[conn.from_module_id])

        if step.startswith('inputs.'):
            inputs[conn.to_input_name] = step
        else:
            if conn.to_input_name in inputs:
                previous_value = inputs[conn.to_input_name]
                if isinstance(previous_value, str):
                    inputs[conn.to_input_name] = [previous_value] + ['%s.%s' % (step, conn.from_output_name)]
                else:
                    inputs[conn.to_input_name].append('%s.%s' % (step, conn.from_output_name))
            else:
                inputs[conn.to_input_name] = '%s.%s' % (step, conn.from_output_name)
    #TODO load metadata from primitives
    primitive_desc = {
        'id': mod.id,
        'version': mod.version,
        'python_path': mod.name,
        'name': mod.name
    }
    with open(join(dirname(__file__), 'resource/primitives_metadata.json')) as f:
        primitives_metadata = json.load(f)
        for primitve in primitives_metadata:
            if primitive_desc['python_path'] == primitve['python_path']:
                primitive_desc['id'] = primitve['id']
                primitive_desc['version'] = primitve['version']
                primitive_desc['name'] = primitve['name']
                primitive_desc['digest'] = primitve['digest']
                break

    outputs = [{'id': 'produce'}]

    # Create step description
    if len(inputs) > 0:
        step = {
            'type': 'PRIMITIVE',
            'primitive': primitive_desc,
            'arguments': {
                name: {
                    'type': 'CONTAINER',
                    'data': data,
                }
                for name, data in inputs.items()
            },
            'outputs': outputs
        }
    else:
        step = {
            'type': 'PRIMITIVE',
            'primitive': primitive_desc,
        }

    # If hyperparameters are set, export them
    if mod.id in params:
        hyperparams = params[mod.id]
        # We check whether the hyperparameters have a value or the complete description
        hyperparams = {
            k: {'type': v['type'] if isinstance(v, dict) and 'type' in v else 'VALUE',
                'data': v['data'] if isinstance(v, dict) and 'data' in v else v}
            for k, v in hyperparams.items()
        }
        step['hyperparams'] = hyperparams

    step_nb = 'steps.%d' % len(steps)
    steps.append(step)
    module_to_step[mod.id] = step_nb

    return step_nb


def to_d3m_json(pipeline):
    """Converts a Pipeline to the JSON schema from metalearning working group.
    """
    steps = []
    modules = {mod.id: mod for mod in pipeline.modules}
    params = pipeline.parameters
    module_to_step = {}
    for _, mod in sorted(modules.items(), key=lambda x: x[0]):
        _add_step(steps, modules, params, module_to_step, mod)

    return {
        'id': str(pipeline.id),
        'name': str(pipeline.id),
        'description': pipeline.origin or '',
        'schema': 'https://metadata.datadrivendiscovery.org/schemas/'
                  'v0/pipeline.json',
        'created': pipeline.created_date.isoformat() + 'Z',
        'context': 'TESTING',
        'inputs': [
            {'name': "input dataset"},
        ],
        'outputs': [
            {
                'data': 'steps.%d.produce' % (len(steps) - 1),
                'name': "predictions",
            }
        ],
        'steps': steps,
    }
