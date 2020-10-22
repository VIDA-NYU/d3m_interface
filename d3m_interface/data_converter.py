import os
import json
import shutil
import logging
import importlib
import pickle
from os.path import join, exists
from d3m.container import Dataset
from d3m.utils import fix_uri
from d3m.container.utils import save_container
from d3m.metadata.problem import PerformanceMetricBase, TaskKeywordBase
from .database import PipelineParameter, PipelineConnection, PipelineModule

logger = logging.getLogger(__name__)
DATASET_ID = 'internal_dataset'


def is_d3m_format(dataset, suffix):
    if isinstance(dataset, str) and exists(join(dataset, 'dataset_%s/datasetDoc.json' % suffix)):
        return True

    return False


def convert_d3m_format(dataset_uri, output_folder, problem_config, suffix):
    logger.info('Reiceving a raw dataset, converting to D3M format')
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


def copy_folder(source_path, destination_path):
    if exists(destination_path):
        shutil.rmtree(destination_path)

    shutil.copytree(source_path, destination_path)


def make_pipeline_module(db, pipeline, name, package='d3m', version='2019.10.10'):
    pipeline_module = PipelineModule(pipeline=pipeline, package=package, version=version, name=name)
    db.add(pipeline_module)
    return pipeline_module


def make_data_module(db, pipeline):
    input_data = make_pipeline_module(db, pipeline, 'dataset', 'data', '0.0')
    return input_data


def connect(db, pipeline, from_module, to_module, from_output='produce', to_input='inputs'):
    db.add(PipelineConnection(pipeline=pipeline,
                              from_module=from_module,
                              to_module=to_module,
                              from_output_name=from_output,
                              to_input_name=to_input))


def set_hyperparams(db, pipeline, module, **hyperparams):
    db.add(PipelineParameter(
        pipeline=pipeline, module=module,
        name='hyperparams', value=pickle.dumps(hyperparams),
    ))


def export_pipeline_code(pipeline_template, ipython_cell=False):
    """Converts a Pipeline Description to an executable python script.
    """
    code = f"""
from d3m_interface.data_converter import connect, make_data_module, make_pipeline_module, set_hyperparams  
from d3m_interface.database import Pipeline, get_session
db = get_session()
pipeline = Pipeline(origin='export', dataset='dataset')
input_data = make_data_module(db, pipeline)
"""
    prev_step = None
    prev_steps = {}
    count_template_steps = 0
    for pipeline_step in pipeline_template['steps']:
        if pipeline_step['type'] == 'PRIMITIVE':
            code += f"""
step_{count_template_steps} = make_pipeline_module(db, pipeline,'{pipeline_step['primitive']['python_path']}')
"""
            if 'outputs' in pipeline_step:
                for output in pipeline_step['outputs']:
                    prev_steps['steps.%d.%s' % (count_template_steps, output['id'])] = "step_%d" % (
                        count_template_steps)

            if 'hyperparams' in pipeline_step:
                code += f"""
hyperparams = {"{}"}
"""
                for hyper, desc in pipeline_step['hyperparams'].items():
                    code += f"""
hyperparams['{hyper}'] = {"{"}'type':'{desc['type']}' ,'data':{desc['data']}{"}"}
"""
                code += f"""
set_hyperparams(db, pipeline,step_{count_template_steps}, **hyperparams)
"""

        else:
            # TODO In the future we should be able to handle subpipelines
            break
        if prev_step:
            if 'arguments' in pipeline_step:
                for argument, desc in pipeline_step['arguments'].items():
                    code += f"""
connect(db, pipeline,{prev_steps[desc['data']]}, step_{count_template_steps}, from_output='{desc['data'].split('.')[-1]}', to_input='{argument}')
"""
                code += f"""
connect(db, pipeline,{prev_step}, step_{count_template_steps}, from_output='index', to_input='index')
"""
        else:
            code += f"""
connect(db, pipeline,input_data, step_{count_template_steps}, from_output='dataset')
"""
        prev_step = "step_%d" % (count_template_steps)
        count_template_steps += 1
    code += f"""
db.add(pipeline)
db.commit()
"""
    if ipython_cell:
        from IPython.core.getipython import get_ipython
        shell = get_ipython()

        payload = dict(
            source='set_next_input',
            text=code,
            replace=False,
        )
        shell.payload_manager.write_payload(payload, single=False)
    return code

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

        # index is a special connection to keep the order of steps in fixed pipeline templates
        if 'index' in conn.to_input_name: continue

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

    klass = get_class(mod.name)
    primitive_desc = {
        key: value
        for key, value in klass.metadata.query().items()
        if key in {'id', 'version', 'python_path', 'name', 'digest'}
    }

    outputs = [{'id': k} for k, v in klass.metadata.query()['primitive_code']['instance_methods'].items()
               if v['kind'] == 'PRODUCE']
    if mod.name.endswith('.Fastlvm'):  # FIXME: Temporal solution, this module will be removed when DB is removed
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
    if mod.id in params and 'hyperparams' in params[mod.id]:
        hyperparams = pickle.loads(params[mod.id]['hyperparams'])
        # We check whether the hyperparameters have a value or the complete description
        hyperparams = {
            k: {'type': v['type'] if isinstance(v,dict) and 'type' in v else 'VALUE',
                'data': v['data'] if isinstance(v,dict) and 'data' in v else v}
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
    params = {}
    for param in pipeline.parameters:
        params.setdefault(param.module_id, {})[param.name] = param.value
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
