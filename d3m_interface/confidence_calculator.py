import re
import copy


def is_position_computeunique(step):
    if step['primitive']['python_path'] == 'd3m.primitives.data_transformation.dataset_to_dataframe.Common':
        return True
    return False


def is_position_constructconfidence(step):
    if step['primitive']['python_path'] == 'd3m.primitives.data_transformation.construct_predictions.Common':
        return True
    return False


def is_estimator(step):
    if step['primitive']['python_path'].startswith('d3m.primitives.classification.'):
        return True
    return False


def update_arguments(step, argument, id_last_input):
    if argument in step['arguments']:
        id_current_input = int(re.match('steps.(\d+).produce', step['arguments'][argument]['data']).groups()[0])
        if id_current_input >= id_last_input:
            step['arguments'][argument]['data'] = 'steps.%d.produce' % (id_current_input + 1)

    return step


def create_confidence_pipeline(pipeline):
    pipeline = copy.deepcopy(pipeline)
    steps = pipeline['steps']
    new_steps = []
    step_index = 0
    id_last_input = None
    estimator_index = None
    
    for step in steps:
        if is_position_computeunique(step):
            # Add compute_unique_values primitive after dataset_to_dataframe primitive
            new_steps.append(step)
            id_last_input = int(re.match('steps.(\d+).produce', step['arguments']['inputs']['data']).groups()[0])
            step_index += 1 

            step = {
                "type": "PRIMITIVE",
                "primitive": {
                    "id": "dd580c45-9fbe-493d-ac79-6e9f706a3619",
                    "version": "0.1.0",
                    "name": "Add all_distinct_values to the metadata of the input Dataframe",
                    "python_path": "d3m.primitives.operator.compute_unique_values.Common",
                    "digest": "04178b5c55f24e4f0a2acd5111affe8fc6fe752ab54a380028d439787ec170ef"
                },
                "arguments": {
                    "inputs": {
                        "type": "CONTAINER",
                        "data": step['arguments']['inputs']['data']
                    }
                },
                "outputs": [
                    {
                        "id": "produce"
                    }
                ]
            }
            
        elif is_position_constructconfidence(step):
            # Replace primitive construct_confidence instead of construct_predictions
            step = {
                "type": "PRIMITIVE",
                "primitive": {
                    "id": "500c4f0c-a040-48a5-aa76-d6463ea7ea37",
                    "version": "0.1.0",
                    "name": "Construct confidence",
                    "python_path": "d3m.primitives.data_transformation.construct_confidence.Common",
                    "digest": "8685b43969b06ee2418e2852d8019b4c638813cca5cb2d94962ce782e0ae4651"
                },
                "arguments": {
                    "inputs": {
                        "type": "CONTAINER",
                        "data": 'steps.%d.produce' % (id_last_input + 1)
                    },
                    "reference": {
                        "type": "CONTAINER",
                        "data": step['arguments']['reference']['data']
                    }
                },
                "outputs": [
                    {
                        "id": "produce"
                    }
                ],
                "hyperparams": {
                    "primitive_learner": {
                        "type": "PRIMITIVE",
                        "data": estimator_index
                    }
                }
            }

        elif is_estimator(step):
            estimator_index = step_index

        if id_last_input is not None:
            step = update_arguments(step, 'inputs', id_last_input)
            step = update_arguments(step, 'outputs', id_last_input)
            step = update_arguments(step, 'reference', id_last_input)

        new_steps.append(step)
        step_index += 1

    pipeline['steps'] = new_steps
    id_output = int(re.match('steps.(\d+).produce', pipeline['outputs'][0]['data']).groups()[0])
    pipeline['outputs'][0]['data'] = 'steps.%d.produce' % (id_output + 1)

    return pipeline
