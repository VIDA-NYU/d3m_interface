import sys
import time
import logging
import subprocess
import pandas as pd
import datetime
from os.path import join, split
from d3m_interface.basic_ta3 import BasicTA3
from d3m_interface.data_converter import is_d3m_format, convert_d3m_format


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
pd.set_option('display.max_colwidth', None)

TA2_DOCKER_IMAGES = {'NYU': 'registry.gitlab.com/vida-nyu/d3m/ta2:latest', 'TAMU': 'dmartinez05/tamuta2:latest',
                     'CMU': 'registry.datadrivendiscovery.org/sheath/cmu-ta2:latest'}

IGNORE_SUMMARY_PRIMITIVES = {'d3m.primitives.data_transformation.construct_predictions.Common',
                             'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                             'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                             'd3m.primitives.data_transformation.denormalize.Common',
                             'd3m.primitives.data_transformation.column_parser.Common'}


class Automl:

    def __init__(self, output_folder, ta2_id='NYU'):
        if ta2_id not in TA2_DOCKER_IMAGES:
            logger.error('TA2 "%s" does not exist' % ta2_id)
            return

        self.output_folder = output_folder
        self.ta2_id = ta2_id
        self.pipelines = {}
        self.ta2 = None
        self.ta3 = None
        self.dataset = None
        self.leaderboard = None
        self.problem_config = None

    def search_pipelines(self, dataset, time_bound, target=None, metric='f1Macro', task_keywords=['classification']):
        suffix = 'TRAIN'
        if not is_d3m_format(dataset, suffix):
            self.problem_config = {'target_name': target, 'metric': metric, 'task_keywords': task_keywords}
            dataset = convert_d3m_format(dataset, self.output_folder, self.problem_config, suffix)

        self.dataset = split(dataset)[0]
        self.start_ta2()

        dataset_in_container = '/input/dataset/TRAIN/dataset_TRAIN/datasetDoc.json'
        problem_path = join(dataset, 'problem_TRAIN/problemDoc.json')
        start_time = datetime.datetime.utcnow()
        pipelines = self.ta3.do_search(dataset_in_container, problem_path, time_bound)

        for pipeline in pipelines:
            end_time = datetime.datetime.utcnow()
            pipeline_json = self.ta3.do_describe(pipeline['id'])
            summary_pipeline = self.get_summary_pipeline(pipeline_json)
            pipeline['json_representation'] = pipeline_json
            pipeline['summary'] = summary_pipeline
            pipeline['found_time'] = end_time.isoformat() + 'Z'
            duration = str(end_time - start_time)
            logger.info('Found pipeline, id=%s, %s=%s, time=%s' %
                        (pipeline['id'], pipeline['metric'], pipeline['score'], duration))
            self.pipelines[pipeline['id']] = pipeline

        if len(self.pipelines) > 0:
            leaderboard = []
            sorted_pipelines = sorted(self.pipelines.values(), key=lambda x: x['normalized_score'], reverse=True)
            metric = sorted_pipelines[0]['metric']
            for position, pipeline_data in enumerate(sorted_pipelines, 1):
                leaderboard.append([position, pipeline_data['id'], pipeline_data['summary'],  pipeline_data['score']])

            self.leaderboard = pd.DataFrame(leaderboard, columns=['ranking', 'id', 'summary', metric])

        return self.pipelines.values()

    def train(self, solution_id):
        dataset_in_container = '/input/dataset/TRAIN/dataset_TRAIN/datasetDoc.json'

        if solution_id not in self.pipelines:
            logger.error('Pipeline id=%s does not exist' % solution_id)
            return

        logger.info('Training model...')
        fitted_solution_id = self.ta3.do_train(solution_id, dataset_in_container)
        fitted_solution = None  # TODO: Call to LoadFittedSolution, but TA2 could not have implemented it yet
        model = {fitted_solution_id: fitted_solution}
        logger.info('Training finished!')

        return model

    def test(self, model, test_dataset):
        suffix = 'TEST'
        if not is_d3m_format(test_dataset, suffix):
            convert_d3m_format(test_dataset, self.output_folder, self.problem_config, suffix)

        dataset_in_container = '/input/dataset/TEST/dataset_TEST/datasetDoc.json'
        fitted_solution_id = list(model.keys())[0]
        logger.info('Testing model...')
        predictions_path_in_container = self.ta3.do_test(fitted_solution_id, dataset_in_container)

        if not predictions_path_in_container.startswith('file://'):
            logger.error('Exposed output "%s" from TA2 cannot be read', predictions_path_in_container)
            return
        logger.info('Testing finished!')
        predictions_path_in_container = predictions_path_in_container.replace('file:///output/', '')
        predictions = pd.read_csv(join(self.output_folder, predictions_path_in_container))

        return predictions

    def test_score(self, solution_id, score_dataset):
        #  TODO: Use TA2TA3 API to score

        if solution_id not in self.pipelines:
            logger.error('Pipeline id=%s does not exist' % solution_id)
            return

        pipeline_id = self.pipelines[solution_id]['json_representation']['id']
        search_id = self.pipelines[solution_id]['search_id']

        dataset_in_container = '/input/dataset/'
        dataset_train_path = join(dataset_in_container, 'TRAIN/dataset_TRAIN/datasetDoc.json')
        dataset_test_path = join(dataset_in_container, 'TEST/dataset_TEST/datasetDoc.json')
        dataset_score_path = join(dataset_in_container, 'SCORE/dataset_SCORE/datasetDoc.json')
        problem_path = join(dataset_in_container, 'TRAIN/problem_TRAIN/problemDoc.json')
        pipeline_path = join('/output/', search_id, 'pipelines_searched', '%s.json' % pipeline_id)
        score_pipeline_path = join('/output/', 'fit_score_%s.csv' % pipeline_id)
        metric = None
        score = None

        try:
            process = subprocess.Popen(
                [
                    'docker', 'exec', 'ta2_container',
                    'python3', '-m', 'd3m',
                    'runtime',
                    '--context', 'TESTING',
                    '--random-seed', '0',
                    'fit-score',
                    '--pipeline', pipeline_path,
                    '--problem', problem_path,
                    '--input', dataset_train_path,
                    '--test-input', dataset_test_path,
                    '--score-input', dataset_score_path,
                    '--scores', score_pipeline_path
                ]
            )
            process.wait()
            df = pd.read_csv(join(self.output_folder, 'fit_score_%s.csv' % pipeline_id))
            score = round(df['value'][0], 5)
            metric = df['metric'][0].lower()
        except:
            logger.exception('Error calculating test score')

        return metric, score

    def create_profiler_inputs(self):
        profiler_inputs = []
        pipeline_ids = set()

        for pipeline in self.pipelines.values():
            if pipeline['id'] not in pipeline_ids:
                pipeline_ids.add(pipeline['id'])
                if 'digest' not in pipeline['json_representation']:
                    pipeline['json_representation']['digest'] = pipeline['id']  # TODO: Compute digest

                pipeline['score'] = [{'metric': {'metric': pipeline['metric']}, 'value': pipeline['score'],
                                      'normalized': pipeline['normalized_score']}]

                profiler_data = {
                    'pipeline_id': pipeline['json_representation']['id'],
                    'inputs': pipeline['json_representation']['inputs'],
                    'steps': pipeline['json_representation']['steps'],
                    'outputs': pipeline['json_representation']['outputs'],
                    'pipeline_digest': pipeline['json_representation']['digest'],
                    'problem': self.dataset,
                    'start': pipeline['json_representation']['created'],
                    'end': pipeline['found_time'],
                    'scores': pipeline['score'],
                    'pipeline_source': {'name': self.ta2_id},
                }
                profiler_inputs.append(profiler_data)

            else:
                logger.warning('Ignoring repeated pipeline id=%s' % pipeline['id'])

        return profiler_inputs

    def start_ta2(self):
        logger.info('Initializing TA2...')

        process = subprocess.Popen(['docker', 'stop', 'ta2_container'])
        process.wait()

        self.ta2 = subprocess.Popen(
            [
                'docker', 'run', '--rm',
                '--name', 'ta2_container',
                '-p', '45042:45042',
                '-e', 'D3MRUN=ta2ta3',
                '-e', 'D3MINPUTDIR=/input',
                '-e', 'D3MOUTPUTDIR=/output',
                '-e', 'D3MSTATICDIR=/output',  # TODO: Temporal assignment for D3MSTATICDIR env variable
                '-v', '%s:/input/dataset/' % self.dataset,
                '-v', '%s:/output' % self.output_folder,
                TA2_DOCKER_IMAGES[self.ta2_id]
            ]
        )
        time.sleep(4)  # Wait for TA2
        while True:
            try:
                self.ta3 = BasicTA3()
                self.ta3.do_hello()
                logger.info('TA2 initialized!')
                break
            except:
                if self.ta3.channel is not None:
                    self.ta3.channel.close()
                    self.ta3 = None

                time.sleep(4)

    def end_session(self):
        logger.info('Ending session...')
        if self.ta2 is not None:
            process = subprocess.Popen(['docker', 'stop', 'ta2_container'])
            process.wait()

        logger.info('Session ended!')

    def get_summary_pipeline(self, pipeline_json):
        primitives_summary = []
        for primitive in pipeline_json['steps']:
            primitive_name = primitive['primitive']['python_path']
            if primitive_name not in IGNORE_SUMMARY_PRIMITIVES:
                primitive_name_short = '.'.join(primitive_name.split('.')[-2:]).lower()
                if primitive_name_short not in primitives_summary:
                    primitives_summary.append(primitive_name_short)

        return ', '.join(primitives_summary)

    @staticmethod
    def add_new_ta2(name, docker_image):
        TA2_DOCKER_IMAGES[name] = docker_image
        logger.info('TA2 "%s" added!', name)



