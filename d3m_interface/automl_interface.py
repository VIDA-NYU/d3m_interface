import os
import sys
import time
import json
import logging
import signal
import subprocess
import pandas as pd
import datetime
from os.path import join, split, exists
from d3m_interface.basic_ta3 import BasicTA3
from d3m_interface.visualization import plot_metadata, plot_comparison_pipelines
from d3m_interface.data_converter import is_d3m_format, convert_d3m_format, copy_folder
from d3m.metadata.problem import PerformanceMetric
from threading import Thread


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
pd.set_option('display.max_colwidth', None)

TA2_DOCKER_IMAGES = {'NYU': 'registry.gitlab.com/vida-nyu/d3m/ta2:latest',
                     'CMU': 'registry.datadrivendiscovery.org/sheath/cmu-ta2:latest',
                     'SRI': 'registry.gitlab.com/daraghhartnett/autoflow:latest',
                     'TAMU': 'dmartinez05/tamuta2:latest'}

IGNORE_SUMMARY_PRIMITIVES = {'d3m.primitives.data_transformation.construct_predictions.Common',
                             'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                             'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                             'd3m.primitives.data_transformation.denormalize.Common',
                             'd3m.primitives.data_transformation.column_parser.Common'}


def kernel_interrupt_handler(signal, frame):
    logger.info('KeyboardInterrupt signal received, ending session...')
    subprocess.call(['docker', 'stop', 'ta2_container'])
    logger.info('Session ended!')

    raise KeyboardInterrupt


class Automl:

    def __init__(self, output_folder, ta2_id='NYU'):
        """Create/instantiate an Automl object

        :param output_folder: Path to the output directory
        :param ta2_id: TA2 system name. It makes reference to the TA2 docker image. The provided TA2 systems are the
            following: `NYU, CMU, SRI, TAMU`
        """
        if ta2_id not in TA2_DOCKER_IMAGES:
            raise ValueError('Unknown "%s" TA2, you should choose among: [%s]' % (ta2_id, ', '.join(TA2_DOCKER_IMAGES)))

        self.output_folder = output_folder
        self.ta2_id = ta2_id
        self.pipelines = {}
        self.ta2 = None
        self.ta3 = None
        self.dataset = None
        self.leaderboard = None
        self.problem_config = None

    def search_pipelines(self, dataset, time_bound, time_bound_run=5, target=None, metric=None, task_keywords=None,
                         method='holdout', stratified=True, shuffle=True, folds=10, train_ratio=0.70, random_seed=0,
                         **kwargs):
        """Perform the search of pipelines

        :param dataset: Path to dataset. It supports D3M dataset, CSV file, OpenML, and Sklearn datasets
        :param time_bound: Limit time in minutes to perform the search
        :param time_bound_run: Limit time in minutes to score a pipeline
        :param target: Column name of the potential target variable for a problem.
        :param metric: The provided metrics are the following: `hammingLoss, accuracy, objectDetectionAP,
            rocAucMicro, f1Macro, meanSquaredError, f1, jaccardSimilarityScore, normalizedMutualInformation, rocAuc,
            f1Micro, hitsAtK, meanAbsoluteError, rocAucMacro, rSquared, recall, meanReciprocalRank, precision,
            precisionAtTopK, rootMeanSquaredError`
        :param task_keywords: A list of keywords that capture the nature of the machine learning task. The keywords
            that can be combined to describe the task are the following: `tabular, nested, multiLabel, video,
            linkPrediction, multivariate, graphMatching, forecasting, classification, graph, semiSupervised, text,
            timeSeries, clustering, collaborativeFiltering, univariate, missingMetadata, remoteSensing, multiClass,
            regression, multiGraph, lupi, relational, audio, grouped, objectDetection, vertexNomination,
            communityDetection, geospatial, image, overlapping, nonOverlapping, speech, vertexClassification, binary`
        :param method: Method to score the pipeline: `holdout, cross_validation`
        :param stratified: Whether or not to split the data using a stratified strategy
        :param shuffle: Whether or not to shuffle the data before splitting
        :param folds: the seed used by the random number generator
        :param train_ratio: Represent the proportion of the dataset to include in the train split
        :param random_seed: The number seed used by the random generator
        :param kwargs: Different arguments for problem's settings (e.g. pos_label for binary problems using F1)
        :returns: List of pipelines
        """
        suffix = 'TRAIN'
        signal.signal(signal.SIGINT, kernel_interrupt_handler)
        if not is_d3m_format(dataset, suffix):
            self.problem_config = {'target_column': target, 'metric': metric, 'task_keywords': task_keywords,
                                   'optional': kwargs}
            dataset = convert_d3m_format(dataset, self.output_folder, self.problem_config, suffix)

        self.dataset = split(dataset)[0]
        self.start_ta2()
        search_id = None
        signal.signal(signal.SIGALRM, lambda signum, frame: self.ta3.do_stop_search(search_id))
        signal.alarm(time_bound * 60)

        dataset_in_container = '/input/dataset/TRAIN/dataset_TRAIN/datasetDoc.json'
        problem_path = join(dataset, 'problem_TRAIN/problemDoc.json')
        start_time = datetime.datetime.utcnow()
        pipelines = self.ta3.do_search(dataset_in_container, problem_path, time_bound, time_bound_run)

        jobs = []

        for pipeline in pipelines:
            end_time = datetime.datetime.utcnow()
            try:
                pipeline_json = self.ta3.do_describe(pipeline['id'])
            except:
                logger.warning('Pipeline id=%s could not be decoded' % pipeline['id'])
                continue
            summary_pipeline = self.get_summary_pipeline(pipeline_json)
            search_id = pipeline['search_id']
            pipeline['json_representation'] = pipeline_json
            pipeline['summary'] = summary_pipeline
            pipeline['found_time'] = end_time.isoformat() + 'Z'
            duration = str(end_time - start_time)
            logger.info('Found pipeline id=%s, time=%s, scoring...' % (pipeline['id'], duration))

            job = Thread(target=self.score_in_search, args=(pipeline, dataset_in_container, problem_path, self.pipelines,
                                                            method, stratified, shuffle, folds, train_ratio, random_seed))
            jobs.append(job)
            job.start()

        if len(jobs) > 0:
            logger.info('Search completed, still scoring some pending pipelines...')

            for job in jobs:
                job.join()

            logger.info('Scoring completed for all pipelines!')
        else:
            logger.info('Search completed, no pipelines found!')

        if len(self.pipelines) > 0:
            leaderboard = []
            sorted_pipelines = sorted(self.pipelines.values(), key=lambda x: x['normalized_score'], reverse=True)
            metric = sorted_pipelines[0]['metric']
            for position, pipeline_data in enumerate(sorted_pipelines, 1):
                leaderboard.append([position, pipeline_data['id'], pipeline_data['summary'],  pipeline_data['score']])

            self.leaderboard = pd.DataFrame(leaderboard, columns=['ranking', 'id', 'summary', metric])

        signal.alarm(0)
        return self.pipelines.values()

    def train(self, pipeline_id, expose_outputs=None):
        """Train a model using an specific ML pipeline

        :param pipeline_id: Pipeline id
        :param expose_outputs: The output of the pipeline steps. If None, it doesn't expouse any output of the steps.
            If str, should be 'all' to shows the output of each step in the pipeline, If list, it should contain the
            ids of the steps, e.g. 'steps.2.produce'
        :returns: A dictionary that contains the id and fitted model with/without the pipeline step outputs
        """

        dataset_in_container = '/input/dataset/TRAIN/dataset_TRAIN/datasetDoc.json'

        if pipeline_id not in self.pipelines:
            raise ValueError('Pipeline id=%s does not exist' % pipeline_id)

        logger.info('Training model...')
        if expose_outputs is None:
            expose_outputs = []
        elif isinstance(expose_outputs, str) and expose_outputs == 'all':
            expose_outputs = ['outputs.0']
            for index, step in enumerate(self.pipelines[pipeline_id]['json_representation']['steps']):
                for id_output in step['outputs']:
                    expose_outputs.append('steps.%d.%s' % (index, id_output['id']))

        fitted_pipeline_id, pipeline_step_outputs = self.ta3.do_train(pipeline_id, dataset_in_container, expose_outputs)
        fitted_pipeline = None  # TODO: Call to LoadFittedSolution, but TA2 could not have implemented it yet

        for step_id, step_csv_uri in pipeline_step_outputs.items():
            if not step_csv_uri.startswith('file://'):
                logger.warning('Exposed step output "%s" cannot be read' % step_id)
                continue
            step_csv_uri = step_csv_uri.replace('file:///output/', '')
            step_dataframe = pd.read_csv(join(self.output_folder, step_csv_uri))
            pipeline_step_outputs[step_id] = step_dataframe

        self.pipelines[pipeline_id]['fitted_id'] = fitted_pipeline_id
        model = {'id': pipeline_id, 'fitted': fitted_pipeline}
        logger.info('Training finished!')

        if len(expose_outputs) == 0:
            return model

        return model, pipeline_step_outputs

    def test(self, model, test_dataset, expose_outputs=None):
        """Test a model

        :param model: Dict that contains the id and fitted model
        :param test_dataset: Path to dataset. It supports D3M dataset, and CSV file
        :param expose_outputs: The output of the pipeline steps. If None, it doesn't expouse any output of the steps.
            If str, should be 'all' to shows the output of each step in the pipeline, If list, it should contain the
            ids of the steps, e.g. 'steps.2.produce'
        :returns: A dataframe that contains the predictions with/without the pipeline step outputs
        """
        suffix = 'TEST'
        dataset_in_container = '/input/dataset/TEST/dataset_TEST/datasetDoc.json'

        if not is_d3m_format(test_dataset, suffix):
            convert_d3m_format(test_dataset, self.output_folder, self.problem_config, suffix)
        elif test_dataset != join(self.dataset, 'TEST'):  # Special case for D3M test dataset with different path
            destination_path = join(self.output_folder, 'temp', 'dataset_d3mformat', 'TEST')
            copy_folder(test_dataset, destination_path)
            dataset_in_container = '/output/temp/dataset_d3mformat/TEST/dataset_TEST/datasetDoc.json'

        pipeline_id = model['id']
        fitted_pipeline_id = self.pipelines[pipeline_id]['fitted_id']
        logger.info('Testing model...')

        if expose_outputs is None:
            expose_outputs = []
        elif isinstance(expose_outputs, str) and expose_outputs == 'all':
            expose_outputs = ['outputs.0']
            for index, step in enumerate(self.pipelines[pipeline_id]['json_representation']['steps']):
                for id_output in step['outputs']:
                    expose_outputs.append('steps.%d.%s' % (index, id_output['id']))

        # Force to generate the predictions
        if 'outputs.0' not in expose_outputs:
            expose_outputs.append('outputs.0')

        pipeline_step_outputs = self.ta3.do_test(fitted_pipeline_id, dataset_in_container, expose_outputs)

        for step_id, step_csv_uri in pipeline_step_outputs.items():
            if not step_csv_uri.startswith('file://'):
                logger.warning('Exposed step output "%s" cannot be read' % step_id)
                continue
            step_csv_uri = step_csv_uri.replace('file:///output/', '')
            step_dataframe = pd.read_csv(join(self.output_folder, step_csv_uri))
            pipeline_step_outputs[step_id] = step_dataframe

        predictions = pipeline_step_outputs['outputs.0']
        logger.info('Testing finished!')

        if len(expose_outputs) == 1:
            return predictions

        return predictions, pipeline_step_outputs

    def score(self, pipeline_id, test_dataset):
        """Compute a proper score of the model

        :param pipeline_id: Pipeline id
        :param test_dataset: Path to dataset. It supports D3M dataset, and CSV file
        :returns: A tuple holding metric name and score value
        """
        suffix = 'SCORE'

        if pipeline_id not in self.pipelines:
            raise ValueError('Pipeline id=%s does not exist' % pipeline_id)

        if not is_d3m_format(test_dataset, suffix):
            convert_d3m_format(test_dataset, self.output_folder, self.problem_config, suffix)

        with open(join(self.output_folder, '%s.json' % pipeline_id), 'w') as fout:
            json.dump(self.pipelines[pipeline_id]['json_representation'], fout)  # Save temporally the json pipeline

        dataset_in_container = '/input/dataset/'
        dataset_train_path = join(dataset_in_container, 'TRAIN/dataset_TRAIN/datasetDoc.json')
        dataset_test_path = join(dataset_in_container, 'TEST/dataset_TEST/datasetDoc.json')
        dataset_score_path = join(dataset_in_container, 'SCORE/dataset_SCORE/datasetDoc.json')
        problem_path = join(dataset_in_container, 'TRAIN/problem_TRAIN/problemDoc.json')
        pipeline_path = join('/output/', '%s.json' % pipeline_id)
        score_pipeline_path = join('/output/', 'fit_score_%s.csv' % pipeline_id)

        #  TODO: Use TA2TA3 API to score
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
        result_path = join(self.output_folder, 'fit_score_%s.csv' % pipeline_id)
        if not exists(result_path):
            raise FileNotFoundError('Pipeline id=%s could not be scored' % pipeline_id)

        df = pd.read_csv(result_path)
        score = round(df['value'][0], 5)
        metric = df['metric'][0].lower()

        return metric, score

    def create_pipelineprofiler_inputs(self, test_dataset=None):
        """Create an proper input supported by PipelineProfiler based on the pipelines generated by a TA2 system

        :param test_dataset: Path to dataset. If None it will use the search scores, otherwise will score the
            pipelines over the passed dataset
        :returns: List of pipelines in the PipelineProfiler input format
        """
        profiler_inputs = []
        pipeline_ids = set()

        if test_dataset is not None:
            logger.info('Calculating scores in the test dataset...')

        for pipeline in self.pipelines.values():
            if pipeline['id'] not in pipeline_ids:
                pipeline_ids.add(pipeline['id'])
                if 'digest' not in pipeline['json_representation']:
                    pipeline['json_representation']['digest'] = pipeline['id']  # TODO: Compute digest

                pipeline_score = [{'metric': {'metric': pipeline['metric']}, 'value': pipeline['score'],
                                   'normalized': pipeline['normalized_score']}]
                problem = self.dataset
                start_time = pipeline['json_representation']['created']
                end_time = pipeline['found_time']

                if test_dataset is not None:
                    problem = test_dataset
                    start_time = datetime.datetime.utcnow().isoformat() + 'Z'
                    try:
                        metric, score,  = self.score(pipeline['id'], test_dataset)
                    except:
                        logger.warning('Pipeline id=%s could not be scored' % pipeline['id'])
                        continue
                    end_time = datetime.datetime.utcnow().isoformat() + 'Z'
                    normalized_score = PerformanceMetric[metric.upper()].normalize(score)
                    pipeline_score = [{'metric': {'metric': metric}, 'value': score,
                                       'normalized': normalized_score}]

                profiler_data = {
                    'pipeline_id': pipeline['json_representation']['id'],
                    'inputs': pipeline['json_representation']['inputs'],
                    'steps': pipeline['json_representation']['steps'],
                    'outputs': pipeline['json_representation']['outputs'],
                    'pipeline_digest': pipeline['json_representation']['digest'],
                    'problem': problem,
                    'start': start_time,
                    'end': end_time,
                    'scores': pipeline_score,
                    'pipeline_source': {'name': self.ta2_id},
                }
                profiler_inputs.append(profiler_data)

            else:
                logger.warning('Ignoring repeated pipeline id=%s' % pipeline['id'])
        logger.info('Inputs for PipelineProfiler created!')

        return profiler_inputs

    def end_session(self):
        """This safely ends session in D3M interface
        """
        logger.info('Ending session...')
        if self.ta2 is not None:
            subprocess.call(['docker', 'stop', 'ta2_container'])

        logger.info('Session ended!')

    def start_ta2(self):
        logger.info('Initializing %s TA2...', self.ta2_id)
        subprocess.call(['docker', 'stop', 'ta2_container'])

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
                '--user', '%s:%s' % (os.geteuid(), os.getgid()),
                TA2_DOCKER_IMAGES[self.ta2_id]
            ]
        )
        time.sleep(4)  # Wait for TA2
        while True:
            try:
                self.ta3 = BasicTA3()
                self.ta3.do_hello()
                logger.info('%s TA2 initialized!', self.ta2_id)
                break
            except:
                if self.ta3.channel is not None:
                    self.ta3.channel.close()
                    self.ta3 = None

                time.sleep(4)

    def score_in_search(self, pipeline, dataset_in_container, problem_path, pipelines, method, stratified, shuffle,
                        folds, train_ratio, random_seed):
        try:
            score_data = self.ta3.do_score(pipeline['id'], dataset_in_container, problem_path, method, stratified,
                                           shuffle, folds, train_ratio, random_seed)
            logger.info('Scored pipeline id=%s, %s=%s' % (pipeline['id'], score_data['metric'], score_data['score']))
            pipeline['score'] = score_data['score']
            pipeline['normalized_score'] = score_data['normalized_score']
            pipeline['metric'] = score_data['metric']
            pipelines[pipeline['id']] = pipeline
        except:
            logger.warning('Pipeline id=%s could not be scored', pipeline['id'])

    def get_summary_pipeline(self, pipeline_json):
        primitives_summary = []
        for primitive in pipeline_json['steps']:
            primitive_name = primitive['primitive']['python_path']
            if primitive_name not in IGNORE_SUMMARY_PRIMITIVES:
                primitive_name_short = '.'.join(primitive_name.split('.')[-2:]).lower()
                if primitive_name_short not in primitives_summary:
                    primitives_summary.append(primitive_name_short)

        return ', '.join(primitives_summary)

    def plot_summary_dataset(self, dataset_path):
        """ Plot histograms of the dataset

        :param dataset_path: Path to dataset.  It supports D3M dataset, and CSV file
        """
        suffix = dataset_path.split('/')[-1]

        if is_d3m_format(dataset_path, suffix):
            dataset_path = join(dataset_path, 'dataset_%s/tables/learningData.csv' % suffix)

        plot_metadata(dataset_path)

    def plot_comparison_pipelines(self, test_dataset=None):
        """ Plot PipelineProfiler visualization

        :param test_dataset: Path to dataset. If None it will use the search scores, otherwise will score the
            pipelines over the passed dataset
        """
        pipelineprofiler_inputs = self.create_pipelineprofiler_inputs(test_dataset)
        plot_comparison_pipelines(pipelineprofiler_inputs)

    @staticmethod
    def add_new_ta2(ta2_id, docker_image_url):
        """Add a new TA2 system that is not already defined in the D3M Interface. It can also be a different version of
        a pre-existing TA2 (however it must be added with a different name)

        :param ta2_id: A id to identify the new TA2
        :param docker_image_url: The docker image url of the new TA2
        """
        TA2_DOCKER_IMAGES[ta2_id] = docker_image_url
        logger.info('%s TA2 added!', ta2_id)

