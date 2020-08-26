import grpc
import logging
import ta3ta2_api.core_pb2 as pb_core
import ta3ta2_api.core_pb2_grpc as pb_core_grpc
import ta3ta2_api.value_pb2 as pb_value
from d3m.metadata.problem import Problem, PerformanceMetric
from ta3ta2_api.utils import encode_problem_description, encode_performance_metric, decode_performance_metric, \
    decode_value, decode_pipeline_description
from d3m.utils import fix_uri, silence
from d3m.metadata import pipeline as pipeline_module


logger = logging.getLogger(__name__)


class BasicTA3:
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:45042')
        self.core = pb_core_grpc.CoreStub(self.channel)

    def do_hello(self):
        self.core.Hello(pb_core.HelloRequest())

    def do_listprimitives(self):
        self.core.ListPrimitives(pb_core.ListPrimitivesRequest())

    def do_search(self, dataset_path, problem_path, time_bound, time_bound_run, pipeline_template=None):
        problem = Problem.load(problem_uri=fix_uri(problem_path))
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[pb_core.protocol_version]

        search = self.core.SearchSolutions(pb_core.SearchSolutionsRequest(
            user_agent='basicta3_stub',
            version=version,
            time_bound_search=time_bound,
            time_bound_run=time_bound_run,
            priority=10,
            rank_solutions_limit=0,
            allowed_value_types=['RAW', 'DATASET_URI', 'CSV_URI'],
            template=pipeline_template,
            problem=encode_problem_description(problem),

            inputs=[pb_value.Value(
                dataset_uri='file://%s' % dataset_path,
            )],
        ))

        results = self.core.GetSearchSolutionsResults(
            pb_core.GetSearchSolutionsResultsRequest(
                search_id=search.search_id,
            )
        )

        for result in results:
            if result.solution_id:
                pipeline_id = result.solution_id
                yield {'id': pipeline_id, 'search_id': str(search.search_id)}

    def do_score(self, solution_id, dataset_path, problem_path):
        problem = Problem.load(problem_uri=fix_uri(problem_path))
        metrics = []
        score_data = None

        for metric in problem['problem']['performance_metrics']:
            metrics.append(encode_performance_metric(metric))
        # Showing only the first metric
        target_metric = problem['problem']['performance_metrics'][0]['metric']

        response = self.core.ScoreSolution(pb_core.ScoreSolutionRequest(
            solution_id=solution_id,
            inputs=[pb_value.Value(
                dataset_uri='file://%s' % dataset_path,
            )],
            performance_metrics=metrics,
            users=[],
            configuration=pb_core.ScoringConfiguration(
                method='HOLDOUT',
                train_test_ratio=0.75,
                shuffle=True,
                random_seed=0
            ),
        ))

        results = self.core.GetScoreSolutionResults(
            pb_core.GetScoreSolutionResultsRequest(
                request_id=response.request_id,
            )
        )

        for result in results:
            if result.progress.state == pb_core.COMPLETED:
                scores = []
                for metric_score in result.scores:
                    metric = decode_performance_metric(metric_score.metric)['metric']
                    if metric == target_metric:
                        score = decode_value(metric_score.value)['value']
                        scores.append(score)

                if len(scores) > 0:
                    avg_score = round(sum(scores) / len(scores), 5)
                    normalized_score = PerformanceMetric[target_metric.name].normalize(avg_score)

                    score_data = {'score': avg_score, 'normalized_score': normalized_score,
                                  'metric': target_metric.name.lower()}
        if score_data is None:
            raise TypeError('Pipeline got a None value during scoring')

        return score_data

    def do_train(self, solution_id, dataset_path):
        fitted_solution = None

        response = self.core.FitSolution(pb_core.FitSolutionRequest(
            solution_id=solution_id,
            inputs=[pb_value.Value(
                dataset_uri='file://%s' % dataset_path,
            )],
            expose_outputs=[],
            expose_value_types=['CSV_URI'],
            users=[],
        ))
        results = self.core.GetFitSolutionResults(
            pb_core.GetFitSolutionResultsRequest(
                request_id=response.request_id,
            )
        )
        for result in results:
            if result.progress.state == pb_core.COMPLETED:
                fitted_solution = result.fitted_solution_id

        if fitted_solution is None:
            raise TypeError('Pipeline got a None value during training')

        return fitted_solution

    def do_test(self, fitted_solution_id, dataset_path):
        tested_solution = None

        response = self.core.ProduceSolution(pb_core.ProduceSolutionRequest(
            fitted_solution_id=fitted_solution_id,
            inputs=[pb_value.Value(
                dataset_uri='file://%s' % dataset_path,
            )],
            expose_outputs=['outputs.0'],
            expose_value_types=['CSV_URI'],
            users=[],
        ))
        results = self.core.GetProduceSolutionResults(
            pb_core.GetProduceSolutionResultsRequest(
                request_id=response.request_id,
            )
        )
        for result in results:
            if result.progress.state == pb_core.COMPLETED:
                tested_solution = result.exposed_outputs['outputs.0'].csv_uri

        if tested_solution is None:
            raise TypeError('Pipeline got a None value during testing')

        return tested_solution

    def do_export(self, fitted):
        for i, fitted_solution in enumerate(fitted.values()):
            try:
                self.core.SolutionExport(pb_core.SolutionExportRequest(
                    solution_id=fitted_solution,
                    rank=(i + 1.0) / (len(fitted) + 1.0),
                ))
            except:
                logger.exception('Exception exporting %r', fitted_solution)

    def do_describe(self, solution_id):
        pipeline = None
        pipeline_description = self.core.DescribeSolution(pb_core.DescribeSolutionRequest(solution_id=solution_id)).pipeline

        with silence():
            pipeline = decode_pipeline_description(pipeline_description, pipeline_module.NoResolver())

        if pipeline is None:
            raise TypeError('Pipeline got a None value during decoding')

        return pipeline.to_json_structure()

    def do_stop_search(self, search_id):
        self.core.StopSearchSolutions(pb_core.StopSearchSolutionsRequest(search_id=search_id))
