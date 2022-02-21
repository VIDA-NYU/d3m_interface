from d3m_interface import AutoML

output_path = '/Users/rlopez/D3M/examples/tmp/'
train_dataset = '/Users/rlopez/D3M/examples/185_baseball/TRAIN'
test_dataset = '/Users/rlopez/D3M/examples/185_baseball/TEST'
score_dataset = '/Users/rlopez/D3M/examples/185_baseball/SCORE'


# Simple search
automl = AutoML(output_path, 'AlphaD3M')
automl.search_pipelines(train_dataset, time_bound=1)
best_pipeline_id = automl.get_best_pipeline_id()
model_id = automl.train(best_pipeline_id)
automl.end_session()


# Blacklist primitives
automl = AutoML(output_path, 'AlphaD3M')
primitives = automl.list_primitives()
print(primitives)
blacklist = ['d3m.primitives.classification.ada_boost.SKlearn']
automl.search_pipelines(train_dataset, time_bound=1, blacklist_primitives=blacklist)
automl.end_session()


# Run using PyPI installation
automl = AutoML(output_path, 'AlphaD3M', 'pypi')
automl.search_pipelines(train_dataset, time_bound=1)
automl.end_session()
