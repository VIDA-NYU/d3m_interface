import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from active_learning import create_committee_learner, dataframe_to_nparray

RATIO = 2


def progressive_sample(initial_dataset, pool_dataset, validation_dataset, threshold=0.90, keep_original=True):
    sample_sizes = geometric_progression(len(initial_dataset), len(pool_dataset))
    n_queries = len(sample_sizes)
    current_size = len(initial_dataset)
    X_seed, y_seed = dataframe_to_nparray(initial_dataset, 'article', 'articleofinterest')
    X_validation, y_validation = dataframe_to_nparray(validation_dataset, 'article', 'articleofinterest')
    X_pool, y_pool = dataframe_to_nparray(pool_dataset, 'article', 'articleofinterest')
    learner = create_committee_learner(X_seed, y_seed)

    d3m_indices = pool_dataset['d3mIndex'].to_numpy()
    new_labeled_data = {'d3mIndex': [], 'article': [], 'articleofinterest': []}

    predictions = learner.predict(X_validation)
    score = f1_score(y_validation, predictions)
    print('F1 score (dataset size={size}): {score:0.4f}'.format(size=current_size, score=score))
    performance_history = [score]

    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).
    for index in range(n_queries):
        batch_size = sample_sizes[index] - current_size
        current_size = sample_sizes[index]
        query_index, query_instance = learner.query(X_pool, n_instances=batch_size)
        # Teach our ActiveLearner model the record it has requested.
        X_instance, y_instance = X_pool[query_index], y_pool[query_index]
        learner.teach(X=X_instance, y=y_instance)
        new_labeled_data['article'].extend(X_instance.tolist())
        new_labeled_data['articleofinterest'].extend(y_instance.tolist())
        new_labeled_data['d3mIndex'].extend(d3m_indices[query_index].tolist())
        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index, axis=0)
        d3m_indices = np.delete(d3m_indices, query_index, axis=0)
        # Calculate and report our model's performance.
        predictions = learner.predict(X_validation)
        score = f1_score(y_validation, predictions)
        print('F1 score (dataset size={size}): {score:0.4f}'.format(size=current_size, score=score))
        # Save our model's performance for plotting.
        performance_history.append(score)

        if score >= threshold:
            print('Threshold reached!')
            break

    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(8, 4))
        plt.title('F1 during Progressive Sampling')
        plt.plot(range(len(performance_history)), performance_history)
        plt.scatter(range(len(performance_history)), performance_history)
        plt.xlabel('Number of iterations')
        plt.ylabel('F1')
        plt.ylim(0.0, 1.0)
        plt.xlim(0, len(performance_history) - 1)
        plt.show()

    new_labeled_data = pd.DataFrame.from_dict(new_labeled_data)

    if keep_original:
        return pd.concat([initial_dataset, new_labeled_data])
    else:
        return new_labeled_data


def geometric_progression(initial_size, max_size):
    current_size = initial_size * RATIO
    sizes = []

    while current_size < max_size:
        sizes.append(current_size)
        current_size = current_size * RATIO

    return sizes
