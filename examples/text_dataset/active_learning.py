import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from IPython import display
from matplotlib import pyplot as plt
from modAL.models import ActiveLearner, Committee
# Need to install: modAL==0.3.6
RANDOM_STATE_SEED = 0
np.random.seed(RANDOM_STATE_SEED)


def automatic_labeling(initial_dataset, unlabeled_dataset, validation_dataset, n_queries=100, keep_original=True):
    #cv_splitter = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    X_seed, y_seed = dataframe_to_nparray(initial_dataset, 'article', 'articleofinterest')
    X_validation, y_validation = dataframe_to_nparray(validation_dataset, 'article', 'articleofinterest')
    X_unlabeled, y_unlabeled = dataframe_to_nparray(unlabeled_dataset, 'article', 'articleofinterest')
    learner = create_committee_learner(X_seed, y_seed)
    d3m_indices = unlabeled_dataset['d3mIndex'].to_numpy()
    new_labeled_data = {'d3mIndex': [], 'article': [], 'articleofinterest': []}

    #score = cross_val_score(learner, X_validation, y_validation, scoring="f1", cv=cv_splitter).mean()
    #print('F1 before queries: {score:0.4f}'.format(score=score))
    predictions = learner.predict(X_validation)
    score = f1_score(y_validation, predictions)
    print('F1 before queries: {score:0.4f}'.format(score=score))
    performance_history = [score]

    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).
    for index in range(n_queries):
        query_index, query_instance = learner.query(X_unlabeled)
        # Teach our ActiveLearner model the record it has requested.
        X_instance, y_instance = X_unlabeled[query_index].reshape(1, ), y_unlabeled[query_index].reshape(1, )
        learner.teach(X=X_instance, y=y_instance)
        new_labeled_data['article'].append(str(X_instance[0]))
        new_labeled_data['articleofinterest'].append(int(y_instance[0]))
        new_labeled_data['d3mIndex'].append(d3m_indices[query_index][0])
        # Remove the queried instance from the unlabeled pool.
        X_unlabeled, y_unlabeled = np.delete(X_unlabeled, query_index, axis=0), np.delete(y_unlabeled, query_index, axis=0)
        d3m_indices = np.delete(d3m_indices, query_index, axis=0)                                                                           
        # Calculate and report our model's performance.
        #score = cross_val_score(learner, X_validation, y_validation, scoring="f1", cv=cv_splitter).mean()
        #print('F1 after query {n}: {score:0.4f}'.format(n=index + 1, score=score))
        predictions = learner.predict(X_validation)
        score = f1_score(y_validation, predictions)
        print('F1 after query {n}: {score:0.4f}'.format(n=index + 1, score=score))
        # Save our model's performance for plotting.
        performance_history.append(score)
    
    with plt.style.context('seaborn-white'):
            plt.figure(figsize=(8, 4))
            plt.title('F1 during the Active Learning')
            plt.plot(range(n_queries+1), performance_history)
            plt.scatter(range(n_queries+1), performance_history)
            plt.xlabel('Number of queries')
            plt.ylabel('F1')
            plt.ylim(0.0, 1.0)
            plt.xlim(0, n_queries)
            plt.show()
            
    new_labeled_data = pd.DataFrame.from_dict(new_labeled_data)
    
    if keep_original:
        return pd.concat([initial_dataset, new_labeled_data])
    else:
        return new_labeled_data


def manual_labeling(initial_dataset, unlabeled_dataset, validation_dataset, n_queries=100, keep_original=True):
    X_seed, y_seed = dataframe_to_nparray(initial_dataset, 'article', 'articleofinterest')
    X_validation, y_validation = dataframe_to_nparray(validation_dataset, 'article', 'articleofinterest')
    X_unlabeled, y_unlabeled = dataframe_to_nparray(unlabeled_dataset, 'article', 'articleofinterest')
    learner = create_single_learner(X_seed, y_seed)
    d3m_indices = unlabeled_dataset['d3mIndex'].to_numpy()
    new_labeled_data = {'d3mIndex': [], 'article': [], 'articleofinterest': []}

    predictions = learner.predict(X_validation)
    score = f1_score(y_validation, predictions)
    performance_history = [score]

    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).
    for index in range(n_queries):
        display.clear_output(wait=True)
        query_index, query_instance = learner.query(X_unlabeled)

        # Teach our ActiveLearner model the record it has requested.
        X_instance = X_unlabeled[query_index].reshape(1, )
        tmp_dataframe = pd.DataFrame(data={'Text': [X_instance[0]]})
        display.display(display.HTML(tmp_dataframe.to_html(index=False)))

        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(8, 4))
            plt.title('F1 during the Active Learning')
            plt.plot(range(index+1), performance_history)
            plt.scatter(range(index+1), performance_history)
            plt.xlabel('Number of queries')
            plt.ylabel('F1')
            plt.ylim(0.0, 1.0)
            plt.xlim(0, n_queries)
            plt.show()
        
        print('Which class is this text?')
        user_input = input()
        if user_input == 'exit': 
            break
        y_instance = np.array([int(user_input)], dtype=int)
        
        learner.teach(X=X_instance, y=y_instance)
        new_labeled_data['article'].append(str(X_instance[0]))
        new_labeled_data['articleofinterest'].append(int(y_instance[0]))
        new_labeled_data['d3mIndex'].append(d3m_indices[query_index][0])

        # Remove the queried instance from the unlabeled pool.
        X_unlabeled, y_unlabeled = np.delete(X_unlabeled, query_index, axis=0), np.delete(y_unlabeled, query_index)

        # Calculate and report our model's accuracy.
        predictions = learner.predict(X_validation)
        score = f1_score(y_validation, predictions)

        # Save our model's performance for plotting.
        performance_history.append(score)

    new_labeled_data = pd.DataFrame.from_dict(new_labeled_data)

    if keep_original:
        return pd.concat([initial_dataset, new_labeled_data])
    else:
        return new_labeled_data


def create_single_learner(X, y, estimator=None):
    if estimator is None:
        estimator = RandomForestClassifier(random_state=RANDOM_STATE_SEED, n_estimators=100) 
        # n_estimators=100 to avoid warning regarding n_estimators will change from 10 in version 0.20 to 100 in 0.22
        
    pipeline = Pipeline(
                        steps=[
                                ('tfidf', TfidfVectorizer()),
                                ('classifier', estimator),
                            ]
                        )

    return ActiveLearner(estimator=pipeline, X_training=X, y_training=y)


def create_committee_learner(X, y):
    learner_list = []
    # n_estimators=100 to avoid warning regarding n_estimators will change from 10 in version 0.20 to 100 in 0.22
    estimators = [RandomForestClassifier(random_state=RANDOM_STATE_SEED, n_estimators=100), 
                  GradientBoostingClassifier(random_state=RANDOM_STATE_SEED),
                  MultinomialNB()]
    
    for estimator in estimators:
        # initializing learner
        learner = create_single_learner(X, y, estimator)
        learner_list.append(learner)
    
    return Committee(learner_list=learner_list)


def dataframe_to_nparray(dataset, text_column, label_column):
    X = dataset[text_column].to_numpy().astype('U')
    y = dataset[label_column].to_numpy()

    return X, y
