import datamart_profiler
import DataProfileViewer
import PipelineProfiler
import VisualTextAnalyzer
import numpy as np
import pandas as pd
from os.path import join
from d3m.utils import silence
from d3m_interface.data_converter import create_artificial_d3mtest
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def plot_metadata(dataset_path):
    with silence():
        metadata = datamart_profiler.process_dataset(dataset_path, plots=True, include_sample=True)

    DataProfileViewer.plot_data_summary(metadata)


def plot_comparison_pipelines(pipelines):
    PipelineProfiler.plot_pipeline_matrix(pipelines)


def plot_text_summary(dataset, text_column, label_column, positive_label, negative_label):
    VisualTextAnalyzer.plot_text_summary(dataset, text_column=text_column, category_column=label_column,
                                         positive_label=positive_label, negative_label=negative_label)


def plot_text_explanation(automl, train_path, artificial_test_path, model_id, instance_text, text_column, label_column, num_features, top_labels):
    train_data = pd.read_csv(join(train_path, 'dataset_TRAIN', 'tables', 'learningData.csv'))
    class_label_encoder = LabelEncoder()
    numerical_labels = class_label_encoder.fit_transform(train_data[label_column])
    class_one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    class_one_hot_encoder.fit(numerical_labels.reshape([-1, 1]))
    explainer = LimeTextExplainer(class_names=np.array(class_label_encoder.classes_))

    def predict_proba(instance_text_list):
        # Computing prediction probabilities
        create_artificial_d3mtest(train_path, artificial_test_path, instance_text_list, label_column, text_column)
        probabilities = automl.test(model_id, artificial_test_path, calculate_confidence=True)
        probabilities = probabilities.pivot(index='d3mIndex', columns=label_column, values='confidence')
        probabilities = probabilities.values

        return probabilities

    with silence():  # Hide some warnings from LIME
        explanation = explainer.explain_instance(instance_text, predict_proba, num_features=num_features, top_labels=top_labels)
    explanation.show_in_notebook()

