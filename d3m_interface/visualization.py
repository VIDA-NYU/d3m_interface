import datamart_profiler
import DataProfileViewer
import PipelineProfiler
import VisualTextAnalyzer
from d3m.utils import silence


def plot_metadata(dataset_path):
    with silence():
        metadata = datamart_profiler.process_dataset(dataset_path, plots=True, include_sample=True)

    DataProfileViewer.plot_data_summary(metadata)


def plot_comparison_pipelines(pipelines):
    PipelineProfiler.plot_pipeline_matrix(pipelines)


def plot_text_summary(dataset, text_column, label_column, positive_label, negative_label):
    VisualTextAnalyzer.plot_text_summary(dataset, text_column=text_column, category_column=label_column,
                                         positive_label=positive_label, negative_label=negative_label)
