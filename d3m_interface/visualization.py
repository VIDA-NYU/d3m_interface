import datamart_profiler
import DataProfileViewer
from d3m.utils import silence


def plot_metadata(dataset_path):
    with silence():
        metadata = datamart_profiler.process_dataset(dataset_path, plots=True, include_sample=True)

    DataProfileViewer.plot_data_summary(metadata)
