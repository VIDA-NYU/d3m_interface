import pandas as pd
import altair as alt


def histogram_summaries(metadata):
    chart_t = alt.hconcat()
    for column in metadata['columns']:
        if 'plot' in column:
            col_type = column['plot']['type']
            plot_data = convert_to_dataframe(column['plot']['data'])
            if col_type == 'histogram_numerical':
                chart = alt.Chart(plot_data).mark_bar().encode(
                    x=alt.X('bin_start', bin='binned', axis=alt.Axis(title='')),
                    x2='bin_end',
                    y=alt.Y('count', axis=alt.Axis(title='')),
                    tooltip=['bin_start', 'bin_end']
                ).properties(title={'text': column['name']},
                             width=100,
                             height=100
                             )
            if col_type == 'histogram_categorical':
                chart = alt.Chart(plot_data).mark_bar().encode(
                    x=alt.X('bin', axis=alt.Axis(title='')),
                    y=alt.Y('count', axis=alt.Axis(title='')),
                    tooltip=['bin']
                ).properties(
                    title=column['name'],
                    width=100,
                    height=100
                )
            if col_type == 'histogram_temporal':
                chart = alt.Chart(plot_data).mark_bar().encode(
                    x=alt.X('date_start', bin='binned'),
                    x2='date_end',
                    y='count',
                    tooltip=['date_start', 'date_end']
                ).properties(
                    title=column['name'],
                    width=100,
                    height=100
                )
            if col_type == 'histogram_text':
                chart = alt.Chart(plot_data).mark_bar().encode(
                    y=alt.X('bin', axis=alt.Axis(title='')),
                    x=alt.Y('count', axis=alt.Axis(title='')),
                    tooltip=['bin', 'count']
                ).properties(
                    title=column['name'],
                    width=100,
                    height=100
                )
            chart.configure_title(
                fontSize=14,
                font='Courier',
                anchor='middle',
                color='gray'
            )
            chart_t |= chart
    chart_t.display()


def convert_to_dataframe(data):
    data_dict = {}
    for key in data[0]:
        data_dict[key] = [element[key] for element in data]
    df = pd.DataFrame(data_dict)

    return df
