
import datetime as datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import datetime as datetime
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def plot_timeseries(df, col, date_untill='2024-05-15'):
    df_plot = df.loc[df.index < date_untill].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], mode='lines', name=col, line=dict(color='#0084B2')))
    if col == 'Cliënten':
        y_axis_title = f'Aantal cliënten'
    elif col == 'Ziekteverzuim':
        y_axis_title = f'Ziekteverzuimpercentage (%)'
    elif col == 'Inkoop':
        y_axis_title = f'Kosten inkoop (euro)'
    elif col == 'Flexpool':
        y_axis_title = f'Flexpool (aantal personen)'
    else: 
        y_axis_title = f'Onbekende eenheid'
    fig.update_layout(title=f'Weergave van {col} over de tijd', 
                      xaxis_title='Datum', 
                      yaxis_title=y_axis_title,
                      plot_bgcolor='white',
            paper_bgcolor='white')
    fig.show()


def plot_prediction_with_shapes(df, start_train=None, end_train=None, start_test=None, end_test=None, shapes=False, show_y_zero=False, yaxis_title="Ziekteverzuimpercentage (%)"):

    color_dict = {'Cliënten':'#0084B2', 
                  'Ziekteverzuim': '#0084B2', # #0084B2 #646A69
                  'Inkoop': '#0084B2', 
                  'Flexpool': '#0084B2',
                  'Gemiddelde': '#F8AF5E', 
                  'Voortschrijdend gemiddelde': '#a55233', # #a55233 #0084B2
                  'Regressie': '#402a23',
                  'Gemiddelde error': '#F8AF5E', 
                  'Voortschrijdend gemiddelde error': '#a55233', # #a55233 #0084B2
                  'Regressie error': '#402a23',
                  'Optie 1': '#F8AF5E',
                  'Optie 2': '#5C605F',
                  'Optie 3': '#BCAECE',
                  'Optie 4': '#fe4a49',
                  }  # #F85EF4 #402a23 #AF5EF8
    ## Plot results
    fig = go.Figure()

    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[col],
            mode='lines', 
            name=str(col),
            line=dict(color=color_dict[col])))
        
    shape_train = dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start_train,
                    y0=0,
                    x1=end_train,
                    y1=1,
                    fillcolor="limegreen",
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                )
    shape_test = dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start_test,
                    y0=0,
                    x1=end_test,
                    y1=1,
                    fillcolor="LightSalmon",
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                )
    
    fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            autosize=False,
            width=1000,  # Set the width to the desired value
            xaxis_title="Datum",
            yaxis_title=yaxis_title,
    )

    if shapes:
        if (start_train is not None) and (end_train is not None):
            fig.add_shape(shape_train)
            fig.add_trace(go.Scatter(
            x=[None],  # No data points
            y=[None],
            mode='lines',
            name='Trainperiode',  # Replace with your shape name
            line=dict(color='limegreen'),
            opacity=0.15
        ))
        if (start_test is not None) and (end_test is not None):
            fig.add_shape(shape_test)
            fig.add_trace(go.Scatter(
            x=[None],  # No data points
            y=[None],
            mode='lines',
            name='Testperiode',  # Replace with your shape name
            line=dict(color='LightSalmon'),
            opacity=0.3
        ))
    else:
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            # Misschien nog witte lijnen toevoegen? 
        )
    
    if show_y_zero:
        fig.add_shape(
            type="line",
            x0=df.index.min(),
            y0=0,
            x1=df.index.max(),
            y1=0,
            line=dict(
                color="#999999",
                width=2,
                dash="dash"
            )
        )

    fig.show()

def make_error_df(df, base_col, with_base_col=False):
    _df = df.copy()
    for col in _df.columns:
        _df[col + ' error'] = _df[col] - _df[base_col]
    if with_base_col:
        return _df
    else:
        error_cols = [col for col in _df.columns if ('error' in col) and (base_col not in col)]
        return _df[error_cols]

def plot_errors(df, base_col, start, end, yaxis_title, shapes=False, show_y_zero=True):

    _df = df.loc[start:end].copy()

    df_error = make_error_df(_df, base_col)

    # error_cols = [col for col in _df.columns if ('error' in col) and (base_col not in col)]

    plot_prediction_with_shapes(df_error, shapes=shapes, show_y_zero=show_y_zero, yaxis_title=yaxis_title)

def plot_distribution(df, base_col, start, end):
    _df = df.loc[start:end].copy()
    df_error = make_error_df(_df, base_col)

    hist_data = [df_error[col] for col in df_error.columns]
    group_labels = list(df_error.columns)

    color_dict = {'Cliënten':'#0084B2', 
                    'Ziekteverzuim': '#0084B2', # #0084B2 #646A69
                    'Inkoop': '#0084B2', 
                    'Flexpool': '#0084B2',
                    'Gemiddelde': '#F8AF5E', 
                    'Voortschrijdend gemiddelde': '#a55233', # #a55233 #0084B2
                    'Regressie': '#402a23',
                    'Gemiddelde error': '#F8AF5E', 
                    'Voortschrijdend gemiddelde error': '#a55233', # #a55233 #0084B2
                    'Regressie error': '#402a23',
                    } 
    colors = [v for k,v in color_dict.items() if k in group_labels]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, colors=colors)

    fig.layout.plot_bgcolor = '#ffffff'
    fig.update_layout(title_text="Distributie van de errors")

    fig.show()

# def onderzoek_afwijkingen(df, onderwerp, start, end):
#     plot_errors(df=df, base_col=onderwerp, start=start, end=end)
#     plot_distribution(df=df, base_col=onderwerp, start=start, end=end)

def accuracy(y_true, y_pred):
    return sum(y_pred)/sum(y_true)

def calc_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calc_sum_of_errors(y_true, y_pred):
    return np.sum(y_true - y_pred)

def calc_average_error(y_true, y_pred):
    return np.mean(y_true - y_pred)

def calc_max_error(y_true, y_pred):
    return np.max(y_true - y_pred)

def calc_MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calc_MAPE(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def root_mean_squared_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
):
    """Root mean squared error regression loss.

    Read more in the :ref:`User Guide <mean_squared_error>`.

    .. versionadded:: 1.4

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sklearn.metrics import root_mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> root_mean_squared_error(y_true, y_pred)
    0.612...
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> root_mean_squared_error(y_true, y_pred)
    0.822...
    """
    output_errors = np.sqrt(
        mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values"
        )
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def calc_RMSE(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

# def bereken_metrieken(df, onderwerp, start, end):
#     _df = df.loc[start:end].copy()
#     print(f"De periode die wordt geanalyseerd is van {start} tot {end.date()}.")
#     df_metrics = pd.DataFrame()
#     metric_cols = [col for col in _df.columns if onderwerp not in col]
#     for col in metric_cols:
#         for metric in [calc_sum_of_errors, calc_average_error, calc_max_error, calc_MAE, calc_MAPE, calc_RMSE]:
#             df_metrics.loc[col, metric.__name__] = metric(_df[onderwerp], _df[col])
#     df_metrics.rename(columns={'calc_sum_of_errors': 'Totale afwijking', 
#                                'calc_average_error': 'Gemiddelde afwijking', 
#                                'calc_max_error': 'Maximale afwijking',
#                                'calc_MAE': 'Mean Absolute Error', 
#                                'calc_MAPE': 'Mean Absolute Percentage Error', 
#                                'calc_RMSE': 'Root Mean Squared Error'}, 
#                                inplace=True)
#     return df_metrics