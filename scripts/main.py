import datetime as datetime
import pandas as pd
from scripts.evaluate.evaluate import plot_prediction_with_shapes
from scripts.preprocess.preprocess import load_timeseries_data, collect_str_input, make_X_y, combine_dfs_of_models, calibrate_dates
from scripts.evaluate.evaluate import plot_timeseries, plot_errors, plot_distribution, accuracy, calc_r2_score, calc_sum_of_errors, calc_average_error, calc_max_error, calc_MAE, calc_MAPE, calc_RMSE
from scripts.model.model import calc_average, predict_with_average, calc_moving_average, LinearRegressionTrain, LinearRegressionPredict

def bekijk_data():
    df = load_timeseries_data()
    df = df.drop(columns=['Inkoop'])
    for col in df.columns:
        plot_timeseries(df, col)

    return df

def laad_data():
    df = load_timeseries_data()
    df = df.drop(columns=['Inkoop'])
    return df

def bekijk_ziekteverzuim(df):
    plot_timeseries(df, 'Ziekteverzuim')

def bekijk_clienten(df):
    plot_timeseries(df, 'Cliënten')

def bekijk_flexpool(df):
    plot_timeseries(df, 'Flexpool')

def kies_onderwerp():
    vraag = """
        Welke reeks wil je voorspellen (a, b, c):
                a. Cliënten (Zorgzwaartepakket 6 of hoger)
                b. Ziekteverzuim(percentage)
                c. Flexpool (aantal personen)
        """
    mogelijke_antwoorden = ['a', 'b', 'c', 'clienten', 'cliënten', 'ziekteverzuim', 'ziekteverzuimpercentage', 'flexpool']

    str = collect_str_input(
        question=vraag, 
        possible_entries=mogelijke_antwoorden)
    dict_antwoorden = {'a': 'Cliënten', 'clienten': 'Cliënten', 'cliënten': 'Cliënten',
                       'b': 'Ziekteverzuim', 'ziekteverzuim': 'Ziekteverzuim', 'ziekteverzuimpercentage': 'Ziekteverzuim',
                       'c': 'Flexpool', 'flexpool': 'Flexpool'}
    print(f'Gekozen antwoord: {dict_antwoorden[str]}')
    return dict_antwoorden[str]

def pas_gemiddelde_toe(df, 
                     onderwerp='Ziekteverzuim', 
                     vanaf_datum_train_periode = None,
                     tot_datum_train_periode = None,
                     vanaf_datum_test_periode = None,
                     tot_datum_test_periode = None,
                     zie_traintest_periodes=False):
    
    vanaf_datum_train_periode, tot_datum_train_periode, \
        vanaf_datum_test_periode, tot_datum_test_periode = calibrate_dates(vanaf_datum_train_periode,
                                                                           tot_datum_train_periode, 
                                                                           vanaf_datum_test_periode, 
                                                                           tot_datum_test_periode)

    df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
                                                            onderwerp, 
                                                            vanaf_datum_train_periode, 
                                                            tot_datum_train_periode, 
                                                            vanaf_datum_test_periode, 
                                                            tot_datum_test_periode)

    # Apply average model
    # Calculate the average of the train data
    average = calc_average(df_y_train)
    # Predict the average for the train data
    y_preds_train_avg = predict_with_average(df_y_train, average)
    # Predict the average for the test data
    y_preds_test_avg = predict_with_average(df_y_test, average)
    y_avg = pd.concat([y_preds_train_avg, y_preds_test_avg])

    df_y_all = pd.DataFrame()
    df_y_all['Gemiddelde'] = y_avg

    df_real = pd.concat([df_y_train, df_y_test], axis=0)
    df_real.name = onderwerp
    df_total = pd.concat([df_real, df_y_all], axis=1)

    plot_prediction_with_shapes(df=df_total, 
                                start_train=vanaf_datum_train_periode, 
                                end_train=tot_datum_train_periode, 
                                start_test=vanaf_datum_test_periode, 
                                end_test=tot_datum_test_periode, 
                                shapes=zie_traintest_periodes)

    return df_total

def pas_voortschrijdend_gemiddelde_toe(data, 
                     onderwerp='Ziekteverzuim', 
                     vanaf_datum_train_periode = None,
                     tot_datum_train_periode = None,
                     vanaf_datum_test_periode = None,
                     tot_datum_test_periode = None,
                     vensterlengte = 7,
                     verschuiving = 0,
                     predict=False,
                     zie_traintest_periodes=True,
                     plot=True):
    _data = data.copy()
    if (isinstance(vensterlengte, int) is False) | (vensterlengte < 1):
        raise ValueError("De vensterlengte moet een geheel getal zijn én groter of gelijk aan 1.")
    if (isinstance(verschuiving, int) is False) | (verschuiving < 0):
        raise ValueError("De verschuiving moet een geheel getal zijn én groter of gelijk aan 0.")

    vanaf_datum_train_periode, tot_datum_train_periode, \
        vanaf_datum_test_periode, tot_datum_test_periode = calibrate_dates(vanaf_datum_train_periode,
                                                                           tot_datum_train_periode, 
                                                                           vanaf_datum_test_periode, 
                                                                           tot_datum_test_periode)
    
    if vensterlengte < 1:
        raise ValueError("De vensterlengte moet groter of gelijk aan 1 zijn.")
    if verschuiving < 0:
        raise ValueError("De verschuiving moet groter of gelijk aan 0 zijn.")
    
    df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(_data, 
                                                            onderwerp, 
                                                            vanaf_datum_train_periode, 
                                                            tot_datum_train_periode, 
                                                            vanaf_datum_test_periode, 
                                                            tot_datum_test_periode)

    # Apply moving average model
    # Calculate the moving average of the training data
    if predict:
        df_y_base = pd.concat([df_y_train, df_y_test])
    else:
        df_y_test_nan = pd.Series(index=df_y_test.index)
        df_y_test_nan.name = df_y_test.name
        df_y_base = pd.concat([df_y_train, df_y_test_nan])
    y_preds_train_mov_avg = calc_moving_average(df_y_base, 
                                                window_size=vensterlengte, 
                                                shift_period=verschuiving, 
                                                predict_to_date=tot_datum_test_periode,
                                                predict=predict)

    df_y_all = pd.DataFrame()
    df_y_all['Voortschrijdend gemiddelde'] = y_preds_train_mov_avg

    df_real = pd.concat([df_y_train, df_y_test], axis=0)
    df_real.name = onderwerp
    df_total = pd.concat([df_real, df_y_all], axis=1)

    
    if plot is True:
        plot_prediction_with_shapes(df=df_total, 
                                    start_train=vanaf_datum_train_periode, 
                                    end_train=tot_datum_train_periode, 
                                    start_test=vanaf_datum_test_periode, 
                                    end_test=tot_datum_test_periode, 
                                    shapes=zie_traintest_periodes)

    return df_total

def pas_regressie_toe(data, 
                     onderwerp='Ziekteverzuim', 
                     vanaf_datum_train_periode = None,
                     tot_datum_train_periode = None,
                     vanaf_datum_test_periode = None,
                     tot_datum_test_periode = None,
                     jaarlijks_seizoenspatroon=False,
                     wekelijks_seizoenspatroon=False,
                     transformatie=None, n_bins=4, strategy='uniform', n_knots=2, graad=1,
                     zie_traintest_periodes=True,
                     plot=True):
    _data = data.copy()
    if transformatie is None:
        if onderwerp == 'Cliënten':
            transformatie = 'lineair'
        elif onderwerp in ['Ziekteverzuim', 'Flexpool']:
            transformatie = 'spline'
    transformatie = str(transformatie).lower()
    if isinstance(jaarlijks_seizoenspatroon, bool) is False:
        raise ValueError("Het getal voor jaarlijks_seizoenspatroon moet een boolean (True/False) zijn.")
    if isinstance(wekelijks_seizoenspatroon, bool) is False:
        raise ValueError("Het getal voor wekelijks_seizoenspatroon moet een boolean (True/False) zijn.")
    if n_knots < 2:
        raise ValueError("Het getal voor k_nots moet 2 of groter zijn.")
    if (isinstance(graad, int) is False) | (graad < 1):
        raise ValueError("Het getal voor graad moet een geheel getal zijn én groter of gelijk aan 1.")
    if transformatie not in ['lineair', 'binnes', 'spline', 'polynomial']:
        raise ValueError("De transformatie moet 'lineair', 'binnes', 'spline' of 'polynomial' zijn.")

    vanaf_datum_train_periode, tot_datum_train_periode, \
        vanaf_datum_test_periode, tot_datum_test_periode = calibrate_dates(vanaf_datum_train_periode,
                                                                           tot_datum_train_periode, 
                                                                           vanaf_datum_test_periode, 
                                                                           tot_datum_test_periode)

    df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(_data, 
                                                            onderwerp, 
                                                            vanaf_datum_train_periode, 
                                                            tot_datum_train_periode, 
                                                            vanaf_datum_test_periode, 
                                                            tot_datum_test_periode)
    
    # Apply Linear Regression model
    # Get Linear Regression model
    model = LinearRegressionTrain(df_X_train, 
                                  df_y_train, 
                                  yearly_seasonality=jaarlijks_seizoenspatroon, 
                                  weekly_seasonality=wekelijks_seizoenspatroon,
                                  transformation=transformatie, n_bins=n_bins, strategy=strategy, n_knots=n_knots, degree=graad)
    # Make predictions with the Linear Regression model for the train data
    y_preds_train_lin_reg = LinearRegressionPredict(df_X_train, 
                                                    model, 
                                                    yearly_seasonality=jaarlijks_seizoenspatroon, 
                                                    weekly_seasonality=wekelijks_seizoenspatroon)
    # Make predictions with the Linear Regression model for the test data
    y_preds_test_lin_reg = LinearRegressionPredict(df_X_test, 
                                                   model, 
                                                   yearly_seasonality=jaarlijks_seizoenspatroon, 
                                                   weekly_seasonality=wekelijks_seizoenspatroon)
    y_lin_reg = pd.concat([y_preds_train_lin_reg, y_preds_test_lin_reg])

    df_real = pd.concat([df_y_train, df_y_test], axis=0)
    df_real.name = onderwerp

    df_y_all = pd.DataFrame()
    df_y_all['Regressie'] = y_lin_reg
    df_y_all.index = df_real.index
    df_total = pd.concat([df_real, df_y_all], axis=1)

    if plot is True:
        plot_prediction_with_shapes(df=df_total, 
                                    start_train=vanaf_datum_train_periode, 
                                    end_train=tot_datum_train_periode, 
                                    start_test=vanaf_datum_test_periode, 
                                    end_test=tot_datum_test_periode, 
                                    shapes=zie_traintest_periodes)

    return df_total

def voorspel(
        data,
        onderwerp='Ziekteverzuim',
        voorspellen_tot_datum=None,
        vensterlengte=None,
        verschuiving=None,
        jaarlijks_patroon=None,
        wekelijks_patroon=None,
        graad=None,
        model=None,
        zie_traintest_periodes=False
):
    _data = data.copy()
    vanaf_datum_train_periode = _data.index.min()
    maximum_data_dataset = datetime.datetime(2024,12,31)
    date_yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    maximum_date = min(date_yesterday, maximum_data_dataset)
    tot_datum_train_periode = maximum_date
    vanaf_datum_test_periode = maximum_date
    tot_datum_test_periode = voorspellen_tot_datum
    if model is None:
        if onderwerp == 'Flexpool':
            model = 'voortschrijdend_gemiddelde'
        elif onderwerp in ['Cliënten', 'Ziekteverzuim']:
            model = 'regressie'
        else:
            raise ValueError("Onderwerp niet gevonden. Kies uit 'Cliënten', 'Ziekteverzuim' of 'Flexpool'.")
    else:
        model = str(model).lower()

    if onderwerp == 'Cliënten':
        ## Instellingen model voortschrijdend gemiddelde
        vensterlengte = 7
        verschuiving = 7

        ## Instellingen regressiemodel
        jaarlijks_patroon=False
        wekelijks_patroon=False
        graad=1 # nvt
    elif onderwerp == 'Ziekteverzuim':
        ## Instellingen model voortschrijdend gemiddelde
        vensterlengte = 14
        verschuiving = 365

        ## Instellingen regressiemodel
        jaarlijks_patroon=True
        wekelijks_patroon=True
        graad=4
    elif onderwerp == 'Flexpool':
        ## Instellingen model voortschrijdend gemiddelde
        vensterlengte = 21
        verschuiving = 1

        ## Instellingen regressiemodel
        jaarlijks_patroon=False
        wekelijks_patroon=False
        graad=9
    
    
    if model not in ['voortschrijdend_gemiddelde', 'regressie']:
        raise ValueError("Model niet gevonden. Kies uit 'voortschrijdend_gemiddelde' of 'regressie'")
    if model == 'voortschrijdend_gemiddelde':
        df = pas_voortschrijdend_gemiddelde_toe(
            data=_data,
            onderwerp=onderwerp,
            vensterlengte=vensterlengte,
            verschuiving=verschuiving,
            vanaf_datum_train_periode = vanaf_datum_train_periode,
            tot_datum_train_periode = tot_datum_train_periode,
            vanaf_datum_test_periode = vanaf_datum_test_periode,
            tot_datum_test_periode = tot_datum_test_periode,
            zie_traintest_periodes=zie_traintest_periodes
        )
    elif model == 'regressie':
        df = pas_regressie_toe(
            data=_data,
            onderwerp=onderwerp,
            jaarlijks_seizoenspatroon=jaarlijks_patroon,
            wekelijks_seizoenspatroon=wekelijks_patroon,
            graad=graad,
            vanaf_datum_train_periode = vanaf_datum_train_periode,
            tot_datum_train_periode = tot_datum_train_periode,
            vanaf_datum_test_periode = vanaf_datum_test_periode,
            tot_datum_test_periode = tot_datum_test_periode,
            zie_traintest_periodes=zie_traintest_periodes
        )
    else:
        df = _data

    return df

def voorspel_met_voortschrijdend_gemiddelde(data,
        onderwerp='Ziekteverzuim',
        voorspellen_tot_datum='2025-01-01',
        vensterlengte=7,
        verschuiving=0,
        jaarlijks_patroon=None,
        wekelijks_patroon=None,
        graad=None,
        model=None,
        zie_traintest_periodes=False):
    
    _data = data.copy()
    vanaf_datum_train_periode = _data.index.min()
    maximum_data_dataset = datetime.datetime(2024,12,31)
    date_yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    maximum_date = min(date_yesterday, maximum_data_dataset)
    tot_datum_train_periode = maximum_date
    vanaf_datum_test_periode = maximum_date
    tot_datum_test_periode = voorspellen_tot_datum

    df = pas_voortschrijdend_gemiddelde_toe(
            data=_data,
            onderwerp=onderwerp,
            vensterlengte=vensterlengte,
            verschuiving=verschuiving,
            vanaf_datum_train_periode = vanaf_datum_train_periode,
            tot_datum_train_periode = tot_datum_train_periode,
            vanaf_datum_test_periode = vanaf_datum_test_periode,
            tot_datum_test_periode = tot_datum_test_periode,
            zie_traintest_periodes=zie_traintest_periodes
        )
    return df

def voorspel_met_regressie(data,
        onderwerp='Ziekteverzuim',
        voorspellen_tot_datum='2025-01-01',
        vensterlengte=None,
        verschuiving=None,
        jaarlijks_patroon=False,
        wekelijks_patroon=False,
        graad=1,
        model=None,
        zie_traintest_periodes=False):

    _data = data.copy()
    vanaf_datum_train_periode = _data.index.min()
    maximum_data_dataset = datetime.datetime(2024,12,31)
    date_yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    maximum_date = min(date_yesterday, maximum_data_dataset)
    tot_datum_train_periode = maximum_date
    vanaf_datum_test_periode = maximum_date
    tot_datum_test_periode = voorspellen_tot_datum

    df = pas_regressie_toe(
            data=_data,
            onderwerp=onderwerp,
            jaarlijks_seizoenspatroon=jaarlijks_patroon,
            wekelijks_seizoenspatroon=wekelijks_patroon,
            graad=graad,
            vanaf_datum_train_periode = vanaf_datum_train_periode,
            tot_datum_train_periode = tot_datum_train_periode,
            vanaf_datum_test_periode = vanaf_datum_test_periode,
            tot_datum_test_periode = tot_datum_test_periode,
            zie_traintest_periodes=zie_traintest_periodes
        )

    return df                

# def pas_modellen_toe(data, 
#                      onderwerp, 
#                      vanaf_datum_train_periode = None,
#                      tot_datum_train_periode = None,
#                      vanaf_datum_test_periode = None,
#                      tot_datum_test_periode = None,
#                      window_size = 7,
#                      shift_period = 365,
#                      yearly_seasonality=False,
#                      weekly_seasonality=False,
#                      transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3,
#                      predict=False):
    
#     vanaf_datum_train_periode, tot_datum_train_periode, \
#         vanaf_datum_test_periode, tot_datum_test_periode = calibrate_dates(vanaf_datum_train_periode,
#                                                                            tot_datum_train_periode, 
#                                                                            vanaf_datum_test_periode, 
#                                                                            tot_datum_test_periode)

#     df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(data, 
#                                                             onderwerp, 
#                                                             vanaf_datum_train_periode, 
#                                                             tot_datum_train_periode, 
#                                                             vanaf_datum_test_periode, 
#                                                             tot_datum_test_periode)

#     # Apply average model
#     # Calculate the average of the train data
#     average = calc_average(df_y_train)
#     # Predict the average for the train data
#     y_preds_train_avg = predict_with_average(df_y_train, average)
#     # Predict the average for the test data
#     y_preds_test_avg = predict_with_average(df_y_test, average)
#     y_avg = pd.concat([y_preds_train_avg, y_preds_test_avg])
    
#     # Apply moving average model
#     # Calculate the moving average of the training data
#     df_y_train_test = pd.concat([df_y_train, df_y_test])
#     y_mov_avg = calc_moving_average(df_y_train_test, 
#                                                 window_size=window_size, 
#                                                 shift_period=shift_period, 
#                                                 predict_to_date=tot_datum_test_periode,
#                                                 predict=predict)

#     # Apply Linear Regression model
#     # Get Linear Regression model
#     model = LinearRegressionTrain(df_X_train, 
#                                   df_y_train, 
#                                   yearly_seasonality=yearly_seasonality, 
#                                   weekly_seasonality=weekly_seasonality,
#                                   transformation=transformation, n_bins=n_bins, strategy=strategy, n_knots=n_knots, degree=degree)
#     # Make predictions with the Linear Regression model for the train data
#     y_preds_train_lin_reg = LinearRegressionPredict(df_X_train, 
#                                                     model, 
#                                                     yearly_seasonality=yearly_seasonality, 
#                                                     weekly_seasonality=weekly_seasonality)
#     # Make predictions with the Linear Regression model for the test data
#     y_preds_test_lin_reg = LinearRegressionPredict(df_X_test, 
#                                                    model, 
#                                                    yearly_seasonality=yearly_seasonality, 
#                                                    weekly_seasonality=weekly_seasonality)
#     y_lin_reg = pd.concat([y_preds_train_lin_reg, y_preds_test_lin_reg])
    
#     df_real = pd.concat([df_y_train, df_y_test], axis=0)
#     df_real.name = onderwerp

#     df_y_all = pd.DataFrame()
#     df_y_all['Gemiddelde'] = y_avg
#     df_y_all['Voortschrijdend gemiddelde'] = y_mov_avg
#     df_y_all['Lineaire regressie'] = y_lin_reg

    
#     df_total = pd.concat([df_real, df_y_all], axis=1)

#     plot_prediction_with_shapes(df=df_total, 
#                                 start_train=vanaf_datum_train_periode, 
#                                 end_train=tot_datum_train_periode, 
#                                 start_test=vanaf_datum_test_periode, 
#                                 end_test=tot_datum_test_periode, 
#                                 shapes=True)

#     return df_total

def onderzoek_afwijkingen(list_of_dfs, onderwerp='Ziekteverzuim', start=None, end=None, show='errors', yaxis_title="Afwijking"):
    df = combine_dfs_of_models(list_of_dfs)
    _, _, start, end = calibrate_dates(None,None, start, end)
    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()
    if show == 'errors':
        plot_errors(df=df, base_col=onderwerp, start=start, end=end, yaxis_title=yaxis_title)
    elif show == 'distribution':
        plot_distribution(df=df, base_col=onderwerp, start=start, end=end)
    elif show == 'both':
        plot_errors(df=df, base_col=onderwerp, start=start, end=end, yaxis_title=yaxis_title)
        plot_distribution(df=df, base_col=onderwerp, start=start, end=end)
    else:
        raise ValueError("Kies een van de volgende opties: 'errors', 'distribution' of 'both'.")


def bereken_metrieken(list_of_dfs, onderwerp='Ziekteverzuim', start=None, end=None, list_metrics=[accuracy, calc_max_error, calc_MAE], print_statement=True):
    """_summary_

    Parameters
    ----------
    list_of_dfs : _type_
        _description_
    onderwerp : _type_
        _description_
    start : _type_
        _description_
    end : _type_
        _description_
    list_metrics : _type_
        _description_[accuracy, calc_r2_score, calc_sum_of_errors, calc_average_error, calc_max_error, calc_MAE, calc_MAPE, calc_RMSE]

    Returns
    -------
    _type_
        _description_
    """
    df = combine_dfs_of_models(list_of_dfs)
    df.dropna(inplace=True)
    _, _, start, end = calibrate_dates(None,None, start, end)
    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()
    _df = df.loc[start:end].copy()
    if print_statement is True:
        print(f"De periode die wordt geanalyseerd is van {start} tot {end}.")
    df_metrics = pd.DataFrame()
    metric_cols = [col for col in _df.columns if onderwerp not in col]

    for col in metric_cols:
        for metric in list_metrics:
            df_metrics.loc[col, metric.__name__] = metric(_df[onderwerp], _df[col])
    df_metrics.rename(columns={
        'accuracy': 'Juistheid',
        'calc_r2_score': 'R2-score',
        'calc_sum_of_errors': 'Totale afwijking', 
        'calc_average_error': 'Gemiddelde afwijking', 
        'calc_max_error': 'Maximale afwijking',
        'calc_MAE': 'Gemiddelde absolute afwijking', 
        'calc_MAPE': 'Mean Absolute Percentage Error', 
        'calc_RMSE': 'Root Mean Squared Error'}, 
        inplace=True)
    return df_metrics

def opties_berekenen(data, 
                     onderwerp='Ziekteverzuim',
                     vanaf_datum_train_periode = None,
                     tot_datum_train_periode = None,
                     vanaf_datum_test_periode = None,
                     tot_datum_test_periode = None,
                     zie_traintest_periodes=False):

    _data = data.copy()
    
    vanaf_datum_train_periode, tot_datum_train_periode, \
        vanaf_datum_test_periode, tot_datum_test_periode = calibrate_dates(vanaf_datum_train_periode,
                                                                           tot_datum_train_periode, 
                                                                           vanaf_datum_test_periode, 
                                                                           tot_datum_test_periode)


    df_optie1 = pas_voortschrijdend_gemiddelde_toe(
        data=_data,
        onderwerp=onderwerp,
        vensterlengte=1,
        verschuiving=365,
        zie_traintest_periodes=zie_traintest_periodes,
        plot=False
    )
    df_optie1['Optie 1'] = df_optie1['Voortschrijdend gemiddelde']
    
    df_optie2 = pas_voortschrijdend_gemiddelde_toe(
        data=_data,
        onderwerp=onderwerp,
        vensterlengte=7,
        verschuiving=365,
        zie_traintest_periodes=zie_traintest_periodes,
        plot=False
    )
    df_optie2['Optie 2'] = df_optie2['Voortschrijdend gemiddelde']
    
    df_optie3 = pas_regressie_toe(
        data=_data,
        onderwerp=onderwerp,
        jaarlijks_seizoenspatroon=True,
        wekelijks_seizoenspatroon=False,
        graad=1,
        n_knots=3,
        zie_traintest_periodes=zie_traintest_periodes,
        plot=False
    )
    df_optie3['Optie 3'] = df_optie3['Regressie']
    
    df_optie4 = pas_regressie_toe(
        data=_data,
        onderwerp=onderwerp,
        jaarlijks_seizoenspatroon=True,
        wekelijks_seizoenspatroon=True,
        graad=2,
        n_knots=3,
        zie_traintest_periodes=zie_traintest_periodes,
        plot=False
    )
    df_optie4['Optie 4'] = df_optie4['Regressie']
    
    ## Visualiseren
    df_all = pd.concat([df_optie1[['Ziekteverzuim','Optie 1']], df_optie2['Optie 2'], df_optie3['Optie 3'], df_optie4['Optie 4']], axis=1)

    # df_plot = df_all.iloc[vanaf_datum_test_periode:zie_traintest_periodes]
    df_plot = df_all.loc[(df_all.index >= vanaf_datum_test_periode) & (df_all.index <= tot_datum_test_periode)]

    plot_prediction_with_shapes(df=df_plot)
    # plot_prediction_with_shapes(df=df_all, 
    #                             start_train=vanaf_datum_train_periode, 
    #                             end_train=tot_datum_train_periode, 
    #                             start_test=vanaf_datum_test_periode, 
    #                             end_test=tot_datum_test_periode, 
    #                             shapes=zie_traintest_periodes)

    
    # Berekenen metrieken
    df_metrics = bereken_metrieken(list_of_dfs=[df_all], 
                    onderwerp=onderwerp)

    return df_metrics

def optie_1(data, onderwerp='Ziekteverzuim'):
    ## Instellingen model voortschrijdend gemiddelde
    vensterlengte = 1
    verschuiving = 7

    ## Toepassen instellingen in model
    df_voortschrijdend_gemiddelde = pas_voortschrijdend_gemiddelde_toe(
        data=data,
        onderwerp=onderwerp,
        vensterlengte=vensterlengte,
        verschuiving=verschuiving,
        zie_traintest_periodes=True
    )

    ## Instellingen regressiemodel
    jaarlijks_patroon=True
    wekelijks_patroon=False
    graad=1

    ## Toepassen instellingen in model
    df_regressie = pas_regressie_toe(data=data,
        onderwerp=onderwerp,
        jaarlijks_seizoenspatroon=jaarlijks_patroon,
        wekelijks_seizoenspatroon=wekelijks_patroon,
        graad=graad,
        n_knots=3,
        zie_traintest_periodes=True
    )

    ## Berekenen metrieken
    df_metrics = bereken_metrieken(list_of_dfs=[df_voortschrijdend_gemiddelde, df_regressie], 
                    onderwerp=onderwerp)

    return df_metrics

def optie_2(data, onderwerp='Ziekteverzuim'):
    ## Instellingen model voortschrijdend gemiddelde
    vensterlengte = 1
    verschuiving = 365

    ## Toepassen instellingen in model
    df_voortschrijdend_gemiddelde = pas_voortschrijdend_gemiddelde_toe(
        data=data,
        onderwerp=onderwerp,
        vensterlengte=vensterlengte,
        verschuiving=verschuiving,
        zie_traintest_periodes=True
    )

    ## Instellingen regressiemodel
    jaarlijks_patroon=False
    wekelijks_patroon=True
    graad=3

    ## Toepassen instellingen in model
    df_regressie = pas_regressie_toe(data=data,
        onderwerp=onderwerp,
        jaarlijks_seizoenspatroon=jaarlijks_patroon,
        wekelijks_seizoenspatroon=wekelijks_patroon,
        graad=graad,
        n_knots=3,
        zie_traintest_periodes=True
    )

    ## Berekenen metrieken
    df_metrics = bereken_metrieken(list_of_dfs=[df_voortschrijdend_gemiddelde, df_regressie], 
                    onderwerp=onderwerp)

    return df_metrics

def optie_3(data, onderwerp='Ziekteverzuim'):
    ## Instellingen model voortschrijdend gemiddelde
    vensterlengte = 7
    verschuiving = 28

    ## Toepassen instellingen in model
    df_voortschrijdend_gemiddelde = pas_voortschrijdend_gemiddelde_toe(
        data=data,
        onderwerp=onderwerp,
        vensterlengte=vensterlengte,
        verschuiving=verschuiving,
        zie_traintest_periodes=True
    )

    ## Instellingen regressiemodel
    jaarlijks_patroon=True
    wekelijks_patroon=True
    graad=12

    ## Toepassen instellingen in model
    df_regressie = pas_regressie_toe(data=data,
        onderwerp=onderwerp,
        jaarlijks_seizoenspatroon=jaarlijks_patroon,
        wekelijks_seizoenspatroon=wekelijks_patroon,
        graad=graad,
        n_knots=3,
        zie_traintest_periodes=True
    )

    ## Berekenen metrieken
    df_metrics = bereken_metrieken(list_of_dfs=[df_voortschrijdend_gemiddelde, df_regressie], 
                    onderwerp=onderwerp)

    return df_metrics

# def pas_parameters_toe_en_evalueer(df, 
#                      onderwerp, 
#                      vanaf_datum_train_periode = '2019-01-01',
#                      tot_datum_train_periode = '2023-05-15',
#                      vanaf_datum_test_periode = '2023-05-15',
#                      tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
#                      window_size = 7,
#                      shift_period = 365,
#                      yearly_seasonality=False,
#                      weekly_seasonality=False,
#                      transformation='spline', n_bins=4, strategy='uniform', n_knots=2, degree=3,
#                      predict=False,
#                      shapes=True):
    
#     df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
#                                                             onderwerp, 
#                                                             vanaf_datum_train_periode, 
#                                                             tot_datum_train_periode, 
#                                                             vanaf_datum_test_periode, 
#                                                             tot_datum_test_periode)

#     # Apply average model
#     # Calculate the average of the train data
#     average = calc_average(df_y_train)
#     # Predict the average for the train data
#     y_preds_train_avg = predict_with_average(df_y_train, average)
#     # Predict the average for the test data
#     y_preds_test_avg = predict_with_average(df_y_test, average)
#     y_avg = pd.concat([y_preds_train_avg, y_preds_test_avg])
    
#     # Apply moving average model
#     # Calculate the moving average of the training data
#     df_y_train_test = pd.concat([df_y_train, df_y_test])
#     y_mov_avg = calc_moving_average(df_y_train_test, 
#                                                 window_size=window_size, 
#                                                 shift_period=shift_period, 
#                                                 predict_to_date=tot_datum_test_periode,
#                                                 predict=predict)

#     # Apply Linear Regression model
#     # Get Linear Regression model
#     model = LinearRegressionTrain(df_X_train, 
#                                   df_y_train, 
#                                   yearly_seasonality=yearly_seasonality, 
#                                   weekly_seasonality=weekly_seasonality,
#                                   transformation=transformation, n_bins=n_bins, strategy=strategy, n_knots=n_knots, degree=degree)
#     # Make predictions with the Linear Regression model for the train data
#     y_preds_train_lin_reg = LinearRegressionPredict(df_X_train, 
#                                                     model, 
#                                                     yearly_seasonality=yearly_seasonality, 
#                                                     weekly_seasonality=weekly_seasonality)
#     # Make predictions with the Linear Regression model for the test data
#     y_preds_test_lin_reg = LinearRegressionPredict(df_X_test, 
#                                                    model, 
#                                                    yearly_seasonality=yearly_seasonality, 
#                                                    weekly_seasonality=weekly_seasonality)
#     y_lin_reg = pd.concat([y_preds_train_lin_reg, y_preds_test_lin_reg])
    
#     df_real = pd.concat([df_y_train, df_y_test], axis=0)
#     df_real.name = onderwerp

#     df_y_all = pd.DataFrame()
#     df_y_all['Gemiddelde'] = y_avg
#     df_y_all['Voortschrijdend gemiddelde'] = y_mov_avg
#     df_y_all['Lineaire regressie'] = y_lin_reg

    
#     df_total = pd.concat([df_real, df_y_all], axis=1)

#     plot_prediction_with_shapes(df=df_total, 
#                                 start_train=vanaf_datum_train_periode, 
#                                 end_train=tot_datum_train_periode, 
#                                 start_test=vanaf_datum_test_periode, 
#                                 end_test=tot_datum_test_periode, 
#                                 shapes=shapes)

#     return df_total

# def voorspel(df, 
#                 onderwerp, 
#                 vanaf_datum_train_periode = '2019-01-01',
#                 tot_datum_train_periode = '2023-05-15',
#                 vanaf_datum_test_periode = '2023-05-15',
#                 tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
#                 window_size = 7,
#                 shift_period = 365,
#                 yearly_seasonality=False,
#                 weekly_seasonality=False,
#                 transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3,
#                 predict=False,
#                 shapes=False):

#     df_voorspel = pas_parameters_toe_en_evalueer(df=df, 
#                 onderwerp=onderwerp, 
#                 vanaf_datum_train_periode = vanaf_datum_train_periode,
#                 tot_datum_train_periode = tot_datum_train_periode,
#                 vanaf_datum_test_periode = vanaf_datum_test_periode,
#                 tot_datum_test_periode = tot_datum_test_periode,
#                 window_size = window_size,
#                 shift_period = shift_period,
#                 yearly_seasonality=yearly_seasonality,
#                 weekly_seasonality=weekly_seasonality,
#                 transformation=transformation, n_bins=n_bins, strategy=strategy, n_knots=n_knots, degree=degree,
#                 predict=predict,
#                 shapes=shapes)
#     return df_voorspel