import datetime as datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, SplineTransformer
from sklearn.linear_model import LinearRegression
from scripts.evaluate.evaluate import plot_prediction_with_shapes
from sklearn.preprocessing import PolynomialFeatures
from scripts.preprocess.preprocess import create_subset_df, create_df_X_and_y, get_second_nan_index, apply_sin_cos_transformation, LinearRegressionTransformation, make_dates_datetime, make_X_y


# def create_subset_df(df, start, end):
#     df_subset = df.loc[(df.index >= start) & (df.index < end)].copy()
#     return df_subset

# def create_df_X_and_y(df, base_col):
#     df_Xy = df.copy()
#     df_y = df_Xy[base_col]
#     df_X = pd.DataFrame(index=df_Xy.index)
#     return df_X, df_y

def calc_average(data, base_col=None):
    if isinstance(data, pd.Series):
        average = data.mean()
    elif isinstance(data, pd.DataFrame):
        if base_col is None:
            raise ValueError('base_col must be specified when data is a DataFrame')
        average = data[base_col].mean()
    else:
        raise ValueError('data must be a Series or DataFrame')
    return average

def predict_with_average(data, average):
    if isinstance(data, pd.Series):
        y_preds = pd.Series([average]*len(data.index), index=data.index)
        y_preds.name = 'Gemiddelde'
    elif isinstance(data, pd.DataFrame):
        y_preds = pd.DataFrame([average]*len(data.index), columns=['Gemiddelde'], index=data.index)
    else:
        raise ValueError('data must be a Series or DataFrame')
    return y_preds

# def get_second_nan_index(s):
#     first_valid = s.index.get_loc(s.first_valid_index())
#     last_valid = s.index.get_loc(s.last_valid_index())
#     second_nan_index = s.iloc[last_valid:].isna().argmin() + last_valid
#     return second_nan_index

def calc_moving_average(data, window_size, shift_period, predict_to_date, base_col=None, predict=True):
    """Calculate the moving average of a time series.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The time series data to calculate the moving average for.
    window_size : int
        The number of periods to include in the moving average calculation.
    shift_period : int
        The number of periods to shift the moving average forward.
    predict_to_date : str
        The date to predict the moving average to, format 'YYYY-MM-DD'.
    base_col : str, optional
        Name of the column to calculate the moving average for, by default None.

    Returns
    -------
    pd.Series
        The moving average of the time series.

    Raises
    ------
    ValueError
        If window_size is not defined.
    ValueError
        If shift_period is not defined.
    ValueError
        If base_col is not specified when data is a DataFrame.
    """
    _data = data.copy()
    if window_size is None:
        raise ValueError('window_size must be defined')
    if shift_period is None:
        raise ValueError('shift_period must be defined')
    else:
        shift_period = shift_period+1
    if isinstance(_data, pd.Series):
        y = _data
    elif isinstance(_data, pd.DataFrame):
        if base_col is None:
            raise ValueError('base_col must be specified when data is a DataFrame')
        y = _data[base_col]
    if predict_to_date is None:
        y_predict = y.shift(shift_period).rolling(window=window_size).mean()
    else:
        y_predict = y
        if isinstance(predict_to_date, str):
            try:
                # Try to convert the end date to a datetime object
                predict_to_date = datetime.datetime.strptime(predict_to_date, '%Y-%m-%d')
            except ValueError:
                print(f"Error: The predict_to_date {predict_to_date} is not in the correct format. Please use 'YYYY-MM-DD'.")         
        max_iterations = y_predict.isnull().sum()
        if predict:
            for _ in range(max_iterations):
                if not y_predict.isnull().any():
                    break
                # print(f"Number of nulls before: {y_predict.isnull().sum()}")
                moving_average = y_predict.shift(shift_period).rolling(window=window_size).mean()
                y_predict = pd.concat([y_predict[y_predict.index < y_predict[y_predict.isna()].index.min()], 
                                        moving_average.loc[moving_average.index >= y_predict[y_predict.isna()].index.min()]], axis=0)
                # print(f"Number of nulls after: {y_predict.isnull().sum()}")
        else:
            moving_average_train = y_predict.shift(shift_period).rolling(window=window_size).mean()
            y_predict = moving_average_train
            for _ in range(max_iterations):
                if not y_predict.isnull().any():
                    break
                # print(f"Number of nulls before: {y_predict.isnull().sum()}")
                moving_average = y_predict.shift(shift_period).rolling(window=window_size).mean()
                y_predict = pd.concat([y_predict[y_predict.index <= y_predict.index[get_second_nan_index(y_predict)]], 
                                        moving_average.loc[moving_average.index > y_predict.index[get_second_nan_index(y_predict)]]], axis=0)
                # print(f"Number of nulls after: {y_predict.isnull().sum()}")

        return y_predict

# def apply_sin_cos_transformation(df, date_col, period='year'):
#     if period == 'year':
#         # Convert the date to the day of the year
#         df[f'num_col_{period}'] = df[date_col].dt.dayofyear
#         cycle_length = 365.25
#     elif period == 'week':
#         # Convert the date to the day of the week (0 = Monday, 6 = Sunday)
#         df[f'num_col_{period}'] = df[date_col].dt.dayofweek
#         cycle_length = 7
#     else:
#         raise ValueError(f"Period {period} is not supported. Please choose 'year' or 'week'.")
    
#     # Apply the sine transformation
#     df[f'sin_transformation_{period}'] = np.sin(2 * np.pi * df[f'num_col_{period}'] / cycle_length)

#     # Apply the cosine transformation
#     df[f'cos_transformation_{period}'] = np.cos(2 * np.pi * df[f'num_col_{period}'] / cycle_length)

#     return df

# def LinearRegressionTransformation(X_data, yearly_seasonality=False, weekly_seasonality=False):
#     # Reset the index of X_train
#     X_train_num = X_data.copy().reset_index()

#     # Convert X_train to ordinal values
#     X_train_num['date_transformation'] = pd.to_datetime(X_train_num['Datum']).map(datetime.datetime.toordinal)
    
#     # Apply the sine and cosine transformations
#     if yearly_seasonality:
#         X_train_num = apply_sin_cos_transformation(df=X_train_num, date_col='Datum', period='year')
#     if weekly_seasonality:
#         X_train_num = apply_sin_cos_transformation(df=X_train_num, date_col='Datum', period='week')
#     # Convert X_train_num to a DataFrame
#     # X_train_df = X_train_num.to_frame()

#     # Select the columns with the transformations
#     X_train_num = X_train_num[[col for col in X_train_num.columns if 'transformation' in col]]
#     return X_train_num

def LinearRegressionTrain(X_train, y_train, yearly_seasonality=False, weekly_seasonality=False, transformation='lineair', n_bins=4, strategy='uniform', n_knots=10, degree=3):
    # Transform the training data
    X_train_num = LinearRegressionTransformation(X_train, yearly_seasonality, weekly_seasonality)
    
    if transformation == 'lineair':
        model = LinearRegression()
    elif transformation == 'binnes':
        # X_train_num['Datum'] = X_train_num['date_transformation'].map(datetime.datetime.fromordinal)
        # X_train_num['Datum'] = pd.to_datetime(X_train_num['Datum'])
        # num_unique_year_month = X_train_num['Datum'].dt.to_period('M').nunique()
        # # print(f"Number of unique year-month combinations: {num_unique_year_month}")
        # X_train_num.drop(columns=['Datum'], inplace=True)
        # Create a LinearRegression object
        binnen_regression = make_pipeline(
            KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy),
            LinearRegression()
        )
        model = binnen_regression
    elif transformation == 'spline':
        spline_regression = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False),
            LinearRegression()
        )
        model = spline_regression
    elif transformation == 'polynomial':
        polynomial_expansion = PolynomialFeatures(degree=degree, include_bias=False)
        polynomial_regression = make_pipeline(
            polynomial_expansion,
            LinearRegression()
        )
        model = polynomial_regression
    else:
        raise ValueError(f"Transformation {transformation} is not supported. Please choose 'lineair', 'binnes', 'spline' or 'polynomial'.")
    
    # print(X_train_num.head())
    # Fit the model to the training data
    model.fit(X_train_num, y_train)
    
    return model

def LinearRegressionPredict(X_test, model, yearly_seasonality=False, weekly_seasonality=False):
    # Transform the test data
    X_test_num = LinearRegressionTransformation(X_test, yearly_seasonality, weekly_seasonality)

    # Make predictions on the test data
    predictions = model.predict(X_test_num)
    y_preds = pd.Series(predictions, index=X_test.index)
    return y_preds

# def make_dates_datetime(date_vars):

#     # Convert all date variables from string to datetime.datetime
#     for i, date_var in enumerate(date_vars):
#         if isinstance(date_var, str):
#             date_vars[i] = datetime.datetime.strptime(date_var, '%Y-%m-%d')

#     for i, date_var in enumerate(date_vars):
#         if isinstance(date_var, pd.Timestamp):
#             date_vars[i] = date_var.to_pydatetime()

#     return date_vars


# def make_X_y(df, onderwerp, vanaf_datum_train_periode, tot_datum_train_periode, vanaf_datum_test_periode, tot_datum_test_periode):    
#     # date_vars = [vanaf_datum_train_periode, tot_datum_train_periode, 
#     #          vanaf_datum_test_periode, tot_datum_test_periode]

#     datetime_vars = make_dates_datetime(date_vars=[vanaf_datum_train_periode, tot_datum_train_periode, 
#              vanaf_datum_test_periode, tot_datum_test_periode])
#     vanaf_datum_train_periode, tot_datum_train_periode, \
#              vanaf_datum_test_periode, tot_datum_test_periode = datetime_vars
#     # # Convert all date variables from string to datetime.datetime
#     # for i, date_var in enumerate(date_vars):
#     #     if isinstance(date_var, str):
#     #         date_vars[i] = datetime.datetime.strptime(date_var, '%Y-%m-%d')

#     # for i, date_var in enumerate(date_vars):
#     #     if isinstance(date_var, pd.Timestamp):
#     #         date_vars[i] = date_var.to_pydatetime()

#     # # Unpack the converted date variables
#     # vanaf_datum_train_periode, tot_datum_train_periode, \
#     # vanaf_datum_test_periode, tot_datum_test_periode = date_vars

#     # return vanaf_datum_train_periode, tot_datum_train_periode, \
#     # vanaf_datum_test_periode, tot_datum_test_periode

#     if tot_datum_train_periode < vanaf_datum_train_periode:
#         raise ValueError("Let op: tot_datum_train_periode moet groter zijn dan vanaf_datum_train_periode. Kies andere waarden aub.")
#     if vanaf_datum_test_periode < tot_datum_train_periode:
#         raise ValueError("Let op: vanaf_datum_test_periode moet groter zijn dan tot_datum_train_periode. Kies andere waarden aub.")
#     if tot_datum_test_periode < vanaf_datum_test_periode:
#         raise ValueError("Let op: tot_datum_test_periode moet groter zijn dan vanaf_datum_test_periode. Kies andere waarden aub.")
#     if tot_datum_train_periode > df.index.max():
#         raise ValueError("Let op: tot_datum_train_periode moet kleiner zijn dan de maximale datum in de dataset. Kies andere waarden aub.")
    
#     minimum_date = datetime.datetime(2019,1,1)
#     maximum_date = datetime.datetime.now() - datetime.timedelta(days=1)
#     data = df.loc[(df.index >= minimum_date) & (df.index < maximum_date)].copy()
#     if tot_datum_test_periode > data.index.max():
#         range_dates_extend = pd.date_range(start=data.index.max() + datetime.timedelta(days=1), 
#                                                 end=pd.to_datetime(tot_datum_test_periode) - datetime.timedelta(days=1), 
#                                                 freq='D')
#         df_extend = pd.DataFrame({onderwerp: [np.nan]*len(range_dates_extend)}, index=range_dates_extend)
#         df_extended = pd.concat([data, df_extend])
#     else:
#         df_extended = data
#     df_extended.index.name = 'Datum'

#     # Create a subset of the data for the training period
#     df_train = create_subset_df(df=df_extended, start=vanaf_datum_train_periode, end=tot_datum_train_periode)

#     # Create a subset of the data for the test period
#     df_test = create_subset_df(df=df_extended, start=vanaf_datum_test_periode, end=tot_datum_test_periode)

#     # Create the X and y DataFrames for the training data
#     df_X_train, df_y_train = create_df_X_and_y(df_train, onderwerp)

#     # Create the X and y DataFrames for the test data
#     df_X_test, df_y_test = create_df_X_and_y(df_test, onderwerp)

#     return df_X_train, df_y_train, df_X_test, df_y_test

# def pas_gemiddelde_toe(df, 
#                      onderwerp, 
#                      vanaf_datum_train_periode = '2019-01-01',
#                      tot_datum_train_periode = '2023-05-15',
#                      vanaf_datum_test_periode = '2023-05-15',
#                      tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d')):
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

#     df_y_all = pd.DataFrame()
#     df_y_all['Gemiddelde'] = y_avg

#     df_real = pd.concat([df_y_train, df_y_test], axis=0)
#     df_real.name = onderwerp
#     df_total = pd.concat([df_real, df_y_all], axis=1)

#     plot_prediction_with_shapes(df=df_total, 
#                                 start_train=vanaf_datum_train_periode, 
#                                 end_train=tot_datum_train_periode, 
#                                 start_test=vanaf_datum_test_periode, 
#                                 end_test=tot_datum_test_periode, 
#                                 shapes=True)

#     return df_total

# def pas_voortschrijdend_gemiddelde_toe(df, 
#                      onderwerp, 
#                      vanaf_datum_train_periode = '2019-01-01',
#                      tot_datum_train_periode = '2023-05-15',
#                      vanaf_datum_test_periode = '2023-05-15',
#                      tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
#                      window_size = 7,
#                      shift_period = 365,
#                      predict=False):
    
#     df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
#                                                             onderwerp, 
#                                                             vanaf_datum_train_periode, 
#                                                             tot_datum_train_periode, 
#                                                             vanaf_datum_test_periode, 
#                                                             tot_datum_test_periode)

#     # Apply moving average model
#     # Calculate the moving average of the training data
#     df_y_train_test = pd.concat([df_y_train, df_y_test])
#     y_preds_train_mov_avg = calc_moving_average(df_y_train_test, 
#                                                 window_size=window_size, 
#                                                 shift_period=shift_period, 
#                                                 predict_to_date=tot_datum_test_periode,
#                                                 predict=predict)

#     df_y_all = pd.DataFrame()
#     df_y_all['Voortschrijdend gemiddelde'] = y_preds_train_mov_avg

#     df_real = pd.concat([df_y_train, df_y_test], axis=0)
#     df_real.name = onderwerp
#     df_total = pd.concat([df_real, df_y_all], axis=1)

#     plot_prediction_with_shapes(df=df_total, 
#                                 start_train=vanaf_datum_train_periode, 
#                                 end_train=tot_datum_train_periode, 
#                                 start_test=vanaf_datum_test_periode, 
#                                 end_test=tot_datum_test_periode, 
#                                 shapes=True)

#     return df_total

# def pas_lineaire_regressie_toe(df, 
#                      onderwerp, 
#                      vanaf_datum_train_periode = '2019-01-01',
#                      tot_datum_train_periode = '2023-05-15',
#                      vanaf_datum_test_periode = '2023-05-15',
#                      tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
#                      yearly_seasonality=False,
#                      weekly_seasonality=False,
#                      transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3):
    
#     df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
#                                                             onderwerp, 
#                                                             vanaf_datum_train_periode, 
#                                                             tot_datum_train_periode, 
#                                                             vanaf_datum_test_periode, 
#                                                             tot_datum_test_periode)

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
#     df_y_all['Lineaire regressie'] = y_lin_reg
#     df_y_all.index = df_real.index
#     df_total = pd.concat([df_real, df_y_all], axis=1)

#     plot_prediction_with_shapes(df=df_total, 
#                                 start_train=vanaf_datum_train_periode, 
#                                 end_train=tot_datum_train_periode, 
#                                 start_test=vanaf_datum_test_periode, 
#                                 end_test=tot_datum_test_periode, 
#                                 shapes=True)

#     return df_total


# def pas_modellen_toe(df, 
#                      onderwerp, 
#                      vanaf_datum_train_periode = '2019-01-01',
#                      tot_datum_train_periode = '2023-05-15',
#                      vanaf_datum_test_periode = '2023-05-15',
#                      tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
#                      window_size = 7,
#                      shift_period = 365,
#                      yearly_seasonality=False,
#                      weekly_seasonality=False,
#                      transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3,
#                      predict=False):
    
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
#                                 shapes=True)

#     return df_total



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
#                      transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3,
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
                                                 




# def calc_moving_average(data, window_size, shift_period, predict_to_date, base_col=None, predict=True):
#     """Calculate the moving average of a time series.

#     Parameters
#     ----------
#     data : pd.Series or pd.DataFrame
#         The time series data to calculate the moving average for.
#     window_size : int
#         The number of periods to include in the moving average calculation.
#     shift_period : int
#         The number of periods to shift the moving average forward.
#     predict_to_date : str
#         The date to predict the moving average to, format 'YYYY-MM-DD'.
#     base_col : str, optional
#         Name of the column to calculate the moving average for, by default None.

#     Returns
#     -------
#     pd.Series
#         The moving average of the time series.

#     Raises
#     ------
#     ValueError
#         If window_size is not defined.
#     ValueError
#         If shift_period is not defined.
#     ValueError
#         If base_col is not specified when data is a DataFrame.
#     """
#     _data = data.copy()
#     if window_size is None:
#         raise ValueError('window_size must be defined')
#     if shift_period is None:
#         raise ValueError('shift_period must be defined')
#     else:
#         shift_period = shift_period+1
#     if isinstance(_data, pd.Series):
#         y = _data
#     elif isinstance(_data, pd.DataFrame):
#         if base_col is None:
#             raise ValueError('base_col must be specified when data is a DataFrame')
#         y = _data[base_col]
#     if predict_to_date is None:
#         y_predict = y
#         y_predict = y_predict.shift(shift_period).rolling(window=window_size).mean()
#     else:
#         if isinstance(predict_to_date, str):
#             try:
#                 # Try to convert the end date to a datetime object
#                 predict_to_date = datetime.datetime.strptime(predict_to_date, '%Y-%m-%d')
#             except ValueError:
#                 print(f"Error: The predict_to_date {predict_to_date} is not in the correct format. Please use 'YYYY-MM-DD'.")
#         if predict_to_date <= _data.index.max():
#             y_predict = y.loc[y.index <= predict_to_date]
#             y_predict = y_predict.shift(shift_period).rolling(window=window_size).mean()
#         else:
#             range_dates_extend = pd.date_range(start=_data.index.max() + datetime.timedelta(days=1), 
#                                             end=pd.to_datetime(predict_to_date) - datetime.timedelta(days=1), 
#                                             freq='D')
#             df_extend = pd.Series(index=range_dates_extend)
#             y_predict = pd.concat([_data, df_extend])
#             # while y_predict.isnull().any(): #.sum() > 0:
#             #     print(f"Number of nulls before: {y_predict.isnull().sum()}")
#             #     moving_average = y_predict.shift(shift_period).rolling(window=window_size).mean()
#             #     y_predict = pd.concat([y_predict[y_predict.index < y_predict[y_predict.isna()].index.min()], moving_average.loc[moving_average.index >= y_predict[y_predict.isna()].index.min()]], axis=0)    
#             #     print(f"Number of nulls after: {y_predict.isnull().sum()}")
            
#             max_iterations = y_predict.isnull().sum()
#             if predict:
#                 for _ in range(max_iterations):
#                     if not y_predict.isnull().any():
#                         break
#                     # print(f"Number of nulls before: {y_predict.isnull().sum()}")
#                     moving_average = y_predict.shift(shift_period).rolling(window=window_size).mean()
#                     y_predict = pd.concat([y_predict[y_predict.index < y_predict[y_predict.isna()].index.min()], 
#                                             moving_average.loc[moving_average.index >= y_predict[y_predict.isna()].index.min()]], axis=0)
#                     # print(f"Number of nulls after: {y_predict.isnull().sum()}")
#             else:
#                 moving_average_train = y_predict.shift(shift_period).rolling(window=window_size).mean()
#                 y_predict = moving_average_train
#                 for _ in range(max_iterations):
#                     if not y_predict.isnull().any():
#                         break
#                     # print(f"Number of nulls before: {y_predict.isnull().sum()}")
#                     moving_average = y_predict.shift(shift_period).rolling(window=window_size).mean()
#                     y_predict = pd.concat([y_predict[y_predict.index <= y_predict.index[get_second_nan_index(y_predict)]], 
#                                             moving_average.loc[moving_average.index > y_predict.index[get_second_nan_index(y_predict)]]], axis=0)
#                     # print(f"Number of nulls after: {y_predict.isnull().sum()}")

#     return y_predict