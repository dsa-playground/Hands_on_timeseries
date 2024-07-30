import datetime as datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scripts.evaluate.evaluate import plot_prediction_with_shapes

def create_train_test_future_split(df, col, date_untill='2024-05-15'):
    df_Xy = df.loc[df.index < date_untill].copy()
    date_cutoff = df_Xy.index.max()-datetime.timedelta(days=365)
    df_Xy['y'] = df_Xy[col]
    df_Xy.drop(columns=[col], inplace=True)
    X_train = df_Xy.loc[df_Xy.index < date_cutoff].copy()
    X_test = df_Xy.loc[df_Xy.index >= date_cutoff].copy()
    y_train = X_train['y']
    y_test = X_test['y']
    X_train.drop(columns=['y'], inplace=True)
    X_test.drop(columns=['y'], inplace=True)
    return X_train, X_test, y_train, y_test

def AverageTrain(X_train, y_train, window_size=None, shift_period=None):
    if len(X_train) != len(y_train):
        raise ValueError('X_train and y_train must have the same length')
    if shift_period is None:
        shift_period = 0
    if window_size is None:
        average = y_train.mean()
    else:
        average = y_train.shift(shift_period)[-window_size:].mean()
    return average

def AveragePredict(X_test, average):
    y_preds = np.full(len(X_test), average)
    return y_preds

def MovingAverageTrain(y_train, window_size, shift_period):
    if window_size is None:
        raise ValueError('window_size must be defined')
    if shift_period is None:
        raise ValueError('shift_period must be defined')
    else:
        shift_period = shift_period+1
    moving_average = y_train.shift(shift_period).rolling(window=window_size).mean()
    return moving_average

def MovingAveragePredict(y_history, window_size, shift_period, predict_to_date):
    y_hist = y_history.copy()
    if window_size is None:
        raise ValueError('window_size must be defined')
    if shift_period is None:
        raise ValueError('shift_period must be defined')
    else:
        shift_period = shift_period+1
    try:
        # Try to convert the end date to a datetime object
        datetime.datetime.strptime(predict_to_date, '%Y-%m-%d')
    except ValueError:
        print(f"Error: The predict_to_date {predict_to_date} is not in the correct format. Please use 'YYYY-MM-DD'.")
    start_date = y_hist.index.max() + datetime.timedelta(days=1)
    end_date = pd.to_datetime(predict_to_date)
    range_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_extend = pd.DataFrame(index=range_dates)
    df_extend = pd.Series(index=range_dates)
    y_predict = pd.concat([y_hist, df_extend])
    # moving_average = pd.Series(index=y_predict.index)
    while y_predict.isnull().sum() > 0:
        # print(f"Number of nulls before: {y_predict.isnull().sum()}")
        moving_average = y_predict.shift(shift_period).rolling(window=window_size).mean()
        y_predict = pd.concat([y_predict[y_predict.index < y_predict[y_predict.isna()].index.min()], moving_average.loc[moving_average.index >= y_predict[y_predict.isna()].index.min()]], axis=0)    
        # print(f"Number of nulls after: {y_predict.isnull().sum()}")
    y_predict = y_predict.loc[y_predict.index > start_date]
    return y_predict

def LinearRegressionTrain(X_train, y_train):
    # Convert X_train to ordinal values
    X_train_num = pd.to_datetime(X_train).map(datetime.datetime.toordinal)
    
    # Convert X_train_num to a DataFrame
    X_train_df = X_train_num.to_frame()
    
    # Create a LinearRegression object
    model = LinearRegression()
    
    # Fit the model to the training data
    model.fit(X_train_df, y_train)
    
    return model

def LinearRegressionPredict(X_test, model):

    # Convert X_train and X_test to ordinal values
    X_test_num = pd.to_datetime(X_test).map(datetime.datetime.toordinal)

    # Convert X_train_num and X_test_num to DataFrames
    X_test_df = X_test_num.to_frame()

    # Make predictions on the test data
    predictions = model.predict(X_test_df)
    y_preds = pd.Series(predictions, index=X_test)
    return y_preds

def make_predictions(X_train, X_test, y_train, model, window_size=7, shift_period=365, predict_to_date='2024-05-15', hist_and_pred=False):
    if model == 'Gemiddelde':
        average = AverageTrain(X_train, y_train, window_size, shift_period)
        y_hist = AveragePredict(X_train, average)
        y_preds = AveragePredict(X_test, average)
    elif model == 'Voortschrijdend gemiddelde':
        y_hist = MovingAverageTrain(y_train=y_train, window_size=window_size, shift_period=shift_period)
        y_preds = MovingAveragePredict(y_history=y_train, window_size=window_size, shift_period=shift_period, predict_to_date=predict_to_date)
    elif model == 'Lineaire regressie':
        LR_model = LinearRegressionTrain(X_train=X_train.index, y_train=y_train)
        y_hist = LinearRegressionPredict(X_test=X_train.index, model=LR_model)
        y_preds = LinearRegressionPredict(X_test=X_test.index, model=LR_model)
    if hist_and_pred:
        y_return = pd.concat([pd.Series(y_hist), pd.Series(y_preds)])
        y_return.index = X_train.index.append(X_test.index)
    else:
        y_return = pd.Series(y_preds)
        y_return.index = X_test.index
    return y_return


def pas_modellen_toe(df, onderwerp):
    X_train, X_test, y_train, y_test = create_train_test_future_split(df=df, col=onderwerp)

    df_preds = pd.DataFrame()
    for model in ['Gemiddelde', 'Voortschrijdend gemiddelde', 'Lineaire regressie']:
        # print(f"Name of model: {model}")
        y_return = make_predictions(X_train=X_train, X_test=X_test, y_train=y_train, model=model, window_size=7, shift_period=365, hist_and_pred=True)
        # print(f"Shape of y_preds: {y_return.shape}")
        df_preds[model] = y_return
    # df_preds.index = X_train.index.append(X_test.index)

    df_real = pd.concat([y_train, y_test], axis=0)
    df_real.name = onderwerp
    df_total = pd.concat([df_real, df_preds], axis=1)

    plot_prediction_with_shapes(df=df_total)

    return df_total















# def ts_train_test_split(df_Xy, cutoff_date='2024-01-01'):
#     cutoff_date = pd.to_datetime(cutoff_date)
    
#     df_train = df_Xy[df_Xy['datum'] < cutoff_date]
#     df_test = df_Xy[df_Xy['datum'] >= cutoff_date]

#     X_train = df_train['datum']
#     y_train = df_train['aantal_clienten']
#     X_test = df_test['datum']
#     y_test = df_test['aantal_clienten']
#     return X_train, X_test, y_train, y_test

# def AverageTrain(X_train, y_train, window_size=None):
#     if len(X_train) != len(y_train):
#         raise ValueError('X_train and y_train must have the same length')
#     if window_size is None:
#         average = y_train.mean()
#     else:
#         average = y_train[-window_size:].mean()
#     return average

# def AveragePredict(X_test, average):
#     y_preds = np.full(len(X_test), average)
#     return y_preds

# # def MovingAveringPredict(X_train, y_train, X_test, window_size=7):
    
# #     df = pd.DataFrame({'datum': pd.concat([X_train, X_test]),
# #                   'aantal_clienten': np.concatenate([y_train, [np.nan]*len(X_test)])})
# #     for i in range(len(df) - len(X_test), len(df)):
# #             df.loc[i, 'aantal_clienten'] = df.loc[i-window_size:i-1, 'aantal_clienten'].mean()
# #     y_preds = df['aantal_clienten'].values[-len(X_test):]
# #     return y_preds


# def MonthlyMovingAveringTrain(X_train, y_train):
#     # Create a DataFrame
#     df_train = pd.DataFrame({'datum': X_train,
#                              'aantal_clienten': y_train})
    
#     # Set 'datum' as the index
#     df_train.set_index('datum', inplace=True)
    
#     # Calculate the mean for each month
#     monthly_mean = df_train.groupby(df_train.index.month).mean()
    
#     return monthly_mean

# def MonthlyMovingAveringPredict(X_test, monthly_mean):    
#     # Create a DataFrame for X_test
#     df_test = pd.DataFrame({'datum': X_test})
#     df_test.set_index('datum', inplace=True)
    
#     # Use the monthly mean of X_train and y_train to predict the values for the dates in X_test
#     df_test['aantal_clienten'] = df_test.index.month.map(monthly_mean['aantal_clienten'])
    
#     # Get the predictions
#     y_preds = df_test['aantal_clienten'].values
    
#     return y_preds

# def LinearRegressionTrain(X_train, y_train):
#     # Convert X_train to ordinal values
#     X_train_num = pd.to_datetime(X_train).map(dt.datetime.toordinal)
    
#     # Convert X_train_num to a DataFrame
#     X_train_df = X_train_num.to_frame()
    
#     # Create a LinearRegression object
#     model = LinearRegression()
    
#     # Fit the model to the training data
#     model.fit(X_train_df, y_train)
    
#     return model

# def LinearRegressionPredict(X_test, model):

#     # Convert X_train and X_test to ordinal values
#     X_test_num = pd.to_datetime(X_test).map(dt.datetime.toordinal)

#     # Convert X_train_num and X_test_num to DataFrames
#     X_test_df = X_test_num.to_frame()

#     # Make predictions on the test data
#     predictions = model.predict(X_test_df)
#     return predictions


