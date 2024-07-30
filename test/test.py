# import datetime
# from scripts.preprocess.preprocess import make_X_y

# # Test all kinds of options given dates
# onderwerp = 'Ziekteverzuim'
# # make df

# # Standaard
# vanaf_datum_train_periode = '2019-01-01'
# tot_datum_train_periode = '2023-05-15'
# vanaf_datum_test_periode = '2023-05-16'
# tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d')
# df_extended, df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, onderwerp, vanaf_datum_train_periode, tot_datum_train_periode, vanaf_datum_test_periode, tot_datum_test_periode)
# # vanaf_datum_train_periode, tot_datum_train_periode, vanaf_datum_test_periode, tot_datum_test_periode = make_X_y(df, onderwerp, vanaf_datum_train_periode, tot_datum_train_periode, vanaf_datum_test_periode, tot_datum_test_periode)
# print("df_extended")
# display(df_extended.head())
# display(df_extended.tail())
# print("df_X_train")
# display(df_X_train.head())
# display(df_X_train.tail())
# print("df_y_train")
# display(df_y_train.head())
# display(df_y_train.tail())
# print("df_X_test")
# display(df_X_test.head())
# display(df_X_test.tail())
# print("df_y_test")
# display(df_y_test.head())
# display(df_y_test.tail())