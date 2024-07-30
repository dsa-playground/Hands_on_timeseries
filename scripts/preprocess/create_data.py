# Imports
import random
import numpy as np
from faker import Faker
import pandas as pd
import scipy.stats as stats


# Functions
def create_norm_data(lower_limit, upper_limit, mu, sigma, sample_size):
    """Generates a list of random samples from a truncated normal distribution.

    The function takes the lower and upper limits of the distribution, the mean (mu), 
    the standard deviation (sigma), and the sample size as input. It then generates 
    random samples from a truncated normal distribution defined by these parameters. 
    The generated samples are converted to integers before being returned.

    Parameters
    ----------
    lower_limit : int
        The lower limit of the truncated normal distribution.
    upper_limit : int
        The upper limit of the truncated normal distribution.
    mu : int
        The mean of the normal distribution.
    sigma : int
        The standard deviation of the normal distribution.
    sample_size : int
        The number of samples to generate.

    Returns
    -------
    list
        A list of integer samples from the specified truncated normal distribution.
    """
    # Define the lower and upper bounds of the truncated normal distribution
    a, b = (lower_limit - mu) / sigma, (upper_limit - mu) / sigma
    # Generate the random samples
    random_samples = stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=sample_size)
    # Convert to integers
    random_samples = [i.astype("int") for i in random_samples]
    return random_samples

def get_random_date_from_range(start_date, end_date):
    """Generates a random date within a specified range.

    Parameters
    ----------
    start_date : date | datetime | timedelta | str | int
        Date or delta to start the range with.
    end_date : date | datetime | timedelta | str | int
        Date or delta to end the range with.

    Returns
    -------
    date_range
        Range of dates between start_date and end_date.
    """
    fake = Faker('nl_NL')
    return fake.date_between(start_date=start_date, end_date=end_date)

def make_end_date(start_date):
    number_of_days = create_norm_data(lower_limit=5, upper_limit=20*(365/12), mu=9*(365/12), sigma=4*(365/12), sample_size=1)[0]
    future_date = start_date + pd.Timedelta(days=number_of_days)
    # If the future date is not in December, January, July, or August, generate a new date with a 50% chance
    while np.random.rand() < 0.4 and future_date.month not in [12, 1, 2]:
        random_days = create_norm_data(lower_limit=5, upper_limit=20*(365/12), mu=9*(365/12), sigma=4*(365/12), sample_size=1)[0]
        future_date = start_date + pd.Timedelta(days=random_days)
    return future_date

def make_date_range(start_date, number_of_days):
    return pd.date_range(start=start_date, periods=number_of_days, freq='D')

def make_date_range_between_dates(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date, freq='D')

def get_random_items_from_list(list, number_of_items=None):
    if number_of_items is None:
        number_of_items = random.choice(range(1, len(list)))
    returned_list = sorted(random.sample(list, number_of_items))
    return returned_list

def get_random_name():
    fake = Faker('nl_NL')
    voorzetsel = random.choice(['Meneer ', 'Mevrouw '])
    return voorzetsel + fake.last_name()

def make_df_one_client(start_date='-5y', end_date='-1y'):
    # Create a dataframe with one client
    # date_range= make_date_range(start_date = get_random_date_from_range(start_date='-5y', end_date='today'), 
    #                         number_of_days=create_norm_data(lower_limit=5, upper_limit=20*(365/12), mu=9*(365/12), sigma=4*(365/12), sample_size=1)[0])
    start_date = get_random_date_from_range(start_date=start_date, end_date=end_date)
    end_date = make_end_date(start_date=start_date)
    date_range= make_date_range_between_dates(start_date = start_date, end_date=end_date)
    lst_zorgzwaartes = get_random_items_from_list([5,6,7,8,9])
    lst_dates_zorgzwaartes = [date_range[0]] + get_random_items_from_list(list=list(date_range), number_of_items=len(lst_zorgzwaartes)-1)
    dict_zorgwaartes = dict(zip(lst_dates_zorgzwaartes, lst_zorgzwaartes))
    naam = get_random_name()

    # maken df
    df = pd.DataFrame(data=date_range, columns=['datum'])
    df['zorgzwaarte'] = df.datum.map(dict_zorgwaartes)
    df['zorgzwaarte'] = df['zorgzwaarte'].ffill().astype(int)
    df['client_naam'] = naam
    return df


def create_data_of_clients(n_clients, start_date='-5y', end_date='-1y'):
    dfs = []  # Initialize a list to store DataFrames
    for i in range(1, n_clients + 1):
        df = make_df_one_client(start_date=start_date, end_date=end_date)
        df['client_id'] = i
        df.set_index(['datum', 'client_id'], inplace=True)
        dfs.append(df)

    # Concatenate all DataFrames at once
    df_all = pd.concat(dfs).reset_index().sort_values(by=['datum', 'client_id'])
    df_all['datum_opgenomen'] = None
    return df_all

def get_date_range(df, col):
    min_date = df[col].min().date()
    max_date = df[col].max().date()
    return pd.date_range(min_date, max_date)

def create_clients_community(n_clients=1000, start_date='-5y', end_date='-1y'):
    return create_data_of_clients(n_clients=n_clients, start_date=start_date, end_date=end_date)    

def create_queue_df(df, max_clients, transform_df=True):
    df_input = df.copy()
    for col in ['sorted_clients_community', 'client_added', 'clients_current', 'clients_leaving', 'clients_in_queue']:
        df_input[col] = False
    date_range_que = get_date_range(df=df, col='datum')
    max_clients = 100
    clients_community = []
    clients_added = []
    clients_leaving = []
    clients_current = []
    clients_in_queue = []
    dict_queue = {}
    # len_capped_queue=100
    clients_current_day_before = []
    df_max_dates = df_input[['client_id', 'datum']].groupby(['client_id']).max()
    for selected_date in date_range_que:
        # Find clients that should be in total queue
        clients_community = list(df_input[df_input['datum'] == selected_date]['client_id'])
        dict_min_dates_clients_community = df_input[df_input['client_id'].isin(clients_community)][['client_id', 'datum']].groupby(['client_id']).min().to_dict()
        sorted_clients_community = sorted(clients_community, key=dict_min_dates_clients_community['datum'].get)
        # Find clients that are waiting in queue
        clients_in_queue = [client for client in sorted_clients_community if client not in clients_current_day_before]
        # Find clients that should be in capped queue
        max_clients_added = max_clients - len(clients_current_day_before)
        # Add new clients to current clienst based on room available
        clients_added = clients_in_queue[:max_clients_added]
        # Update current clients (with new clients based on room available)
        clients_current = clients_current_day_before + clients_added
        # Update clients in queue
        clients_in_queue = [client for client in sorted_clients_community if client not in clients_current]
        # Find clients that don't belong in queue (anymore)
        clients_leaving = list(df_max_dates[df_max_dates['datum'] == selected_date].index)
        # Update current clients (without leaving clients)
        clients_current = [client for client in clients_current if client not in clients_leaving]
        # Save current clients for next iteration
        clients_current_day_before = clients_current
        # len_capped_queue = len(capped_queue)
        dict_queue[selected_date] = {'clients_community': sorted_clients_community, 'clients_added': clients_added, 'clients_current': clients_current, 'clients_leaving': clients_leaving, 'clients_in_queue': clients_in_queue}
        if transform_df:
            df_input.loc[(df['client_id'].isin(sorted_clients_community)) & (df_input['datum'] == selected_date), 'sorted_clients_community'] = True
            df_input.loc[(df['client_id'].isin(clients_added)) & (df_input['datum'] == selected_date), 'client_added'] = True
            df_input.loc[(df['client_id'].isin(clients_added)) & (df_input['datum'] == selected_date), 'datum_opgenomen'] = selected_date.date()
            df_input.loc[(df['client_id'].isin(clients_current)) & (df_input['datum'] == selected_date), 'clients_current'] = True
            df_input.loc[(df['client_id'].isin(clients_leaving)) & (df_input['datum'] == selected_date), 'clients_leaving'] = True
            df_input.loc[(df['client_id'].isin(clients_in_queue)) & (df_input['datum'] == selected_date), 'clients_in_queue'] = True
    if transform_df == False:
        df_queue = pd.DataFrame.from_dict(dict_queue, orient='index')
        return df_queue
    else:
        return df_input
    

def create_dataset(n_clients=1000, max_clients=100, start_date='-5y', end_date='-1y'):
    df_community = create_clients_community(n_clients=n_clients, start_date=start_date, end_date=end_date)
    df_queue = create_queue_df(df=df_community, max_clients=max_clients)
    return df_queue

def aggretate_data_for_workshop(df, max_clients=100):
    _df= df.copy()
    df_minmax = _df.groupby(['datum']).agg({'clients_current': 'sum'}).reset_index()
    min_date = df_minmax[df_minmax['clients_current']==max_clients]['datum'].min()
    max_date = df_minmax[df_minmax['clients_current']==max_clients]['datum'].max()
    df_agg = _df[(_df['datum']>min_date) & (_df['datum']<max_date)].groupby(['datum', 'zorgzwaarte']).agg({'clients_current': 'sum'}).reset_index().sort_values(by=['datum', 'zorgzwaarte'])
    df_agg.rename(columns={'clients_current': 'aantal_clienten'}, inplace=True)
    return df_agg