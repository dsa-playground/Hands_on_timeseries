# Imports
import pandas as pd
import time


def collect_int_input(question, boundaries=[]):
    """Collect integer input given a question and range.

    Parameters
    ----------
    question : str
        Question that will be printed before the 'input()' command.
    boundaries : list, optional
        List with two integer values that determine the range within integer should be:
            * minimum (boundaries[0])
            * maximum (boundaries[1])

    Returns
    -------
    int
        Integer value collected from the user.
    """
    print(question)
    time.sleep(1)
    found = False

    while not found:
        try: 
            myInt = int(input("Antwoord: "), 10)
            # time.sleep(1)
        except Exception:
            print('Geef een geheel getal.')
            continue
        
        if len(boundaries)>0:
            if boundaries[0] <= myInt <= boundaries[1]:
                return myInt
            elif myInt > boundaries[1]:
                print(f"Vul een getal in dat lager is dan {boundaries[1]}.")
            elif myInt < boundaries[0]:
                print(f"Vul een getal in dat hoger is dan {boundaries[0]}.")
        else:
            return myInt


def collect_str_input(question, possible_entries=[]):
    """Collect string input given a question and possible entries (restrictions).

    Parameters
    ----------
    question : str
        Question that will be printed before the 'input()' command.
    possible_entries : list, optional
        List of strings that are allowed, by default all entries are allowed.

    Returns
    -------
    str
        String value (lowercase) collected from the user.

    Raises
    ------
    ValueError
        ValueError will be raised if string value is empty.
    """
    print(question)
    time.sleep(1)

    possible_entries = [entry.lower() for entry in possible_entries if isinstance(entry, str)]
    found = False

    while not found:
        try: 
            myStr = input("Antwoord: ").lower()
            # time.sleep(1)
            if not (myStr and myStr.strip()):
                raise ValueError('Leeg veld.')
        except Exception:
            print('Een leeg antwoord is niet bruikbaar.')
            continue
        
        if len(possible_entries)>0:
            if myStr in possible_entries:
                return myStr
            else:
                print(f"Je antwoord moet een van de volgende opties zijn: {possible_entries}.")
        else:
          return myStr


def add_record(dict_with_items_to_collect):
    """Constructs a dictionary with item-name as key and anwser as value.

    Parameters
    ----------
    dict_with_items_to_collect : dict
        Dictionary with a key for each item and a dictionary-value of the format (key:value):
            - type: int/str.
            - question: String sentence of question.
            - restriction: list of integer/string restrictions of possible answers.

    Returns
    -------
    dict
        Dictionary with anwser per item requested.
    """
    answers = {}
    for item in dict_with_items_to_collect.keys():
        if dict_with_items_to_collect[item]['type'] == 'str':
            answer = collect_str_input(
                question=dict_with_items_to_collect[item]['question'],
                possible_entries=dict_with_items_to_collect[item]['restriction']).capitalize()
        elif dict_with_items_to_collect[item]['type'] == 'int':
            answer = collect_int_input(
                question=dict_with_items_to_collect[item]['question'],
                boundaries=dict_with_items_to_collect[item]['restriction'])
        print(f"Je hebt ingevuld: {answer} \n")
        answers[item] = answer
    
    return answers


def add_multiple_records(dict_with_items_to_collect, continue_key='add', all_records=[]):
    """Constructs list of dictionaries with multiple anwsers. 

    Parameters
    ----------
    dict_with_items_to_collect : dict
        Dictionary with a key for each item and a dictionary-value of the format (key:value):
            - type: int/str.
            - question: String sentence of question.
            - restriction: list of integer/string restrictions of possible answers.
    continue_key : str, optional
        String value to check in last dictionary to add new record or return current collected anwsers, by default 'add'
    all_records : list, optional
        Empty list by default, because of recurrent function will be build up, by default []

    Returns
    -------
    list(dict)
        List of dictionaries for all anwsers given.
    """
    new_record = add_record(dict_with_items_to_collect=dict_with_items_to_collect)
    # print(f"start: {all_records}")
    if new_record[continue_key].lower() in ['ja', 'j']:
        all_records.append(new_record)
        # print(f"if loop: {all_records}")
        return add_multiple_records(
            dict_with_items_to_collect=dict_with_items_to_collect, 
            all_records=all_records)
    else:
        # print(f"elif loop: {all_records}")
        all_records.append(new_record)
        # print(f"elif loop + : {all_records}")
        return all_records


def transform_multiplechoice_anwser(list_with_dicts, dict_with_multiplechoice_anwsers):
    """Transforms multiple choice answer in multiple values of in dictionaries.

    Parameters
    ----------
    list_with_dicts : list(dict)
        List with dictionary of anwsers.
    dict_with_multiplechoice_anwsers : dict(str)
        Dict with key:value corresponding with multiple choice anwser.

    Returns
    -------
    list(dict)
        Corrected list of dictionaries for all anwsers given.
    """
    updated_list = []
    for item in list_with_dicts:
        updated_dict = {**item, **dict_with_multiplechoice_anwsers[item['multi']]}
        updated_list.append(updated_dict)
    return updated_list


def transform_multi_records_to_df(list_with_all_new_records, list_with_drop_columns):
    """Transforms list with dictionaries into Dataframe.

    Parameters
    ----------
    list_with_all_new_records : list(dict)
        List with dictionaries whereas each dictionary is one record of the new DataFrame
    list_with_drop_columns : list(str)
        List of strings with columnnames that can be dropped from the DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame constructed from underlying dictionaries with their keys as columnnames and values as values in the records. 
    """
    df_new_records = pd.DataFrame(list_with_all_new_records)
    df_new_records['Passagier_Id'] = df_new_records.index+10_000
    # df_new_records['Workshop_passagier'] = 1
    df_new_records = df_new_records.drop(columns=list_with_drop_columns)

    return df_new_records






# def add_records_ugly(config=_config):
#     name = collect_str_input(
#         question='Wat is je naam?')
#     age = collect_int_input(
#         question='Vul hier je leeftijd in:',
#         max=100,
#         min=0)
#     sex = collect_str_input(
#         question='Wat is je geslacht (man, vrouw, neutraal)?',
#         possible_entries=['man', 'vrouw', 'neutraal'])
#     kids = collect_int_input(
#         question='Hoeveel kinderen neem je mee op reis?',
#         max=10,
#         min=0)
#     family = collect_int_input(
#         question='Hoeveel familieleden gaan mee op reis gaan?',
#         max=10,
#         min=0)
#     multi = collect_str_input(
#         question="""
#             Geef aan welke optie je voorkeur geniet voor de overige variabelen:
#             A. Frankrijk, 1e klasse
#             B. Engeland, 1e klasse
#             C. Ierland, 1e klasse
#             """,
#         possible_entries=['a', 'b', 'c'])
#     multi = collect_str_input(
#         question="Wil je nog een passagier toevoegen?",
#         possible_entries=['ja', 'nee', 'j', 'n'])
#     return [name, age, sex, kids, family, multi]


# def spielerij_entry():
#     a = input("Flauwekul antwoord:")
#     b = input("Doorgaan met meer vragen?")
#     return [a,b]


# def test_str_input():
#     name = give_str_input(
#         question="Wat is je naam?")
#     sex = give_str_input(
#         question="Wat is je geslacht (man, vrouw, neutraal)?",
#         possible_entries=['man', 'vrouw', 'neutraal'])
#     return [name, sex]


# def add_record_old(config=_config):
#     answers = {}
#     for item in config['preprocess']['data']['collect']['items_to_collect']:
#         if config['preprocess']['data']['collect']['items_to_collect'][item]['type'] == 'str':
#             answer = collect_str_input(
#                 question=config['preprocess']['data']['collect']['items_to_collect'][item]['question'],
#                 possible_entries=config['preprocess']['data']['collect']['items_to_collect'][item]['restriction'])
#         elif config['preprocess']['data']['collect']['items_to_collect'][item]['type'] == 'int':
#             answer = collect_int_input(
#                 question=config['preprocess']['data']['collect']['items_to_collect'][item]['question'],
#                 boundaries=config['preprocess']['data']['collect']['items_to_collect'][item]['restriction'])
#         answers[item] = answer
    
#     return answers

# def transform_multiplechoice_anwser_old(list_with_dicts, config=_config):
#     updated_list = []
#     for item in list_with_dicts:
#         updated_dict = {**item, **config['preprocess']['data']['collect']['transform_multi'][item['multi']]}
#         updated_list.append(updated_dict)
#     return updated_list