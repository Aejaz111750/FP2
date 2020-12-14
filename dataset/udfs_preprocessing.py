import pandas as pd
import numpy as np
import time
import re
import collections
from collections import defaultdict
import math
from functools import reduce

#NLP
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
from spacy.matcher import Matcher
import nltk
from textblob import TextBlob

from IPython.display import IFrame
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from text2digits import text2digits
t2d = text2digits.Text2Digits(convert_ordinals=False)
t2d1 = text2digits.Text2Digits()

import pickle as pkl
from udfs_preprocessing import *

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score,classification_report


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 3000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    display(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def print_full1(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 3000)
    pd.set_option('display.float_format', '{:20,.0f}'.format)
    pd.set_option('display.max_colwidth', None)
    display(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
    
def split_alias(alias):
    return re.sub('___$','',re.sub('^(.*?)(?:\((.*)\))?$','\\1___\\2',alias)).split('___')

def lookup_key(name,country):
    if ' ' in name:
        name=name.lower()
    country=re.sub(' ','',country).lower()
    return name+'__'+country

def alias_list_tokenize_old(name,alias):
    player_stop_words=['ten']
    alias_list=split_alias(alias)
    name_list=[name]+alias_list
    if 'Ryan ten Doeschate' in name_list:
        name_list.append('ten doeschate')
        #name_list.append('10 doeschate')
        #name_list.append('ryan 10 doeschate')
    if 'Sachin Tendulkar' in name_list:
        name_list.append('Master Blaster')
    if 'AB de Villiers' in name_list:
        name_list.append('de Villiers')
    if 'Faf du Plessis' in name_list:
        name_list.append('du Plessis')
    if 'Faf du Plessis' in name_list:
        name_list.append('du Plessis')
    if 'MN van Wyk' in name_list:
        name_list.append('van Wyk')
    name_list=list(set(name_list))
    name_tokens=[]
    
    name_list=[re.sub(' +',' ',i).strip() for i in name_list]
    for name in name_list:
        for token in re.split('-| ',name):
            if len(token)>1 and token not in name_tokens and token not in player_stop_words:
                name_tokens.append(token)
    name_list = sorted(name_list, key=len,reverse=True) 
    name_tokens = sorted(name_tokens, key=len,reverse=True) 
    return name_list,name_tokens
# #testcase
# i=146
# name=df_player.iloc[i]['name']
# alias=df_player.iloc[i]['name_aliases']
# print('Name: %s'%name)
# print('Alias: %s'%alias)
# name_obj=alias_list_tokenize_old(name,alias)
# print('Multi-grams: %s'%name_obj[0])
# print('Uni-grams: %s'%name_obj[1])


def alias_list_tokenize(name, alias):
    player_stop_words = ['ten']
    alias_list = split_alias(alias)
    name_list = [name] + alias_list
    if 'Ryan ten Doeschate' in name_list:
        name_list.append('ten doeschate')
        # name_list.append('10 doeschate')
        # name_list.append('ryan 10 doeschate')
    if 'Sachin Tendulkar' in name_list:
        name_list.append('Master Blaster')
    if 'AB de Villiers' in name_list:
        name_list.append('de Villiers')
    if 'Faf du Plessis' in name_list:
        name_list.append('du Plessis')
    if 'Faf du Plessis' in name_list:
        name_list.append('du Plessis')
    if 'MN van Wyk' in name_list:
        name_list.append('van Wyk')
    name_list = [re.sub(' +', ' ', i).strip() for i in name_list]
    name_list = list(set(name_list))
    name_list_bigrams = []
    for name in name_list:
        name = re.sub(' +', ' ', name)
        name_list_bigrams.append([' '.join(i) for i in list(nltk.bigrams(nltk.word_tokenize(name)))])

    name_list_bigrams1 = []
    for name in name_list_bigrams:
        for bigram in name:
            name_list_bigrams1.append(bigram)
    name_list_bigrams = list(set(name_list_bigrams1))
    name_list = list(set(name_list_bigrams).union(set(name_list)))
    name_tokens = []
    for name in name_list:
        for token in re.split('-| ', name):
            if len(token) > 1 and token not in name_tokens and token not in player_stop_words:
                name_tokens.append(token)
    name_list = sorted(name_list, key=len, reverse=True)
    name_tokens = sorted(name_tokens, key=len, reverse=True)
    return name_list, name_tokens
# i=19
# name=df_player.iloc[i]['name']
# name_aliases=df_player.iloc[i]['name_aliases']
# alias_list_tokenize(name,name_aliases)

### Identify Country on strike
def batting_country(row,df_player,df_match):
    #print(row)
    player_country = df_player.loc[df_player['name']==row['batsman_on_strike'],'country'].values[0]
    country1 = df_match.loc[df_match['match_number']==row['match_number'],'country1'].values[0]
    country2 = df_match.loc[df_match['match_number']==row['match_number'],'country2'].values[0]
    if(player_country==country1):
        return(country1)
    else:
        return(country2)

def chasing_country(row,df_player,df_match):
    #print(row)
    player_country = df_player.loc[df_player['name']==row['batsman_on_strike'],'country'].values[0]
    country1 = df_match.loc[df_match['match_number']==row['match_number'],'country1'].values[0]
    country2 = df_match.loc[df_match['match_number']==row['match_number'],'country2'].values[0]
    if(player_country==country1):
        return(country2)
    else:
        return(country1)

def winby_wick_runs(x):
    if(x['winning_country']==x['batting_country']):
        runs = int(abs(x['runs_c1']-x['runs_c2']))
        return(str(runs)+' Runs')
       
    else:
        if str(x['out_c2'])=='nan':
            w=0
        else:
            w=int(abs(10-x['out_c2']))
        wickets = w
        return(str(wickets)+' Wickets')    

def find_overs(x):
    overs = np.floor((x['ball']-x['extra'])/6)
    balls = (x['ball']-x['extra'])%6
    return(str(int(overs))+'.'+str(balls))
    
def get_kw_dict(col,df_commentary):
    list_kw=df_commentary[df_commentary[col].notnull()][col].tolist()
    list_kw=[i.split(';') for i in list_kw]
    dict_kw={}
    for elements in list_kw:
        for element in elements:
            if element!='':
                dict_kw[element]=dict_kw.get(element,0)+1
    return dict_kw,set(dict_kw.keys())
def get_kw_counts(keyword):
    out_dict={}
    out_dict['shot_count']=dict_shot_type.get(keyword,0)
    out_dict['ball_type_count']=dict_ball_type.get(keyword,0)
    out_dict['fielding_position_count']=dict_fielding_position.get(keyword,0)
    out_dict['keywords_count']=dict_keywords.get(keyword,0)
    return out_dict

def def_value(): 
    return {}

def parse_article(article_text):
    # meta=''
    # PAT_META_SPLIT='^(.*?)(#p#.*)$'
    # PAT_META_SPLIT_search = re.search(PAT_META_SPLIT, article_text)
    # meta_text=PAT_META_SPLIT_search.group(1).strip()
    # text=PAT_META_SPLIT_search.group(2).strip()
    # peras=[i.strip() for i in text.split('#p#') if i!='']

    peras = [i.strip() for i in article_text.split('#p#') if i != '']
    peras_count = len(peras)

    line_num = 0
    line_list = []
    line_list_text = []
    for pera_num, pera_text in enumerate(peras):
        pera_num = pera_num + 1
        para_segments = [i.strip() for i in re.split('\(|\)', pera_text)]
        for segment in para_segments:
            # segment=re.sub('</?m[0-9]+>','',segment)
            doc = nlp(segment)
            for sent in doc.sents:
                sent_text = sent.text
                if sent_text != '.' or sent_text != '. ':
                    line_list_text.append(sent_text)
        line_list_text[-1] = line_list_text[-1] + '#p#'
    # print(line_list_text)

    # changed on 14th Dec
    # test_list = line_list_text[0:-1]
    test_list = line_list_text

    new_test_list = [test_list[0]]
    for i in range(1, len(test_list)):
        if test_list[i] == '>' or test_list[i] == '>,' or test_list[i] == '>.':
            new_test_list[-1] = new_test_list[-1] + test_list[i]
        else:
            new_test_list.append(test_list[i])
    test_list = new_test_list
    for i in range(0, 10):
        new_test_list = [test_list[0]]
        for i in range(1, len(test_list)):
            PAT_MENTION_ID_end = '</m([0-9]+)>?'
            PAT_MENTION_ID_end_search = re.search(PAT_MENTION_ID_end, test_list[i])
            if PAT_MENTION_ID_end_search:
                m_no = PAT_MENTION_ID_end_search.group(1)
                PAT_MENTION_ID_start = '<m{}>'.format(m_no)
                PAT_MENTION_ID_start_search = re.search(PAT_MENTION_ID_start, test_list[i])
                if not PAT_MENTION_ID_start_search:
                    new_test_list[-1] = new_test_list[-1] + test_list[i]
                else:
                    new_test_list.append(test_list[i])
            else:
                new_test_list.append(test_list[i])
        test_list = new_test_list
    # for element in test_list:
    #    print(element)
    test_list = [i.strip() for i in test_list if i not in ['', '>']]

    new_test_list = []
    for i in test_list:
        i = re.sub('^(>|,)* ?', '', i)
        i = re.sub('^\.#p#', '#p#', i)
        PAT_MENTION_ID_incomplete = re.search('(</m[0-9]+)$', i)
        if PAT_MENTION_ID_incomplete:
            i = re.sub('(</m[0-9]+)$', PAT_MENTION_ID_incomplete.group(1) + '>', i)
        new_test_list.append(i)

    line_list_text = new_test_list

    pera_num = 1
    line_num = 1
    line_list = []
    new_line_list = []
    for i in line_list_text:
        if '#p#' in i:
            pera_num = pera_num + 1
            i = re.sub('#p#', '', i)
        if i != '':
            mention_ids = re.findall('<m[0-9]+>?', i)
            mention_ids = [re.sub('(<m|>)', '', i) for i in mention_ids]
            # print(i,mention_ids)
            line_list.append([pera_num, line_num, i, mention_ids])
            line_num = line_num + 1
            new_line_list.append(i)
    new_line_list = [line_list[0] + ['']]

    for i in range(1, len(line_list)):
        if line_list[i - 1][2] != '':
            new_line_list.append(line_list[i] + [line_list[i - 1][2]])
        else:
            if line_list[i - 2][2] != '':
                new_line_list.append(line_list[i] + [line_list[i - 2][2]])
    # line_list_text=new_line_list
    out_dict = {}
    # out_dict['meta_text']=meta_text
    out_dict['peras_count'] = pera_num
    out_dict['line_count'] = len(line_list_text)
    out_dict['line_list'] = new_line_list
    # out_dict['line_list_text']=line_list_text
    return out_dict
# #testcase
# i=6
# df_match_articles.iloc[i]['article_text']
# test_list=parse_article(df_match_articles.iloc[i]['article_text'])
# df_match_articles.iloc[i]['article_number']
# test_list

def parse_article_try(article_text):
    try:
        return parse_article(article_text)
    except:
        out_dict={}
        #out_dict['meta_text']=meta_text
        out_dict['peras_count']=-1
        out_dict['line_count']=-1
        out_dict['line_list']=[]
        return out_dict

def parse_batsman_score(score):
    runs = score.split('(')[0]
    if '*' in runs:
        notout = 1
    else:
        notout = 0
    runs = re.sub('\*', '', runs)
    part2 = score.split('(')[1][:-1].split(' ')
    balls = re.sub('b', '', part2[0])
    fours = part2[1].split('x')[0]
    sixes = part2[2].split('x')[0]
    outdict = {}
    outdict['runs'] = runs.strip()
    outdict['notout'] = notout
    outdict['balls'] = balls.strip()
    outdict['fours'] = fours.strip()
    outdict['sixes'] = sixes.strip()
    return outdict
# i=100
# score=df_commentary.iloc[i]['batsman1_stats']
# print(score,parse_batsman_score(score))

def parse_bowler_score(score):
    score_parts = score.split('-')
    runs = score_parts[2]
    overs = score_parts[0]
    wickets = score_parts[3]
    outdict = {}
    outdict['runs'] = runs.strip()
    outdict['overs'] = overs.strip()
    outdict['wickets'] = wickets.strip()
    return outdict
# i=1100
# score=df_commentary.iloc[i]['bowler_stats']
# print(score,parse_bowler_score(score))

def country_player_identification(mention_text, country1, country2, dict_players_name, dict_players_name_token):
    contains_country1 = contains_country2 = 0
    contains_country1_player = contains_country2_player = 0
    maching_name_grams = []
    c1_player_ids = []
    c2_player_ids = []
    name_list = [j[0] for j in [i.split('__') for i in list(dict_players_name.keys())] if j[1] == country1.lower()]
    name_list = name_list + [j[0] for j in [i.split('__') for i in list(dict_players_name.keys())] if
                             j[1] == country2.lower()]
    name_list = sorted(name_list, key=len, reverse=True)
    name_token_list = [j[0] for j in [i.split('__') for i in list(dict_players_name_token.keys())] if
                       j[1] == country1.lower()]
    name_token_list = name_token_list + [j[0] for j in [i.split('__') for i in list(dict_players_name_token.keys())] if
                                         j[1] == country2.lower()]
    name_token_list = sorted(name_token_list, key=len, reverse=True)

    for name in name_list:
        pat = '(?i)(^| |>)' + name + '([: \',.<]|$)'
        if re.search(pat, mention_text):
            has_player_name = 1
            maching_name_grams.append(name)
            c1_player_id = dict_players_name.get(name + '__' + country1.lower(), 'UN')
            if c1_player_id != 'UN':
                country_name = country1.lower()
                country = 1
                contains_country1_player = 1
                c1_player_ids.append(c1_player_id)
                replacement = '<<C' + str(country) + 'P' + c1_player_id + '>>'
                mention_text = re.sub(name, replacement, mention_text, flags=re.IGNORECASE)
            c2_player_id = dict_players_name.get(name + '__' + country2.lower(), 'UN')
            if c2_player_id != 'UN':
                country_name = country2.lower()
                country = 2
                contains_country2_player = 1
                c2_player_ids.append(c2_player_id)
                replacement = '<<C' + str(country) + 'P' + c2_player_id + '>>'
                mention_text = re.sub(name, replacement, mention_text, flags=re.IGNORECASE)
    for name in name_token_list:
        pat = '(^| |>)' + name + '([: \',.<]|$)'
        if re.search(pat, mention_text):
            has_player_name = 1
            maching_name_grams.append(name)
            c1_player_id = dict_players_name_token.get(name + '__' + country1.lower(), 'UN')
            # correffrance example i=1086
            if c1_player_id != 'UN' and ('_' not in c1_player_id or not (
                    c1_player_id.split('_')[0] in c1_player_ids or c1_player_id.split('_')[1] in c1_player_ids)):
                # if c1_player_id!='UN':
                country_name = country1.lower()
                country = 1
                contains_country1_player = 1
                c1_player_ids.append(c1_player_id)
                replacement = '<<C' + str(country) + 'P' + c1_player_id + '>>'
                mention_text = re.sub(name, replacement, mention_text, flags=re.IGNORECASE)
            c2_player_id = dict_players_name_token.get(name + '__' + country2.lower(), 'UN')
            if c2_player_id != 'UN' and ('_' not in c2_player_id or not (
                    c2_player_id.split('_')[0] in c2_player_ids or c2_player_id.split('_')[1] in c2_player_ids)):
                # if c2_player_id!='UN':
                country_name = country2.lower()
                country = 2
                contains_country2_player = 1
                c2_player_ids.append(c2_player_id)
                replacement = '<<C' + str(country) + 'P' + c2_player_id + '>>'
                mention_text = re.sub(name, replacement, mention_text, flags=re.IGNORECASE)

    if country1 in mention_text:
        contains_country1 = 1
        mention_text = re.sub(country1, '<<C1>>', mention_text)
    if country2 in mention_text:
        contains_country2 = 1
        mention_text = re.sub(country2, '<<C2>>', mention_text)

    c1_player_ids = list(set(c1_player_ids))
    c2_player_ids = list(set(c2_player_ids))
    out_dict = {}
    out_dict['mention_text'] = mention_text
    out_dict['contains_country1'] = contains_country1
    out_dict['contains_country2'] = contains_country2
    out_dict['count_country'] = contains_country1 + contains_country2
    out_dict['contains_country1_player'] = contains_country1_player
    out_dict['contains_country2_player'] = contains_country2_player
    out_dict['c1_player_ids'] = c1_player_ids
    out_dict['c2_player_ids'] = c2_player_ids
    out_dict['count_players_country'] = contains_country1_player + contains_country2_player
    out_dict['count_players'] = len(c1_player_ids) + len(c2_player_ids)
    out_dict['maching_name_grams'] = maching_name_grams
    return out_dict
# #testcase
# i=0
# mention_text=train_data1.iloc[i]['canonical_mention_text']
# country1=train_data1.iloc[i]['country1']
# country2=train_data1.iloc[i]['country2']
# print('Mention Text: %s'%mention_text)
# print('Country1: %s'%country1)
# print('Country2: %s'%country2)
# country_player_identification(mention_text,country1,country2)

def clean_br(input_string):
    input_string = re.sub(' +', ' ', re.sub('>>', ' ', re.sub('<<', ' ', input_string))).strip()
    return input_string

def country_player_identification1(input_string, country1, country2, dict_players_name, dict_players_name_token):
    input_string = \
    country_player_identification(input_string, country1, country2, dict_players_name, dict_players_name_token)[
        'mention_text']
    return clean_br(input_string)


def process_files1(df_player, df_commentary,other_prams):
    [fn_player_ids, fn_player_dicts_pkls, fn_country_wise_stats, fn_match_stats, fn_batter_matches,
                   fn_bowler_matches]=other_prams
    ## Player Data
    ### Countries - Team Size
    df_player_country = df_player.groupby(['country']).agg({'name': 'count'}).reset_index().rename(
        columns={'name': 'players'})
    df_player_country.sort_values(by=['players'], ascending=[False], inplace=True)
    df_player = df_player.reset_index()
    df_player['player_id'] = (df_player['index'] + 1).astype('str')
    df_player.drop(columns=['index'], inplace=True)
    df_player['name_obj'] = df_player.apply(lambda x: alias_list_tokenize(x['name'], x['name_aliases']), axis=1)
    df_player['name_list'] = df_player.apply(lambda x: x['name_obj'][0], axis=1)
    df_player['name_tokens'] = df_player.apply(lambda x: x['name_obj'][1], axis=1)
    col_order = ['player_id', 'country', 'name', 'name_aliases', 'name_list', 'name_tokens']
    df_player = df_player[col_order]
    ### name
    df_player_ids = df_player[['name_list', 'country', 'player_id']].explode('name_list')
    df_player_ids_grouped = df_player_ids.groupby(['name_list', 'country']).aggregate(
        {'player_id': lambda x: list(x)}).reset_index()
    df_player_ids_grouped['id_count'] = df_player_ids_grouped.apply(lambda x: len(x['player_id']), axis=1)
    df_player_ids_grouped.columns = ['name', 'country', 'player_ids_list', 'id_count']
    df_player_ids_grouped['player_ids_str'] = df_player_ids_grouped.apply(
        lambda x: '_'.join([str(i) for i in x['player_ids_list']]), axis=1)
    df_player_ids_grouped = df_player_ids_grouped[['name', 'country', 'player_ids_str', 'player_ids_list', 'id_count']]
    df_player_ids_grouped['lookup_key'] = df_player_ids_grouped.apply(lambda x: lookup_key(x['name'], x['country']),
                                                                      axis=1)

    ### name token
    df_player_ids_name_token = df_player[['name_tokens', 'country', 'player_id']].explode('name_tokens')
    df_player_ids_grouped_tokens = df_player_ids_name_token.groupby(['name_tokens', 'country']).aggregate(
        {'player_id': lambda x: list(x)}).reset_index()
    df_player_ids_grouped_tokens['id_count'] = df_player_ids_grouped_tokens.apply(lambda x: len(x['player_id']), axis=1)
    df_player_ids_grouped_tokens.columns = ['name_token', 'country', 'player_ids_list', 'id_count']
    df_player_ids_grouped_tokens['player_ids_str'] = df_player_ids_grouped_tokens.apply(
        lambda x: '_'.join([str(i) for i in x['player_ids_list']]), axis=1)
    df_player_ids_grouped_tokens = df_player_ids_grouped_tokens[
        ['name_token', 'country', 'player_ids_str', 'player_ids_list', 'id_count']]

    df_player_ids_grouped_tokens['lookup_key'] = df_player_ids_grouped_tokens.apply(
        lambda x: lookup_key(x['name_token'], x['country']), axis=1)
    dict_players_name = dict(df_player_ids_grouped[['lookup_key', 'player_ids_str']].values.tolist())
    dict_players_name_token = dict(df_player_ids_grouped_tokens[['lookup_key', 'player_ids_str']].values.tolist())

    dict_player_ids = dict(df_player[['name', 'player_id']].values.tolist())
    dict_player_names = dict(df_player[['player_id', 'name', ]].values.tolist())
    df_player[['player_id', 'name']].to_csv(fn_player_ids, index=False, encoding='utf_8_sig')
    pkl.dump([dict_players_name, dict_players_name_token, dict_player_ids, dict_player_names],
             open(fn_player_dicts_pkls, 'wb'))

    ## Commentary Data
    ###Calculate Number of winnings per country
    ###Identify the Batting Country and Chasing country for each record
    ###Calculate runs made by each country in each match
    ###Identify the highest and lowest runs for each country
    ###Calculate fours, sixes and wickets taken by each country
    df_match = df_commentary[['match_number', 'country1', 'country2', 'winning_country']].drop_duplicates().reset_index(
        drop=True)
    countries_matches = df_match.country1.to_list()
    countries_matches.extend(df_match.country2)
    country_matches = pd.DataFrame.from_dict(collections.Counter(countries_matches), orient='index',
                                             columns=['Matches']).reset_index()
    country_matches = country_matches.rename(columns={'index': 'Country'})
    country_match = pd.merge(country_matches, pd.DataFrame(df_match.winning_country.value_counts().reset_index()),
                             left_on='Country', right_on='index', how='left').drop('index', axis=1)
    country_match = country_match.fillna(0)  ## Replace NaNs with zero
    country_match = country_match.rename(columns={'winning_country': 'Wins'})
    country_match.sort_values(by=['Wins'], ascending=False, inplace=True)
    df_tied_NR = df_match[~((df_match['winning_country'].isin(df_match['country1'])) |
                            (df_match['winning_country'].isin(df_match['country2'])))][
        ['match_number', 'country1', 'country2', 'winning_country']]
    df_tied_NR = pd.wide_to_long(df_tied_NR, ["country"], i="match_number", j="country_id").reset_index()
    df_tied_NR = pd.DataFrame(df_tied_NR.country.value_counts().reset_index())
    df_tied_NR.columns = ['Country', 'tied_nr']
    country_match = pd.merge(country_match, df_tied_NR, on='Country', how='left').fillna(0)
    country_match['Lost'] = country_match['Matches'] - country_match['Wins'] - country_match['tied_nr']
    ### Percentage of Winnings
    country_match['Win_Per'] = np.round(country_match['Wins'] / country_match['Matches'], 2)
    df_commentary['batting_country'] = df_commentary.apply(lambda x: batting_country(x, df_player, df_match), axis=1)
    df_commentary['chasing_country'] = df_commentary.apply(lambda x: chasing_country(x, df_player, df_match), axis=1)
    match_country_score = df_commentary.groupby(['match_number',
                                                 'batting_country']).agg({'total_runs': 'max'}).reset_index()
    ### Calculate highest score for each country
    highest_score = match_country_score.groupby(['batting_country']).agg({'total_runs': 'max'}).reset_index()
    highest_score = highest_score.rename(columns={'total_runs': 'Highest_score'})
    ### Calculate lowest score for each country
    lowest_score = match_country_score.groupby(['batting_country']).agg({'total_runs': 'min'}).reset_index()
    lowest_score = lowest_score.rename(columns={'total_runs': 'Lowest_score'})
    ### Merge country runs, highest score and lowest score
    country_match = pd.merge(country_match, highest_score, left_on='Country', right_on='batting_country', how='left')
    country_match = pd.merge(country_match, lowest_score, left_on='Country', right_on='batting_country', how='left')
    country_match.drop(columns=['batting_country_x', 'batting_country_y'], inplace=True)
    Country_wise_scores = match_country_score.groupby(['batting_country']).agg({'total_runs': 'sum'}).reset_index()
    Country_wise_scores = Country_wise_scores.rename(columns={'total_runs': 'runs'})
    ### Calculate avg runs for each country
    country_wise_stats = pd.merge(country_match, Country_wise_scores, left_on='Country', right_on='batting_country')
    country_wise_stats['Avg_runs'] = country_wise_stats['runs'] / country_wise_stats['Matches']
    #### Total wickets taken by each country
    wickets_each_match = df_commentary.groupby(['chasing_country', 'match_number']).agg(
        {'total_wickets': 'max'}).reset_index()
    country_wickets = wickets_each_match.groupby('chasing_country').agg({'total_wickets': 'sum'}).reset_index()
    country_wise_stats = pd.merge(country_wise_stats, country_wickets, left_on='Country', right_on='chasing_country')
    country_wise_stats = country_wise_stats.loc[:,
                         ~country_wise_stats.columns.str.contains('_country')]  ## REmove duplicate col names
    ###   Fours by each country
    fours_per_country = df_commentary[df_commentary['event_type'] == 'four'].groupby('batting_country').agg(
        {'event_type': 'count'}).reset_index()
    fours_per_country = fours_per_country.rename(columns={'event_type': 'fours'})
    ### 6s by each country
    six_per_country = df_commentary[df_commentary['event_type'] == 'six'].groupby('batting_country').agg(
        {'event_type': 'count'}).reset_index()
    six_per_country = six_per_country.rename(columns={'event_type': 'six'})
    ### Merge all the above to country wise stats
    country_wise_stats = pd.merge(country_wise_stats, fours_per_country, left_on='Country', right_on='batting_country')
    country_wise_stats = pd.merge(country_wise_stats, six_per_country, left_on='Country', right_on='batting_country')
    country_wise_stats = country_wise_stats.loc[:, ~country_wise_stats.columns.str.contains('_country')]
    country_wise_stats['Boundaries_score'] = country_wise_stats['fours'] * 4 + country_wise_stats['six'] * 6
    country_wise_stats = country_wise_stats.sort_values(by='Wins', ascending=False)
    country_wise_stats.columns = ['country', 'matches', 'wins', 'tied_nr', 'losses', 'win%', 'highest_score',
                                  'lowest_score', 'runs', 'avg_runs', 'wickets', '4s', '6s', '4s_6s_runs']
    country_wise_stats['avg_runs'] = np.round(country_wise_stats['avg_runs'], 1)
    country_wise_stats['wins'] = country_wise_stats['wins'].astype('int')
    country_wise_stats['losses'] = country_wise_stats['losses'].astype('int')
    country_wise_stats['tied_nr'] = country_wise_stats['tied_nr'].astype('int')

    ### Match Results
    df_match = df_commentary[['match_number', 'country1', 'country2', 'winning_country']].drop_duplicates().reset_index(
        drop=True)
    # Calculate total runs made by country-1 in Innings-1 and country-2 in Innings-2
    # Calculate fours, sixes, wickets taken and extras given by each country
    runs_pivot = df_commentary.pivot_table('total_runs', ['match_number'], 'innings_number',
                                           aggfunc='max').reset_index()
    ### Group by
    event_type_pivot = df_commentary.groupby(['match_number', 'innings_number', 'event_type']).agg(
        {'event_type': 'count'}).reset_index(level=[0, 1])
    event_type_pivot['event'] = event_type_pivot.index
    event_type_pivot = event_type_pivot.reset_index(drop=True)
    event_type_pivot = event_type_pivot.rename(columns={'event_type': 'count', 'event': 'event_type'})
    event_type_pivot_2 = event_type_pivot.pivot_table('count', ['match_number', 'innings_number'], 'event_type',
                                                      aggfunc='sum').reset_index()
    # event_type_pivot_2.head()
    ## Fours and six
    fours_pivot = event_type_pivot_2.pivot_table('four', ['match_number'], 'innings_number',
                                                 aggfunc='sum').reset_index()
    out_pivot = event_type_pivot_2.pivot_table('out', ['match_number'], 'innings_number', aggfunc='sum').reset_index()
    six_pivot = event_type_pivot_2.pivot_table('six', ['match_number'], 'innings_number', aggfunc='sum').reset_index()
    wide_pivot = event_type_pivot_2.pivot_table('wide', ['match_number'], 'innings_number', aggfunc='sum').reset_index()
    df_match_countries = df_commentary[['match_number', 'country1', 'country2', 'winning_country',
                                        'batting_country', 'chasing_country']]
    df_match_countries = df_match_countries.drop_duplicates(subset=['match_number'])
    df_match['runs_c1'] = runs_pivot.iloc[0:][1]
    df_match['runs_c2'] = runs_pivot.iloc[0:][2]
    df_match['four_c1'] = fours_pivot.iloc[0:][1]
    df_match['four_c2'] = fours_pivot.iloc[0:][2]
    df_match['six_c1'] = six_pivot.iloc[0:][1]
    df_match['six_c2'] = six_pivot.iloc[0:][2]
    df_match['out_c1'] = out_pivot.iloc[0:][1]
    df_match['out_c2'] = out_pivot.iloc[0:][2]
    df_match['wide_c1'] = wide_pivot.iloc[0:][1]
    df_match['wide_c2'] = wide_pivot.iloc[0:][2]
    df_match_countries = df_commentary[
        ['match_number', 'country1', 'country2', 'winning_country', 'batting_country', 'chasing_country']]
    df_match_countries = df_match_countries.drop_duplicates(subset=['match_number'])
    df_match = pd.merge(df_match, df_match_countries.drop(columns=['country1', 'country2', 'winning_country']),
                        on='match_number')
    df_match['win_margin'] = df_match.apply(lambda x: winby_wick_runs(x), axis=1)
    df_match.rename(columns={'match_number': 'match', 'winning_country': 'won_by', 'out_c1': 'w_c1', 'out_c2': 'w_c2'},
                    inplace=True)
    cols = ['match', 'country1', 'country2', 'chasing_country', 'won_by', 'win_margin',
            'runs_c1', 'runs_c2', 'w_c1', 'w_c2', 'four_c1', 'four_c2', 'six_c1', 'six_c2',
            'wide_c1', 'wide_c2']
    df_match = df_match[cols]
    df_match = df_match.fillna(0)
    df_match['runs_c1'] = df_match['runs_c1'].astype('int')
    df_match['runs_c2'] = df_match['runs_c2'].astype('int')
    df_match['w_c1'] = df_match['w_c1'].astype('int')
    df_match['w_c2'] = df_match['w_c2'].astype('int')
    df_match['four_c1'] = df_match['four_c1'].astype('int')
    df_match['four_c2'] = df_match['four_c2'].astype('int')
    df_match['six_c1'] = df_match['six_c1'].astype('int')
    df_match['six_c2'] = df_match['six_c2'].astype('int')
    sel_cols = ['match', 'country1', 'country2', 'chasing_country', 'won_by', 'win_margin',
                'runs_c1', 'runs_c2', 'w_c1', 'w_c2', 'four_c1', 'four_c2', 'six_c1', 'six_c2']
    ### Batting Records
    # Group by 'batsman_on_strike' and 'match_number' to get number of matches played by each player
    # Calculate below stats for each batter:
    #     total runs
    #     total fours and sixes
    #     highest Scores of each player:
    #     average score of each batter across the matches
    #     total centuries and half centuries by each player across the matches
    # No of matches by player
    batter_matches = df_commentary.groupby(['batsman_on_strike']).agg({'match_number': pd.Series.nunique}).reset_index()
    batter_matches.rename(columns={'match_number': 'No of Innings'}, inplace=True)
    batter_matches = pd.merge(batter_matches, df_player, left_on='batsman_on_strike', right_on='name', how='left')
    batter_matches = pd.merge(batter_matches,
                              df_commentary.groupby(['batsman_on_strike']).agg({'runs': 'sum'}).reset_index(),
                              on='batsman_on_strike', how='left')
    ## group by match number and batsman on strike for getting different types of events for the player in the match
    event_type_pivot_player = df_commentary.groupby(['match_number', 'batsman_on_strike', 'event_type']).agg(
        {'event_type': 'count'}).reset_index(level=[0, 1]).rename(columns={'event_type': 'Count'})
    event_type_pivot_player['event_type'] = event_type_pivot_player.index
    player_fours = event_type_pivot_player[event_type_pivot_player['event_type'] == 'four'].rename(
        columns={'Count': 'Fours'})
    player_six = event_type_pivot_player[event_type_pivot_player['event_type'] == 'six'].rename(
        columns={'Count': 'Sixes'})
    batter_matches = pd.merge(batter_matches,
                              player_fours.groupby('batsman_on_strike').agg({'Fours': 'sum'}).reset_index(),
                              on='batsman_on_strike', how='left')
    batter_matches = pd.merge(batter_matches,
                              player_six.groupby('batsman_on_strike').agg({'Sixes': 'sum'}).reset_index(),
                              on='batsman_on_strike', how='left')
    ## Calculate Highest score in a match
    player_match_scores = df_commentary.groupby(['batsman_on_strike', 'match_number']).agg(
        {'runs': 'sum'}).reset_index()
    player_high_scores = player_match_scores.groupby(['batsman_on_strike']).agg({'runs': 'max'}).reset_index()
    player_high_scores = player_high_scores.rename(columns={'runs': 'higest_score'})
    player_avg_scores = player_match_scores.groupby(['batsman_on_strike']).agg({'runs': 'mean'}).reset_index()
    player_avg_scores = player_avg_scores.rename(columns={'runs': 'avg'})
    ## Merge batter matches, player high scores and avg scores
    players_scores = [batter_matches, player_high_scores, player_avg_scores]
    batter_matches = reduce(lambda left, right: pd.merge(left, right, on='batsman_on_strike'), players_scores)
    # batter_matches = batter_matches.drop(columns=['url'])
    batter_stats = df_commentary.groupby(['batsman_on_strike', 'match_number']).agg({'runs': 'sum'}).reset_index()
    batter_stats['100s'] = batter_stats['runs'].apply(lambda x: math.floor(x / 100))
    batter_stats['50s'] = batter_stats['runs'].apply(lambda x: math.floor(x / 50))
    batter_stats_ms = batter_stats.groupby('batsman_on_strike').agg({'100s': 'sum', '50s': 'sum'}).reset_index()
    batter_matches = pd.merge(batter_matches, batter_stats_ms, on='batsman_on_strike', how='left')
    batter_matches = batter_matches[['name', 'country', 'No of Innings', 'runs', 'higest_score', 'avg',
                                     'Sixes', 'Fours', '100s', '50s']]
    batter_matches = batter_matches.rename(columns={'name': 'player_name',
                                                    'Fours': '4s', 'Sixes': '6s'})
    ## Fill 'NaN's with '0'
    batter_matches = batter_matches.fillna(0)
    batter_matches['6s'] = batter_matches['6s'].astype('int')
    batter_matches['4s'] = batter_matches['6s'].astype('int')
    batter_matches.rename(
        columns={'player_name': 'batter', 'No of Innings': 'innings', 'higest_score': 'highest_score'}, inplace=True)

    # Bowler Matches
    # Calculate below stats for each bowler
    #     total runs given by each bowler across the matches
    #     total overs and balls boweled by each bowler
    #     number of wickets taken by each bowler
    #     economoy of bowler
    #         Total runs / overs
    #     average of bowler
    #         total runs / total wickets
    #     total 4s, 6s and extras given by each player
    df_commentary['bol_obj'] = df_commentary.apply(lambda x: x['ball_number'].split('.'), axis=1)
    df_commentary['over'] = df_commentary.apply(lambda x: x['bol_obj'][0], axis=1).astype('int')
    df_commentary['ball'] = df_commentary.apply(lambda x: x['bol_obj'][1], axis=1).astype('int')
    df_commentary['extra'] = df_commentary.apply(lambda x: x['bol_obj'][2], axis=1).astype('int')
    df_commentary.drop(columns=['bol_obj'], inplace=True)
    ## Extras over wise
    df_commentary['extra'] = pd.to_numeric(df_commentary['extra'], errors='coerce')
    # No of matches by player
    bowler_matches = df_commentary.groupby(['bowler']).agg({'match_number': pd.Series.nunique,
                                                            'ball': 'count', 'extra': 'sum',
                                                            'runs': 'sum'}).reset_index().rename(
        columns={'match_number': 'Inns'})
    bowler_matches = pd.merge(df_player, bowler_matches, left_on='name', right_on='bowler', how='right')

    # bowler_matches['overs'] = np.flo((bowler_matches['ball']-bowler_matches['extra'])/6,2)
    bowler_matches['overs'] = bowler_matches.apply(find_overs, axis=1)
    bowler_matches['econ'] = bowler_matches.apply(lambda x: np.round(x['runs'] / np.float(x['overs']), 2), axis=1)
    event_type_pivot_bowler = df_commentary.groupby(['match_number', 'bowler',
                                                     'event_type']).agg({'event_type': 'count'}).reset_index(
        level=[0, 1])
    event_type_pivot_bowler.rename(columns={'event_type': 'Count'}, inplace=True)
    event_type_pivot_bowler['event_type'] = event_type_pivot_bowler.index
    event_type_pivot_bowler.reset_index(drop=True, inplace=True)
    bowler_event_stats = event_type_pivot_bowler.groupby(['bowler', 'event_type']).agg({'Count': 'sum'}).reset_index(
        level=[0, 1])
    bowler_event_stats = bowler_event_stats.pivot_table(values='Count', index=['bowler'],
                                                        columns="event_type").reset_index()
    bowler_event_stats = bowler_event_stats.rename(columns={'out': 'wickets'})
    bowler_matches = pd.merge(bowler_matches, bowler_event_stats[['bowler', 'wickets', 'four', 'six']],
                              on='bowler', how='left')
    ### Calculate average for each bowler
    bowler_matches['ave'] = bowler_matches.apply(lambda x: np.round(x['runs'] / x['wickets'], 2), axis=1)
    # bowler_matches = bowler_matches.drop(columns=['player','bowler','ball']).sort_values(by='wickets',ascending=False)
    bowler_matches = bowler_matches[['name', 'country', 'Inns', 'overs', 'runs', 'wickets', 'ave',
                                     'econ', 'four', 'six', 'extra']].sort_values(by='wickets', ascending=False)
    ## Rename columns
    bowler_matches = bowler_matches.rename(columns={'name': 'bolwer', 'Inns': 'innings',
                                                    'four': '4s', 'six': '6s', 'extra': 'extras', 'ave': 'avg'})
    bowler_matches = bowler_matches.fillna(0)
    bowler_matches['wickets'] = bowler_matches['wickets'].astype('int')
    bowler_matches['6s'] = bowler_matches['6s'].astype('int')
    bowler_matches['4s'] = bowler_matches['6s'].astype('int')

    df_commentary['commentary_line_length'] = df_commentary.apply(lambda x: len(x['original_commentary_line']), axis=1)
    for col in ['batsman_on_strike', 'other_batsman', 'bowler', 'out_player']:
        out_col = col + '_id'
        df_commentary[out_col] = df_commentary.apply(lambda x: dict_player_ids.get(x[col], ''), axis=1)
    df_commentary.sort_values(by=['match_number', 'innings_number', 'over', 'ball', 'extra'], inplace=True)
    df_commentary = df_commentary.reset_index(drop=True).reset_index()
    df_commentary['match_inn_ball_number'] = df_commentary.groupby(
        ['match_number', 'innings_number'])['index'].rank("dense", ascending=True).astype('int')
    # add lastball flag
    df_commentary_inn_last_balls = df_commentary.groupby(['match_number', 'innings_number'])[
        'match_inn_ball_number'].max().reset_index()
    df_commentary_inn_last_balls['inn_lastball'] = 1
    df_commentary = pd.merge(df_commentary,
                             df_commentary_inn_last_balls,
                             how='left',
                             left_on=['match_number', 'innings_number', 'match_inn_ball_number'],
                             right_on=['match_number', 'innings_number', 'match_inn_ball_number'])
    df_commentary['inn_lastball'].fillna(0, inplace=True)
    df_commentary['inn_lastball'] = df_commentary['inn_lastball'].astype('int')
    df_commentary['inn_ball_number'] = df_commentary.apply(lambda x: str(x['innings_number']) + ':' +
                                                                     x['ball_number'], axis=1)
    df_commentary['match_ball_id'] = df_commentary.apply(lambda x: str(x['match_number']) + '_' + x['inn_ball_number'],
                                                         axis=1)

    df_commentary['batsman_score_obj'] = df_commentary.apply(lambda x: parse_batsman_score(x['batsman1_stats']), axis=1)
    df_commentary['batsman_runs'] = df_commentary.apply(lambda x: x['batsman_score_obj']['runs'], axis=1)
    df_commentary['batsman_balls'] = df_commentary.apply(lambda x: x['batsman_score_obj']['balls'], axis=1)
    df_commentary.drop(columns=['batsman_score_obj'], inplace=True)

    df_commentary['bowler_score_obj'] = df_commentary.apply(lambda x: parse_bowler_score(x['bowler_stats']), axis=1)
    df_commentary['bowler_runs'] = df_commentary.apply(lambda x: x['bowler_score_obj']['runs'], axis=1)
    df_commentary['bowler_overs'] = df_commentary.apply(lambda x: x['bowler_score_obj']['overs'], axis=1)
    df_commentary['bowler_wickets'] = df_commentary.apply(lambda x: x['bowler_score_obj']['wickets'], axis=1)
    df_commentary.drop(columns=['bowler_score_obj'], inplace=True)

    df_commentary['canonical_commentary_line'] = df_commentary.apply(
        lambda x: norm_mention_text(x['original_commentary_line']), axis=1)
    df_commentary['canonical_commentary_line'] = \
        df_commentary.apply(lambda x: country_player_identification1(
            x['canonical_commentary_line'], x['country1'], x['country2'], dict_players_name, dict_players_name_token),
                            axis=1)

    # df_commentary['canonical_commentary_line']=df_commentary.apply(lambda x: clean_brs(x['canonical_commentary_line']),axis=1)
    # export
    country_wise_stats.to_csv(fn_country_wise_stats, index=False, encoding='utf_8_sig')
    df_match.to_csv(fn_match_stats, index=False, encoding='utf_8_sig')
    batter_matches.to_csv(fn_batter_matches, index=False, encoding='utf_8_sig')
    bowler_matches.to_csv(fn_bowler_matches, index=False, encoding='utf_8_sig')

    ## Process Keywords (ball, shot, fielding, others) and build Taxonomies
    dict_shot_type, list_shot_type = get_kw_dict('shot_type', df_commentary)
    dict_ball_type, list_ball_type = get_kw_dict('ball_type', df_commentary)
    dict_fielding_position, list_fielding_position = get_kw_dict('fielding_position', df_commentary)
    dict_keywords, list_keywords = get_kw_dict('keywords', df_commentary)
    list_keywords_filtered = []
    for i in list(list_keywords):
        if i not in list_shot_type and i not in list_fielding_position and i not in list_ball_type:
            list_keywords_filtered.append(i)
    list_keywords_filtered = set(list_keywords_filtered)

    df_ball_kw = pd.DataFrame.from_dict(dict_ball_type, orient='index', columns=['count']).reset_index().rename(
        columns={'index': 'ball'}).sort_values(by=['count'], ascending=False)
    df_shot_kw = pd.DataFrame.from_dict(dict_shot_type, orient='index', columns=['count']).reset_index().rename(
        columns={'index': 'shot'}).sort_values(by=['count'], ascending=False)
    df_field_kw = pd.DataFrame.from_dict(dict_fielding_position, orient='index',
                                         columns=['count']).reset_index().rename(
        columns={'index': 'field_position'}).sort_values(by=['count'], ascending=False)
    df_kw = pd.DataFrame.from_dict(dict_keywords, orient='index', columns=['count']).reset_index().rename(
        columns={'index': 'kws'}).sort_values(by=['count'], ascending=False)
    df_other_kws = df_kw[df_kw['kws'].isin(list_keywords_filtered)].rename(columns={'kws': 'other_keywords'})
    ent_tagg_taxonomy = [list_shot_type, list_ball_type, list_fielding_position, list_keywords_filtered]
    out_dict = {}
    out_dict['df_match'] = df_match
    out_dict['df_commentary'] =df_commentary
    out_dict['df_player'] =df_player
    out_dict['df_player_ids'] =df_player_ids
    out_dict['df_player_ids_grouped'] =df_player_ids_grouped
    out_dict['ent_tagg_taxonomy'] =ent_tagg_taxonomy
    return out_dict


def process_labelled_data_complete(df_labeled_data, df_match, df_match_articles, other_prams):
    [dict_players_name, dict_players_name_token] = other_prams
    df_labeled_data = df_labeled_data.merge(df_match, left_on='match_number', right_on='match').drop(columns=['match'])
    df_labeled_data['mention_key'] = df_labeled_data.apply(
        lambda x: str(x['article_number']) + '_' + re.sub('m', '', x['mention_id']), axis=1)
    df_labeled_data['mention_text'] = df_labeled_data.apply(lambda x: re.sub(' +', ' ', x['mention_text']).strip(),
                                                            axis=1)
    df_labeled_data['label_detailed_obj'] = df_labeled_data.apply(
        lambda x: parse_label_detailed(x['label_detailed'], x['country1'],
                                       x['country2'], dict_players_name, dict_players_name_token), axis=1)
    df_labeled_data['label_part1'] = df_labeled_data.apply(lambda x: x['label_detailed_obj']['part1'], axis=1)
    df_labeled_data['label_part2'] = df_labeled_data.apply(lambda x: x['label_detailed_obj']['part2'], axis=1)
    df_labeled_data['label_sub'] = df_labeled_data.apply(lambda x: x['label_detailed_obj']['sub'], axis=1)
    df_labeled_data['label_runs'] = df_labeled_data.apply(lambda x: x['label_detailed_obj']['runs'], axis=1)
    df_labeled_data['label_wickets_balls'] = df_labeled_data.apply(lambda x: x['label_detailed_obj']['wickets_balls'],
                                                                   axis=1)
    df_labeled_data['label_rrr'] = df_labeled_data.apply(lambda x: x['label_detailed_obj']['rrr'], axis=1)
    df_labeled_data['label_powerplay_info'] = df_labeled_data.apply(lambda x: x['label_detailed_obj']['powerplay_info'],
                                                                    axis=1)
    df_labeled_data.drop(columns=['label_detailed_obj'], inplace=True)
    df_labeled_data['country1'] = df_labeled_data.apply(lambda x: re.sub(' ', '', x['country1']), axis=1)
    df_labeled_data['country2'] = df_labeled_data.apply(lambda x: re.sub(' ', '', x['country2']), axis=1)
    df_labeled_data['chasing_country'] = df_labeled_data.apply(lambda x: re.sub(' ', '', x['chasing_country']), axis=1)
    df_labeled_data['batting_country_id'] = df_labeled_data.apply(
        lambda x: get_batting_country_id(x['country1'], x['country2'], x['chasing_country']), axis=1)
    df_labeled_data['ball_count'] = df_labeled_data.apply(lambda x: len(x['linked_balls'].split(',')), axis=1)
    df_labeled_data['entity_is_single_ball'] = df_labeled_data.apply(lambda x: 1 if x['ball_count'] == 1 else 0, axis=1)
    df_labeled_data['entity_type'] = df_labeled_data.apply(
        lambda x: 'Single Ball' if x['ball_count'] == 1 else 'Multi Balls', axis=1)

    cols1 = ['mention_key', 'article_number', 'mention_id', 'match_number', 'mention_text']
    cols_label = ['linked_balls', 'ball_count', 'entity_is_single_ball', 'entity_type', 'sub_entity',
                  'label_detailed', 'label_part1', 'label_part2', 'label_sub', 'label_runs',
                  'label_wickets_balls', 'label_rrr', 'label_powerplay_info'
                  ]
    cols_match = ['country1', 'country2', 'chasing_country', 'batting_country_id', 'won_by', 'win_margin',
                  'runs_c1', 'runs_c2', 'w_c1', 'w_c2', 'four_c1', 'four_c2', 'six_c1', 'six_c2',
                  'wide_c1', 'wide_c2']

    df_labeled_data = df_labeled_data[cols1 + cols_label + cols_match]
    df_entity_type_distribution = df_labeled_data.groupby(['entity_type']).aggregate(
        {'match_number': ['count']}).reset_index()
    df_entity_type_distribution.columns = ['entity_type', 'count']
    df_entity_type_distribution['percent'] = round(
        df_entity_type_distribution['count'] * 100 / df_entity_type_distribution['count'].sum(), 2)

    df_article_mention_distribution = df_labeled_data.groupby(['article_number']).agg(
        {'match_number': 'count'}).reset_index()
    df_article_mention_distribution.columns = ['article_number', 'number_of_mentions']

    df_articles = df_labeled_data[['article_number']].drop_duplicates()
    train_articles, test_articles = train_test_split(df_articles, test_size=0.2, random_state=42)
    train_article_ids = train_articles.article_number.values.tolist()
    test_articles_ids = test_articles.article_number.values.tolist()
    df_labeled_data['set'] = df_labeled_data.apply(
        lambda x: 'train' if x['article_number'] in train_article_ids else 'test', axis=1)
    df_labeled_data_split_entity_type_dist = df_labeled_data.groupby(['set', 'entity_is_single_ball']).aggregate(
        {'match_number': ['count']}).reset_index()
    df_labeled_data_split_entity_type_dist.columns = ['set', 'entity_is_single_ball', 'count']
    df_labeled_data_split_entity_type_dist_train = df_labeled_data_split_entity_type_dist[
        df_labeled_data_split_entity_type_dist['set'] == 'train'].copy()
    df_labeled_data_split_entity_type_dist_train['percent'] = round(
        df_labeled_data_split_entity_type_dist_train['count'] * 100 / df_labeled_data_split_entity_type_dist_train[
            'count'].sum(), 2)
    df_labeled_data_split_entity_type_dist_test = df_labeled_data_split_entity_type_dist[
        df_labeled_data_split_entity_type_dist['set'] == 'test'].copy()
    df_labeled_data_split_entity_type_dist_test['percent'] = round(
        df_labeled_data_split_entity_type_dist_test['count'] * 100 / df_labeled_data_split_entity_type_dist_test[
            'count'].sum(), 2)
    df_article_mention_distribution_train = df_labeled_data[df_labeled_data['set'] == 'train'].groupby(
        ['article_number']).agg({'match_number': 'count'}).reset_index()
    df_article_mention_distribution_train.columns = ['article_number', 'number_of_mentions']
    df_article_mention_distribution_test = df_labeled_data[df_labeled_data['set'] == 'test'].groupby(
        ['article_number']).agg({'match_number': 'count'}).reset_index()
    df_article_mention_distribution_test.columns = ['article_number', 'number_of_mentions']

    complete_data_avg_mentions = round(df_article_mention_distribution['number_of_mentions'].mean(), 2)
    complete_data_median_mentions = round(df_article_mention_distribution['number_of_mentions'].median(), 2)

    train_data_avg_mentions = round(df_article_mention_distribution_train['number_of_mentions'].mean(), 2)
    train_data_median_mentions = round(df_article_mention_distribution_train['number_of_mentions'].median(), 2)

    test_data_avg_mentions = round(df_article_mention_distribution_test['number_of_mentions'].mean(), 2)
    test_data_median_mentions = round(df_article_mention_distribution_test['number_of_mentions'].median(), 2)

    complete_data_single_ball_percent = df_entity_type_distribution.loc[1]['percent']
    training_data_single_ball_percent = df_labeled_data_split_entity_type_dist_train[
        df_labeled_data_split_entity_type_dist_train['entity_is_single_ball'] == 1].percent.values[0]
    test_data_single_ball_percent = df_labeled_data_split_entity_type_dist_test[
        df_labeled_data_split_entity_type_dist_test['entity_is_single_ball'] == 1].percent.values[0]

    data = {'Average of Mention Count': [
        complete_data_avg_mentions, train_data_avg_mentions, test_data_avg_mentions],
        'Median of Mention Count': [
            complete_data_median_mentions, train_data_median_mentions, test_data_median_mentions],
        '% Single Ball Entities': [
            complete_data_single_ball_percent, training_data_single_ball_percent, test_data_single_ball_percent]
    }
    df_complete_vs_train_vs_test_mention_dist = pd.DataFrame(data, index=['Complete Data', 'Train Data', 'Test Data'])
    train_data = df_labeled_data[df_labeled_data.set == 'train'].copy().reset_index(drop=True)
    test_data = df_labeled_data[df_labeled_data.set == 'test'].copy().reset_index(drop=True)
    data = {'# Mention Texts': [
        train_data.shape[0], test_data.shape[0]],
    }
    df_train_test_split_numbers = pd.DataFrame(data, index=['Train Data', 'Test Data'])

    df_train_test_split_numbers['% of Mention Texts'] = \
        round(df_train_test_split_numbers \
                  ['# Mention Texts'] * 100 / df_train_test_split_numbers['# Mention Texts'].sum(), 2)

    train_data_articles_entity_types = train_data.groupby(['article_number', 'entity_is_single_ball']). \
        aggregate({"match_number": 'count'}).reset_index()
    # train_data_articles_entity_types.colums=['article_number','entity_is_single_ball','count']
    train_data_articles_entity_types_pivot = pd.pivot_table(train_data, values='match_number', index=['article_number'],
                                                            columns=['entity_is_single_ball'],
                                                            aggfunc='count').reset_index()
    train_data_articles_entity_types_pivot.columns = ['article_number', 'count_multi_ball', 'count_single_ball']
    train_data_articles_entity_types_pivot.fillna(0, inplace=True)

    train_data_articles_entity_types_pivot['entity_total'] = train_data_articles_entity_types_pivot[
                                                                 'count_multi_ball'] + \
                                                             train_data_articles_entity_types_pivot['count_single_ball']
    train_data_articles_entity_types_pivot = train_data_articles_entity_types_pivot

    train_data_articles_entity_types_pivot['percent_multi_ball'] = train_data_articles_entity_types_pivot[
                                                                       'count_multi_ball'] * 100 / \
                                                                   train_data_articles_entity_types_pivot[
                                                                       'entity_total']
    train_data_articles_entity_types_pivot['percent_single_ball'] = train_data_articles_entity_types_pivot[
                                                                        'count_single_ball'] * 100 / \
                                                                    train_data_articles_entity_types_pivot[
                                                                        'entity_total']
    train_data_articles_entity_types_pivot.sort_values(by=['percent_multi_ball'], inplace=True)
    train_data_articles_entity_types_pivot = train_data_articles_entity_types_pivot.reset_index(drop=True).reset_index()
    train_data_articles_entity_types_pivot['article_s_number'] = train_data_articles_entity_types_pivot['index'] + 1
    train_data_articles_entity_types_pivot.drop(columns=['index'], inplace=True)
    col_order = ['article_s_number', 'article_number', 'count_multi_ball', 'count_single_ball',
                 'entity_total', 'percent_multi_ball', 'percent_single_ball']
    train_data_articles_entity_types_pivot = train_data_articles_entity_types_pivot[col_order]

    ####

    article_ids = train_article_ids + test_articles_ids
    df_match_articles_train_test = df_match_articles[df_match_articles['article_number'].isin(article_ids)].reset_index(
        drop=True).copy()

    df_match_articles_train_test['text_parse_obj'] = df_match_articles_train_test.apply(
        lambda x: parse_article(x['article_text']), axis=1)
    # df_match_articles_train_test['article_meta_text']=df_match_articles_train_test.apply(lambda x:x['text_parse_obj']['meta_text'],axis=1)
    df_match_articles_train_test['peras_count'] = df_match_articles_train_test.apply(
        lambda x: x['text_parse_obj']['peras_count'],
        axis=1)
    df_match_articles_train_test['line_count'] = df_match_articles_train_test.apply(
        lambda x: x['text_parse_obj']['line_count'],
        axis=1)
    df_match_articles_train_test['line_list'] = df_match_articles_train_test.apply(
        lambda x: x['text_parse_obj']['line_list'],
        axis=1)
    df_match_articles_train_test.drop(columns=['text_parse_obj'], inplace=True)
    df_match_articles_train_test_lines = df_match_articles_train_test[
        ['match_number', 'article_number', 'line_list']].explode(
        'line_list')
    df_match_articles_train_test_lines['pera_number'] = df_match_articles_train_test_lines.apply(
        lambda x: x['line_list'][0],
        axis=1)
    df_match_articles_train_test_lines['line_number'] = df_match_articles_train_test_lines.apply(
        lambda x: x['line_list'][1],
        axis=1)
    df_match_articles_train_test_lines['line_text'] = df_match_articles_train_test_lines.apply(
        lambda x: x['line_list'][2],
        axis=1)
    df_match_articles_train_test_lines['previous_line_text'] = df_match_articles_train_test_lines.apply(
        lambda x: x['line_list'][4], axis=1)
    df_match_articles_train_test_lines['mention_ids'] = df_match_articles_train_test_lines.apply(
        lambda x: x['line_list'][3],
        axis=1)
    df_match_articles_train_test_lines['mention_count'] = df_match_articles_train_test_lines.apply(
        lambda x: len(x['mention_ids']), axis=1)
    df_match_articles_train_test_lines.drop(columns=['line_list'], inplace=True)
    df_match_articles_train_test_lines = df_match_articles_train_test_lines.reset_index(drop=True)

    # parsed articles mentions
    df_match_articles_train_test_mentions = \
        df_match_articles_train_test_lines[df_match_articles_train_test_lines['mention_count'] > 0][
            ['match_number', 'article_number', 'pera_number', 'line_number', 'line_text', 'previous_line_text',
             'mention_ids']].explode('mention_ids').rename(columns={'mention_ids': 'mention_id'})
    df_match_articles_train_test_mentions['mention_key'] = df_match_articles_train_test_mentions.apply(
        lambda x: str(x['article_number']) + '_' + str(x['mention_id']), axis=1)
    df_match_articles_train_test_mentions = df_match_articles_train_test_mentions[
        ['match_number', 'article_number', 'pera_number', 'line_number', 'mention_id', 'mention_key', 'line_text',
         'previous_line_text']]

    mention_dict = defaultdict(def_value)
    for k, pera_number, line_number, line_text, previous_line_text in df_match_articles_train_test_mentions[
        ['mention_key', 'pera_number', 'line_number', 'line_text', 'previous_line_text']].values.tolist():
        out_dict_pera_line = {}
        out_dict_pera_line['pera_number'] = pera_number
        out_dict_pera_line['line_number'] = line_number
        out_dict_pera_line['line_text'] = line_text
        out_dict_pera_line['previous_line_text'] = previous_line_text
        mention_dict[k] = out_dict_pera_line

    ####

    out_dict = {}
    out_dict['df_labeled_data'] = df_labeled_data
    out_dict['df_article_mention_distribution'] = df_article_mention_distribution
    out_dict['df_articles'] = df_articles
    out_dict['train_article_ids'] = train_article_ids
    out_dict['test_articles_ids'] = test_articles_ids
    out_dict['df_labeled_data_split_entity_type_dist'] = df_labeled_data_split_entity_type_dist
    out_dict['df_article_mention_distribution_train'] = df_article_mention_distribution_train
    out_dict['df_article_mention_distribution_test'] = df_article_mention_distribution_test
    out_dict['df_complete_vs_train_vs_test_mention_dist'] = df_complete_vs_train_vs_test_mention_dist
    out_dict['df_match_articles_train_test_mentions'] =df_match_articles_train_test_mentions

    out_dict['train_data'] = train_data
    out_dict['test_data'] = test_data
    out_dict['df_train_test_split_numbers'] = df_train_test_split_numbers
    out_dict['train_data_articles_entity_types'] = train_data_articles_entity_types
    out_dict['train_data_articles_entity_types_pivot'] = train_data_articles_entity_types_pivot
    # out_dict['df_match_articles_train'] = df_match_articles_train
    # out_dict['df_match_articles_train_lines'] = df_match_articles_train_lines
    # out_dict['df_match_articles_train_mentions'] = df_match_articles_train_mentions
    out_dict['mention_dict'] = mention_dict
    return out_dict

pattern_replacement=[]
pattern_replacement.append(['asking-rate','REQUIREDRATE'])
pattern_replacement.append(['ASKINGRATE','REQUIREDRATE'])
pattern_replacement.append(['asking rate','REQUIREDRATE'])
pattern_replacement.append(['required-rate','REQUIREDRATE'])
pattern_replacement.append(['required rate','REQUIREDRATE'])
pattern_replacement.append(['half-century','HALFCENTURY'])
pattern_replacement.append(['half century','HALFCENTURY'])
pattern_replacement.append(['half-centuries','HALFCENTURIES'])
pattern_replacement.append(['half centuries','HALFCENTURIES'])
pattern_replacement.append(['run-out','RUNOUT'])
pattern_replacement.append(['run out','RUNOUT'])
pattern_replacement.append(['all-out','ALLOUT'])
pattern_replacement.append(['all out','ALLOUT'])
pattern_replacement.append(['bowled out','ALLOUT'])
pattern_replacement.append(['stepped out','STEPPEDOUT'])
pattern_replacement.append(['came out','CAMEOUT'])
pattern_replacement.append(['not out','NOTOUT'])
pattern_replacement.append(['not-out','NOTOUT'])
pattern_replacement.append(['played out','PLAYEDOUT'])
pattern_replacement.append(['pulling out','PULLINGOUT'])
pattern_replacement.append(['batted out','BATTEDOUT'])
pattern_replacement.append(['spilled out','SPILLEDOUT'])
pattern_replacement.append(['walked out','WALKEDOUT'])
pattern_replacement.append(['get out','GETOUT'])
pattern_replacement.append(['lash out','LASHOUT'])
pattern_replacement.append(['leave out','LEAVEOUT'])
pattern_replacement.append(['take out','TAKEOUT'])
pattern_replacement.append(['miles out','MILESOUT'])
pattern_replacement.append(['wrists out','WRISTSSOUT'])
pattern_replacement.append(['shot out','SHOTSOUT'])
pattern_replacement.append(['deliveries','BALLS'])
pattern_replacement.append(['victory','WIN'])
pattern_replacement.append(['win','WIN'])
pattern_replacement.append(['winning','WINNING'])
pattern_replacement.append(['won','WON'])
pattern_replacement.append(['defeat','DEFEAT'])
pattern_replacement.append(['defeated','DEFEATED'])
pattern_replacement.append(['loss to','LOSSTO'])
pattern_replacement.append(['lost','LOST'])
pattern_replacement.append(['tie','TIE'])
pattern_replacement.append(['tied','TIED'])
pattern_replacement.append(['triumph','TRIUMPH'])
pattern_replacement.append(['beating','BEATING'])
pattern_replacement.append(['upset','UPSET'])
pattern_replacement.append(['chasing down','CHASINGDOWN'])
pattern_replacement.append(['chase','CHASE'])
pattern_replacement.append(['thrashed','THRASHED'])
pattern_replacement.append(['wallop','WOLLOP'])
pattern_replacement.append(['result','WOLLOP'])
pattern_replacement.append(['New Zealand','NewZealand'])
pattern_replacement.append(['Sri Lanka','SriLanka'])
pattern_replacement.append(['West Indies','WestIndies'])
pattern_replacement.append(['South Africa','SouthAfrica'])

def norm_mention_text(mention_text):
    for pattern, replacement in pattern_replacement:
        PAT_KW='(?i)(?:^|[^a-z])({})(?:[^a-z]|$)'.format(pattern)
        PAT_KW_search = re.search(PAT_KW, mention_text)
        if PAT_KW_search:
            mention_text=re.sub(pattern,replacement,mention_text,flags=re.IGNORECASE)
    try:
        mention_text=t2d.convert(mention_text)
        mention_text=re.sub('(?i)Ryan 10 Doeschate','Ryan Ten Doeschate',mention_text)
        return mention_text
    except:
        return mention_text
         
    
#norm_mention_text("the second got four miss, off the bowling of Abdur Rehman again, was a true Akmal clanger; Sangakkara was miles out, the ball wasn't even that far down the leg-side and Akmal knocked the bails off swiftly enough. He just didn't have the ball with him.")

## Model Specific
def verdict_type(label,pred):
    if label==1 and pred==1:
        verdict='TP'
    elif label==0 and pred==0:
        verdict='TN'
    elif label==1 and pred==0:
        verdict='FN'        
    elif label==0 and pred==1:
        verdict='FP'
    else:
        verdict='UN'
    return verdict

def get_model_score(train_data1, y_train_pred_prob, y_train):
    start_time = time.time()
    train_data1['prediction_proab'] = y_train_pred_prob[:,1] 
    train_data1['prediction'] = train_data1.apply(lambda x: 1 if x['prediction_proab']>0.5 else 0,axis=1)
    train_data1['verdict']=train_data1.apply(lambda x: verdict_type(x['entity_is_single_ball'],x['prediction']),axis=1)

    print("\033[1m\n5 Fold Cross Validation Results\033[0m \n")
    _ = sns.heatmap(confusion_matrix(y_train, train_data1['prediction']),annot=True,fmt='0.0f',cbar=False)
    print(classification_report(y_train,train_data1['prediction']))
    print_full(train_data1.verdict.value_counts().reset_index().rename(columns={'index':'verdict','verdict':'count'}))
    end_time = time.time()
    print('\033[1m','\nTime Taken : \033[0m', end_time - start_time, '\n')

def get_player_id(part2,country1,country2,dict_players_name,dict_players_name_token):
    player_c1_lookup_name=dict_players_name.get(part2.lower()+'__'+country1.lower(),None)
    player_c2_lookup_name=dict_players_name.get(part2.lower()+'__'+country2.lower(),None)
    player_c1_lookup_token=dict_players_name_token.get(part2+'__'+country1.lower(),None)
    player_c2_lookup_token=dict_players_name_token.get(part2+'__'+country2.lower(),None)
    if player_c1_lookup_name is not None:
        part2=player_c1_lookup_name
    elif player_c2_lookup_name is not None:
        part2=player_c2_lookup_name
    elif player_c1_lookup_token is not None:
        part2=player_c1_lookup_token
    elif player_c2_lookup_token is not None:
        part2=player_c2_lookup_token
    return part2
# part2,country1,country2='Sehwag','India','Bangladesh'
# part2,country1,country2='Kohli','India','Bangladesh'
# get_player_id(part2,country1,country2)

def get_batting_country_id(country1, country2,chasing_country):
    if country1==chasing_country:
        return 'c2'
    else:
        return 'c1'

def parse_label_detailed(label_detailed,country1,country2,dict_players_name,dict_players_name_token):
    part1=None
    part2=None
    sub=None
    runs=None
    wickets_balls=None
    rrr=None
    powerplay_info=None
    
    label_detailed=re.sub(' _',' ',label_detailed).strip()
    label_detailed_list=label_detailed.split('(')
    part1=label_detailed_list[0]
    
    if ') ' in label_detailed or 'UNION' in label_detailed or len(label_detailed_list)>2:
        part1='#NOTRESOLVED'
    elif len(label_detailed_list)>1:
        part2=label_detailed_list[1]
        part2=re.sub('\)$','',part2)
        if part2==country1:
            part2='country1'
        elif part2==country2:
            part2='country2'
        else:
            part2=get_player_id(part2,country1,country2,dict_players_name,dict_players_name_token)
        if part2 in ['country1','country2','INNINGS1','INNINGS2'] or re.search('(?i)[a-z]', part2) is None:
            sub=part2
    
    if part1 in ['PARTNERSHIP','SINGLES AND PARTNERSHIP']:
        part2_list=part2.split(',')
        part2_list=[i.strip() for i in part2_list]
        part2_ids=[]
        for i in part2_list:
            pid=get_player_id(i,country1,country2,dict_players_name,dict_players_name_token)
            if pid is not None:
                part2_ids.append(pid)
        if len(part2_ids)==2:
            part2='|'.join(part2_ids)
        sub=part2
    elif part1 in ['BALL']:
        part2=re.sub(',$','',re.sub(', ',',',part2))
        part2_list=part2.strip().split(',')
        if len(part2_list)==4:
            sub=part2_list[0]
            if sub==country1:
                sub='country1'
            elif sub==country2:
                sub='country2'
            else:
                sub=get_player_id(sub,country1,country2,dict_players_name,dict_players_name_token)
            runs=part2_list[2]
            wickets_balls=part2_list[3]
        elif len(part2_list)==3:
            sub=part2_list[0]
            rrr=part2_list[2]
    elif part1 in ['POWERPLAY']:
        powerplay_info=re.sub(', ',',',part2)
    out_dict={}
    out_dict['part1']=part1
    out_dict['part2']=part2
    out_dict['sub']=sub
    out_dict['runs']=runs
    out_dict['wickets_balls']=wickets_balls
    out_dict['rrr']=rrr
    out_dict['powerplay_info']=powerplay_info
    return out_dict
# i=23
# country1=df_labeled_data.iloc[i]['country1']
# country2=df_labeled_data.iloc[i]['country2']
# label_detailed=df_labeled_data.iloc[i]['label_detailed']
# parse_label_detailed(label_detailed,country1,country2)






def get_nlp_tags_old(input_text,country1,country2,dict_players_name,dict_players_name_token,dict_player_names):
    input_text_doc = nlp(input_text)
    token_list=[]
    token_dep_list=[]
    token_pos_list=[]
    subject_cands=[]
    subjects=[]
    subject_players=[]
    subject_players_names=[]
    nsubj_count=0
    for token in input_text_doc:
        token_list.append(token.text)
        token_dep_list.append(token.dep_)
        token_pos_list.append(token.pos_)
        if token.dep_=='nsubj':
            nsubj_count=nsubj_count+1
    subject_cands=get_chunks1(input_text_doc,kw=None)
    
    if (token_list[0]==country1 or token_list[0]==country2) \
        and token_dep_list[0]=='ROOT' and token_list[0] not in subject_cands:
        subject_cands.append(token_list[0])
    else:
        if(len(token_list)>=2):
            look_up_token=' '.join([token_list[0],token_list[1]])
            look_up_dep=' '.join([token_dep_list[0],token_dep_list[1]])
            if look_up_token==country1 or look_up_token==country2:
                if look_up_dep in ['compound ROOT','compound POSS'] and look_up_token not in subject_cands:
                    subject_cands.append(look_up_token)
            elif token_dep_list[0]=='ROOT':

                lookup_token1=token_list[0]+'__'+country1.lower()
                lookup_token2=token_list[0]+'__'+country2.lower()

                lookup_token1_val=dict_players_name_token.get(lookup_token1,None)
                lookup_token2_val=dict_players_name_token.get(lookup_token2,None)


                if lookup_token1_val is not None:
                    subject_cands.append(token_list[0])
                elif lookup_token2_val is not None:
                    subject_cands.append(token_list[0])                
                
    if token_list[0].lower() in ['his'] and token_dep_list[0]=='poss':
        subject_cands.append(token_list[0])

                 
    for subject_cand in subject_cands:
        if subject_cand.lower() in ['he','they','his','my']:
            subjects.append(subject_cand)
        elif subject_cand in ['I']:
            subjects.append(subject_cand)
            
        elif subject_cand==country1:
            subjects.append('c1')
        elif subject_cand==country2:
            subjects.append('c2')
        else:
            if ' ' in subject_cand:
                lookup1=subject_cand.lower()+'__'+country1.lower()
                lookup2=subject_cand.lower()+'__'+country2.lower()
                lookup1_val=dict_players_name.get(lookup1,None)
                lookup2_val=dict_players_name.get(lookup2,None)
                if lookup1_val is not None:
                    subjects.append(lookup1_val)
                elif lookup2_val is not None:
                    subjects.append(lookup2_val)
            else:
                lookup1=subject_cand+'__'+country1.lower()
                lookup2=subject_cand+'__'+country2.lower()
                lookup1_val=dict_players_name_token.get(lookup1,None)
                lookup2_val=dict_players_name_token.get(lookup2,None)
                if lookup1_val is not None:
                    subjects.append(lookup1_val)
                elif lookup2_val is not None:
                    subjects.append(lookup2_val)

    subjects=list(set(subjects))
    for subject in subjects:
        name_val=dict_player_names.get(subject)
        if name_val is not None:
            subject_players.append(subject)
            subject_players_names.append(name_val)
    out_dict={}
    out_dict['subjects']=subjects
    out_dict['subject_players']=subject_players
    out_dict['subject_players_names']=subject_players_names
    out_dict['subject_cands']=subject_cands
    out_dict['token_list']=token_list
    out_dict['token_dep_list']=token_dep_list
    out_dict['token_pos_list']=token_pos_list
    out_dict['nsubj_count']=nsubj_count
    out_dict['subjects_count']=len(subjects)
    return out_dict

# i=2
# input_text=train_data1.iloc[i]['canonical_mention_text']
# country1=train_data1.iloc[i]['country1']
# country2=train_data1.iloc[i]['country2']
# input_text
# get_nlp_tags_old(input_text,country1,country2,dict_players_name,dict_players_name_token,dict_player_names)

