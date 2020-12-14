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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

from text2digits import text2digits
t2d = text2digits.Text2Digits(convert_ordinals=False)
t2d1 = text2digits.Text2Digits()


from udfs_preprocessing import norm_mention_text, country_player_identification, clean_br

def get_chunks(doc, kw):
    matcher = Matcher(nlp.vocab)
    # pattern=[{'DEP': 'amod','OP': '?'},{'DEP': 'nummod','OP': '?'},{'TEXT': kw}]
    pattern = [{'DEP': 'amod', 'OP': '?'}, {'DEP': 'nummod', 'OP': '?'}, {'DEP': 'compound', 'OP': '?'}, {'LOWER': kw}]
    matcher.add('pat1', None, pattern)

    pattern = [{'DEP': 'nummod', 'OP': '?'}, {'DEP': 'amod', 'OP': '?'}, {'DEP': 'compound', 'OP': '?'}, {'LOWER': kw}]
    matcher.add('pat2', None, pattern)

    pattern = [{'DEP': 'pobj', 'OP': '?'}, {'LOWER': kw}]
    matcher.add('pat3', None, pattern)

    matches = matcher(doc)
    # print('Total matches found:', len(matches))
    match_list = []
    for match_id, start, end in matches:
        match_list.append(doc[start:end].text)
    if len(match_list) > 0:
        return sorted(match_list, key=len, reverse=True)[0]
    else:
        return ''


# #testcase
# i=130
# doc=train_data1.iloc[i]['doc']
# doc=nlp('his first World Cup hundred from 98 balls')
# doc=nlp('His match-winning century')
# print(doc)
# print(train_data1.iloc[i]['tags'])
# print(get_chunks(doc,'over'))



def get_additional_country_players(mention_text, article_mention_line,
                                   article_mention_previous_line, country1, country2, dict_players_name,
                                   dict_players_name_token):
    mention_dict = country_player_identification(mention_text, country1, country2, dict_players_name,
                                                 dict_players_name_token)
    article_mention_line_dict = country_player_identification(article_mention_line, country1, country2,
                                                              dict_players_name, dict_players_name_token)
    article_mention_previous_line_dict = country_player_identification(article_mention_previous_line,
                                                                       country1, country2,
                                                                       dict_players_name, dict_players_name_token)
    out_dict = {}
    out_dict['mention_text'] = mention_dict['mention_text']
    out_dict['contains_country1'] = mention_dict['contains_country1']
    out_dict['contains_country2'] = mention_dict['contains_country2']
    out_dict['count_country'] = mention_dict['count_country']
    out_dict['contains_country1_player'] = mention_dict['contains_country1_player']
    out_dict['contains_country2_player'] = mention_dict['contains_country2_player']
    out_dict['c1_player_ids'] = mention_dict['c1_player_ids']
    out_dict['c2_player_ids'] = mention_dict['c2_player_ids']
    out_dict['count_players_country'] = mention_dict['count_players_country']
    out_dict['count_players'] = mention_dict['count_players']
    out_dict['maching_name_grams'] = mention_dict['maching_name_grams']

    additional_contains_country1 = 0
    additional_contains_country2 = 0
    additional_c1_player_ids = []
    additional_c2_player_ids = []

    if mention_dict['contains_country1'] == 0:
        if article_mention_line_dict['contains_country1'] == 1:
            additional_contains_country1 = 1
        else:
            additional_contains_country1 = article_mention_previous_line_dict['contains_country1']
    if mention_dict['contains_country2'] == 0:
        if article_mention_line_dict['contains_country2'] == 1:
            additional_contains_country2 = 1
        else:
            additional_contains_country2 = article_mention_previous_line_dict['contains_country2']

    out_dict['additional_contains_country1'] = additional_contains_country1
    out_dict['additional_contains_country2'] = additional_contains_country2
    out_dict['additional_count_country'] = additional_contains_country1 + additional_contains_country2

    if mention_dict['c1_player_ids'] == []:
        if len(article_mention_line_dict['c1_player_ids']) > 0:
            additional_c1_player_ids = article_mention_line_dict['c1_player_ids']
        else:
            additional_c1_player_ids = article_mention_previous_line_dict['c1_player_ids']
    if mention_dict['c2_player_ids'] == []:
        if len(article_mention_line_dict['c2_player_ids']) > 0:
            additional_c2_player_ids = article_mention_line_dict['c2_player_ids']
        else:
            additional_c2_player_ids = article_mention_previous_line_dict['c2_player_ids']
    if len(additional_c1_player_ids) > 0:
        out_dict['additional_contains_country1_player'] = 1
    else:
        out_dict['additional_contains_country1_player'] = 0
    if len(additional_c2_player_ids) > 0:
        out_dict['additional_contains_country2_player'] = 1
    else:
        out_dict['additional_contains_country2_player'] = 0
    out_dict['additional_c1_player_ids'] = additional_c1_player_ids
    out_dict['additional_c2_player_ids'] = additional_c2_player_ids
    out_dict['additional_count_players_country'] = out_dict['additional_contains_country1_player'] + out_dict[
        'additional_contains_country2_player']
    out_dict['additional_count_players'] = len(additional_c1_player_ids) + len(additional_c2_player_ids)
    return out_dict
# #testcase
# i=137
# i=1873
# mention_text=train_data1.iloc[i]['canonical_mention_text']
# article_mention_line=train_data1.iloc[i]['article_mention_line']
# article_mention_previous_line=train_data1.iloc[i]['article_mention_previous_line']
# country1=train_data1.iloc[i]['country1']
# country2=train_data1.iloc[i]['country2']
# print('Mention Text: %s'%mention_text)
# print('article_mention_line: %s'%article_mention_line)
# print('article_mention_previous_line: %s'%article_mention_previous_line)
# print('Country1: %s'%country1)
# print('Country2: %s'%country2)
# get_additional_country_players(mention_text,article_mention_line,article_mention_previous_line,country1,country2)


def choose_out_of2(f1, f2):
    if f1 is not None and f1 != '':
        return f1
    else:
        return f2

def entity_tagging(mention_text, country1, country2, doc, c1_player_ids, c2_player_ids, ent_tagg_taxonomy):
    [list_shot_type, list_ball_type, list_fielding_position, list_keywords_filtered] = ent_tagg_taxonomy
    contains_shot_kw = contains_ball_kw = contains_fieldpostion_kw = contains_other_kw = 0
    contains_score = 0
    contains_country_score = 0
    contains_player_score = 0
    score = None
    contains_RS = 0
    contains_4 = contains_6 = contains_R = contains_W = contains_OV = contains_B = contains_O = 0
    contains_PP = contains_PS = contains_SP = 0
    contains_other_mb_kw = contains_other_sb_kw = contains_conclusion_kw = 0
    # taxonomy
    PAT_shots = '(?i)(?:^|[^a-z])({})(?:[^a-z]|$)'.format('|'.join(list_shot_type))
    PAT_balls_kw = '(?i)(?:^|[^a-z])({})(?:[^a-z]|$)'.format('|'.join(list_ball_type))
    PAT_fieldposition_kw = '(?i)(?:^|[^a-z])({})(?:[^a-z]|$)'.format('|'.join(list_fielding_position))
    PAT_other_kw = '(?i)(?:^|[^a-z])({})(?:[^a-z]|$)'.format('|'.join(list_keywords_filtered))
    other_mb_keyswords = ['referrals', 'pair', 'stand', 'boundaries', 'HALFCENTURIES', 'CENTURIES',
                          'singles', 'figures', 'bowlers', 'openers', 'extras', 'yorkers', 'three']
    other_sb_keyswords = ['catch', 'delivery', 'edge', 'defeat', 'lbw', 'replays', 'yorker',
                          'umpire', 'stump', 'ALLOUT', 'RUNOUT', 'REQUIREDRATE', 'HALFCENTURY', 'CENTURY']
    conclusion_sb_keyswords = ['WIN', 'WINNING', 'WON', 'DEFEAT', 'DEFEATED', 'LOSSTO', 'LOSS', 'TIED', 'TIE',
                               'TRIUMPH',
                               'BEATING', 'UPSET', 'CHASINGDOWN', 'CHASE', 'THRASHED', 'WOLLOP', 'WOLLOP']

    PAT_other_mb_kw = '(?i)(?:^|[^a-z])({})(?:[^a-z]|$)'.format('|'.join(other_mb_keyswords))
    PAT_other_sb_kw = '(?i)(?:^|[^a-z])({})(?:[^a-z]|$)'.format('|'.join(other_sb_keyswords))
    PAT_conclusion_sb_kw = '(?i)(?:^|[^a-z])({})(?:[^a-z]|$)'.format('|'.join(conclusion_sb_keyswords))
    PAT_run_w_neg = '(?i)(?:overs) ([0-9]{2,3})(?:-| for )([0-9]{2,3})'
    PAT_run_w = '(?i)(?:^| )([0-9]{1,3})(?:-| for )([0-9]{1,2})'
    PAT_country_score = '<<C(1|2)>>(?:\'s)?(?: | on )<<SCORE>>'
    PAT_player_score = '<<C(?:1|2)P([0-9_]*)>>(?:\'s)? <<SCORE>>'
    PAT_4S_search = '(?i)(?:^|[^a-z])(fours|boundaries)(?:$|[^a-z])'
    PAT_6S_search = '(?i)(?:^|[^a-z])(sixes)(?:$|[^a-z])'
    PAT_RS_search = '(?i)(?:^|[^a-z])(runs)(?:$|[^a-z])'
    PAT_WS_search = '(?i)(?:^|[^a-z])(wickets)(?:$|[^a-z])'
    PAT_BS_search = '(?i)(?:^|[^a-z])(balls|deliveries)(?:[^a-z]|$)'
    PAT_4_search = '(?i)(?:^|[^a-z])(four|boundary)(?:$|[^a-z])'
    PAT_6_search = '(?i)(?:^|[^a-z])(six)(?:$|[^a-z])'
    PAT_R_search = '(?i)(?:^|[^a-z])(run)(?:$|[^a-z])'
    PAT_W_search = '(?i)(?:^|[^a-z])(wicket)(?:$|[^a-z])'
    PAT_OV_search = '(?i)([^a-z0-9]|^)(over)([^a-z0-9]|$)'
    PAT_B_search = '(?i)(?:[^a-z]|$)(ball|delivery)(?:[^a-z]|$)'
    PAT_O_search = '(?i)([^a-z0-9]|^)(out|dismissed|dismissal|RUNOUT|ALLOUT)([^a-z0-9]|$)'
    PAT_PP_search = '(?i)(?:^|[^a-z])(powerplay(?:s)?)(?:$|[^a-z])'
    PAT_PS_search = '(?i)(?:^|[^a-z])(partnership(?:s)?)(?:$|[^a-z])'
    PAT_SP_search = '(?i)(?:^|[^a-z])(spell(?:s)?)(?:$|[^a-z])'
    ents = ['balls', 'overs', 'over', 'fours', 'sixes', 'powerplay', 'wickets', 'wicket', 'victims']
    out_dict = {}
    for ent in ents:
        ent_info = get_chunks(doc, ent)
        if len(ent_info) > 0:
            # ent_dict[ent]={'contains':1,'info':out_dict}
            out_dict['contains_' + ent] = 1
            out_dict['info_' + ent] = ent_info
            mention_text = re.sub(ent_info, ent.upper(), mention_text)
        else:
            out_dict['contains_' + ent] = 0
            out_dict['info_' + ent] = ''
    PAT_shots_search = re.search(PAT_shots, mention_text)
    if PAT_shots_search:
        contains_shot_kw = 1
        pattern = PAT_shots_search.group(1)
        replacement = re.sub(' ', '', pattern.upper())
        mention_text = re.sub(pattern, replacement, mention_text)
    PAT_balls_kw_search = re.search(PAT_balls_kw, mention_text)
    if PAT_balls_kw_search:
        contains_ball_kw = 1
        pattern = PAT_balls_kw_search.group(1)
        replacement = re.sub(' ', '', pattern.upper())
        mention_text = re.sub(pattern, replacement, mention_text)
    PAT_fieldposition_kw_search = re.search(PAT_fieldposition_kw, mention_text)
    if PAT_fieldposition_kw_search:
        contains_fieldpostion_kw = 1
        pattern = PAT_fieldposition_kw_search.group(1)
        replacement = re.sub(' ', '', pattern.upper())
        mention_text = re.sub(pattern, replacement, mention_text)
    PAT_other_kw_search = re.search(PAT_other_kw, mention_text)
    if PAT_other_kw_search:
        contains_other_kw = 1
        pattern = PAT_other_kw_search.group(1)
        replacement = re.sub(' ', '', pattern.upper())
        mention_text = re.sub(pattern, replacement, mention_text)
    PAT_other_mb_kw_search = re.search(PAT_other_mb_kw, mention_text)
    if PAT_other_mb_kw_search:
        contains_other_mb_kw = 1
        pattern = PAT_other_mb_kw_search.group(1)
        replacement = re.sub(' ', '', pattern.upper())
        mention_text = re.sub(pattern, replacement, mention_text)
    PAT_other_sb_kw_search = re.search(PAT_other_sb_kw, mention_text)
    if PAT_other_sb_kw_search:
        contains_other_sb_kw = 1
        pattern = PAT_other_sb_kw_search.group(1)
        replacement = re.sub(' ', '', pattern.upper())
        mention_text = re.sub(pattern, replacement, mention_text)
    PAT_conclusion_sb_kw_search = re.search(PAT_conclusion_sb_kw, mention_text)
    if PAT_conclusion_sb_kw_search:
        contains_conclusion_kw = 1
        # pattern=PAT_other_sb_kw_search.group(1)
        # replacement=re.sub(' ','',pattern.upper())
        # mention_text=re.sub(pattern,replacement,mention_text)
    run_w_neg_search = re.search(PAT_run_w_neg, mention_text)
    run_w_search = re.search(PAT_run_w, mention_text)
    if run_w_search and not run_w_neg_search:
        contains_score = 1
        runs = run_w_search.group(1)
        wickets = run_w_search.group(2)
        score = runs + '__' + wickets
        mention_text = re.sub(' +', ' ', re.sub(PAT_run_w, ' <<SCORE>> ', mention_text)).strip()
    mention_text1 = tagged_text_format(mention_text, c1_player_ids, c2_player_ids)
    PAT_bowler_score = '\(([0-9]{1,2}(?:\.[0-9])?)-([0-9])-([0-9]{1,2})-([0-9])\)'
    bowler_score_search = re.search(PAT_bowler_score, mention_text)
    if bowler_score_search:
        contains_score = 1
        runs = bowler_score_search.group(3)
        wickets = bowler_score_search.group(4)
        overs = bowler_score_search.group(1)
        score = runs + '__' + wickets + '__' + overs
        mention_text = re.sub(' +', ' ', re.sub(PAT_bowler_score, ' <<SCORE>> ', mention_text)).strip()
    mention_text1 = tagged_text_format(mention_text, c1_player_ids, c2_player_ids)
    PAT_country_score_search = re.search(PAT_country_score, mention_text)
    if PAT_country_score_search:
        contains_score = 0
        contains_country_score = 1
        mention_text = re.sub(PAT_country_score, '<<C\\1SCORE>>', mention_text)

    PAT_player_score_search = re.search(PAT_player_score, mention_text)
    if PAT_player_score_search:
        contains_score = 0
        contains_player_score = 1
        mention_text = re.sub(PAT_player_score, '<<P\\1SCORE>>', mention_text)

    if re.search(PAT_4S_search, mention_text):
        contains_4S = 1
        mention_text = re.sub(PAT_4S_search, ' <<4S>> ', mention_text)
    if re.search(PAT_4_search, mention_text):
        contains_4 = 1
        mention_text = re.sub(PAT_4_search, ' <<4>> ', mention_text)
    if re.search(PAT_6S_search, mention_text):
        contains_6S = 1
        mention_text = re.sub(PAT_6S_search, ' <<6S>> ', mention_text)
    if re.search(PAT_6_search, mention_text):
        contains_6 = 1
        mention_text = re.sub(PAT_6_search, ' <<6>> ', mention_text)
    if re.search(PAT_RS_search, mention_text):
        contains_RS = 1
        mention_text = re.sub(PAT_RS_search, ' <<RS>> ', mention_text)
    if re.search(PAT_R_search, mention_text):
        contains_R = 1
        mention_text = re.sub(PAT_R_search, ' <<R>> ', mention_text)
    if re.search(PAT_WS_search, mention_text):
        contains_WS = 1
        mention_text = re.sub(PAT_WS_search, ' <<WS>> ', mention_text)
    if re.search(PAT_W_search, mention_text):
        contains_W = 1
        mention_text = re.sub(PAT_W_search, ' <<w>> ', mention_text)

    if re.search(PAT_OV_search, mention_text):
        contains_OV = 1
        mention_text = re.sub(PAT_OV_search, ' <<OV>> ', mention_text)

    if re.search(PAT_BS_search, mention_text):
        contains_BS = 1
        mention_text = re.sub(PAT_BS_search, ' <<BS>> ', mention_text)
    if re.search(PAT_B_search, mention_text):
        contains_B = 1
        mention_text = re.sub(PAT_B_search, ' <<B>> ', mention_text)

    if re.search(PAT_O_search, mention_text):
        contains_O = 1
        mention_text = re.sub(' +', ' ', re.sub(PAT_O_search, ' <<O>> ', mention_text)).strip()

    if re.search(PAT_PP_search, mention_text):
        contains_PP = 1
        mention_text = re.sub(' +', ' ', re.sub(PAT_PP_search, ' <<PP>> ', mention_text)).strip()

    if re.search(PAT_PS_search, mention_text):
        contains_PS = 1
        mention_text = re.sub(' +', ' ', re.sub(PAT_PS_search, ' <<PS>> ', mention_text)).strip()
    if re.search(PAT_SP_search, mention_text):
        contains_SP = 1
        mention_text = re.sub(' +', ' ', re.sub(PAT_SP_search, ' <<SP>> ', mention_text)).strip()

    mention_text = re.sub(' +', ' ', mention_text).strip()
    out_dict['tagged_mention_text1'] = mention_text
    out_dict['tagged_mention_text2'] = mention_text1
    out_dict['contains_score'] = contains_score
    out_dict['contains_country_score'] = contains_country_score
    out_dict['contains_player_score'] = contains_player_score
    out_dict['score'] = score
    out_dict['contains_shot_kw'] = contains_shot_kw
    out_dict['contains_ball_kw'] = contains_ball_kw
    out_dict['contains_fieldpostion_kw'] = contains_fieldpostion_kw
    out_dict['contains_other_kw'] = contains_other_kw
    out_dict['contains_other_mb_kw'] = contains_other_mb_kw
    out_dict['contains_other_sb_kw'] = contains_other_sb_kw
    out_dict['contains_conclusion_kw'] = contains_conclusion_kw
    out_dict['contains_RS'] = contains_RS
    out_dict['contains_4'] = contains_4
    out_dict['contains_6'] = contains_6
    out_dict['contains_R'] = contains_R
    out_dict['contains_W'] = contains_W
    out_dict['contains_OV'] = contains_OV
    out_dict['contains_B'] = contains_B
    out_dict['contains_O'] = contains_O
    out_dict['contains_PS'] = contains_PS
    out_dict['contains_SP'] = contains_SP
    return out_dict
# testcase
# i=33
# i=1874
# i = 298
#
# doc = train_data1.iloc[i]['doc']
# mention_text = train_data1.iloc[i]['mention_text']
# canonical_mention_text = train_data1.iloc[i]['canonical_mention_text']
# tagged_mention_text = train_data1.iloc[i]['tagged_mention_text']
# c1_player_ids = train_data1.iloc[i]['c1_player_ids']
# c2_player_ids = train_data1.iloc[i]['c2_player_ids']
# country1 = train_data1.iloc[i]['country1']
# country2 = train_data1.iloc[i]['country2']
# print('Mention Text: %s' % mention_text)
# print('Country1: %s' % country1)
# print('Country2: %s' % country2)
# out_dict = entity_tagging(tagged_mention_text, country1, country2, doc, c1_player_ids, c2_player_ids, ent_tagg_taxonomy)
# print('Tagged Mention Text: %s' % tagged_mention_text)
# print('Tagged Mention Text1: %s' % out_dict['tagged_mention_text1'])
# print('Tagged Mention Text2: %s' % out_dict['tagged_mention_text2'])
# print('info_wickets: %s' % out_dict['info_wicket'])
# print('Contains Score: %s' % out_dict['contains_score'])
# print('Contains Country Score: %s' % out_dict['contains_country_score'])
# print('Contains Player Score: %s' % out_dict['contains_player_score'])
# print('Contains Shot KW: %s' % out_dict['contains_shot_kw'])
# print('Contains Ball KW: %s' % out_dict['contains_ball_kw'])
# print('Contains FieldPosition KW: %s' % out_dict['contains_fieldpostion_kw'])
# print('Contains Other KW: %s' % out_dict['contains_other_kw'])
# print('Contains Other MB KW: %s' % out_dict['contains_other_mb_kw'])
# print('Contains Other SB KW: %s' % out_dict['contains_other_sb_kw'])
# print('Score: %s' % out_dict['score'])
# print('Contains balls: %s' % out_dict['contains_balls'])
# print('Contains overs: %s' % out_dict['contains_overs'])
# print('Contains 4S: %s' % out_dict['contains_fours'])
# print('Contains 6S: %s' % out_dict['contains_sixes'])
# print('Contains powerplay: %s' % out_dict['contains_powerplay'])
# print('Contains WS: %s' % out_dict['contains_wickets'])
# print('Contains WS: %s' % out_dict['contains_victims'])
# print('Contains RS: %s' % out_dict['contains_RS'])
# print('Balls Info: %s' % out_dict['info_balls'])
# print('Overs Info: %s' % out_dict['info_overs'])
# print('Fours Info: %s' % out_dict['info_fours'])
# print('Sixes Info: %s' % out_dict['info_sixes'])
# print('Powerplay Info: %s' % out_dict['info_powerplay'])
# print('Wickets Info: %s' % out_dict['info_wickets'])
# print('Victims Info: %s' % out_dict['info_victims'])
# print('Contains 4: %s' % out_dict['contains_4'])
# print('Contains 6: %s' % out_dict['contains_6'])
# print('Contains R: %s' % out_dict['contains_R'])
# print('Contains W: %s' % out_dict['contains_W'])
# print('Contains B: %s' % out_dict['contains_B'])
# print('Contains O: %s' % out_dict['contains_O'])
# print('Contains Partnership: %s' % out_dict['contains_PS'])
# print('Contains Spell: %s' % out_dict['contains_SP'])

def tagged_text_format(tagged_mention_text, c1_player_ids, c2_player_ids):
    tagged_mention_text = re.sub('<<C1>>', 'Country1', re.sub('<<C2>>', 'Country2', tagged_mention_text))
    player_ids = c1_player_ids + c2_player_ids
    player_num = 1
    for player_id in player_ids:
        tagged_mention_text = re.sub('<<C(?:1|2)P' + player_id + '>>', 'Player' + str(player_num), tagged_mention_text)
        player_num = player_num + 1

    tagged_mention_text = re.sub('<<SCORE>>', 'Score', tagged_mention_text)
    return tagged_mention_text


# #testcase
# i=137
# mention_text=train_data1.iloc[i]['mention_text']
# tagged_mention_text=train_data1.iloc[i]['tagged_mention_text']
# c1_player_ids=train_data1.iloc[i]['c1_player_ids']
# c2_player_ids=train_data1.iloc[i]['c2_player_ids']
# print(mention_text)
# print(tagged_mention_text)
# print(tagged_text_format(tagged_mention_text,c1_player_ids,c2_player_ids))


def country_mention(contains_country1, contains_country2):
    if contains_country1 == 1 and contains_country2 == 1:
        status = 'country 1 and 2'
    elif contains_country1 == 0 and contains_country2 == 1:
        status = 'country 2'
    elif contains_country1 == 1 and contains_country2 == 0:
        status = 'country 1'
    else:
        status = 'no country'
    return status


def player_mention(c1_player_count, c2_player_count):
    if c1_player_count >= 1 and c2_player_count >= 1:
        status = 'country 1 and 2 player'
    elif c1_player_count == 0 and c2_player_count >= 1:
        status = 'country 2 player'
    elif c1_player_count >= 1 and c2_player_count == 0:
        status = 'country 1 player'
    else:
        status = 'no player'
    return status

def get_sentiment_score(input_sentence):
    polarity=analyser.polarity_scores(input_sentence)
    textblob_sentiment= TextBlob(input_sentence)
    if polarity['compound']>=0.05:
        polarity['orientation']='positive'
    elif polarity['compound']<=-0.05:
        polarity['orientation']='negative'
    else:
        polarity['orientation']='neutral'
    textblob_sentiment={'polarity':textblob_sentiment.sentiment.polarity,'subjectivity':round(textblob_sentiment.sentiment.subjectivity,2)}
    avg_sentiment=np.mean([polarity['compound'],textblob_sentiment['polarity']])
    return avg_sentiment

def try_text_2_digit(input_text):
    try:
        return t2d1.convert(input_text)
    except:
        return input_text

def get_facts(canonical_mention_text,info_wickets,info_overs,tagged_mention_text=None,score=None):
    canonical_mention_text1=try_text_2_digit(canonical_mention_text)
    canonical_mention_text1=re.sub('(?i)Ryan 10 Doeschate','Ryan Ten Doeschate',canonical_mention_text1)
    if tagged_mention_text is None:
        tagged_mention_text=canonical_mention_text
    runs=None
    overs=None
    wickets=None
    balls=None
    if score is not None and str(score)!='nan':
        score_list=score.split('__')
        if len(score_list)==3:
            runs=score_list[0]
            wickets=score_list[1]
            overs=score_list[2]
        elif int(score_list[1])>=10:
            runs=score_list[1]
            wickets=score_list[0]
        else:
            runs=score_list[0]
            wickets=score_list[1]
    start_delimiter='(?i)(?:^|[^a-z0-9])'
    end_delimiter='(?:[^a-z0-9]|$)'
    #87 without loss
    subpat_run_wicket1='(?:([0-9]+) (?:without loss|for none))'
    PAT_run_wicket1='{}{}{}'.format(start_delimiter,subpat_run_wicket1,end_delimiter)
    #69/0
    subpat_run_wicket2='(?:([0-9]+)/([0-9]+))'
    PAT_run_wicket2='{}{}{}'.format(start_delimiter,subpat_run_wicket2,end_delimiter)
    century_keyswords=['CENTURY','hundred','ton','100','centuries','hundreds']
    halfcentury_keyswords=['HALFCENTURY','fifty','50','HALFCENTURIES']
    PAT_century='{}({}){}'.format(start_delimiter,'|'.join(century_keyswords),end_delimiter)
    PAT_halfcentury='{}({}){}'.format(start_delimiter,'|'.join(halfcentury_keyswords),end_delimiter)
    #27 off 33
    subpat_run_ball1='(?:([0-9]+) off ([0-9]+))'
    #88 from 83 balls
    subpat_run_ball2='(?:([0-9]+) from ([0-9]+) (?:balls?|deliveries))'
    PAT_run_ball1='{}{}{}'.format(start_delimiter,subpat_run_ball1,end_delimiter)
    start_delimiter='(?i)(?:^|[^a-z0-9.])'
    end_delimiter='(?:[^a-z0-9]|$)'
    PAT_run_ball2='{}{}{}'.format(start_delimiter,subpat_run_ball2,end_delimiter)
    #7 wickets in 25 balls
    subpat_wicket_ball1='(?:([0-9]+) wickets? (?:from|in|with|and) ([0-9]+)) (?:balls?|deliveries)'
    PAT_wicket_ball1='{}{}{}'.format(start_delimiter,subpat_wicket_ball1,end_delimiter)
    #28 in his first 47 balls
    #47 from his first 47 balls
    #17 from his first 35 balls
    subpat_run_ball3='(?:([0-9]+) (?:[a-z]+ ){1,3}([0-9]+)) (?:balls?|deliveries)'
    PAT_run_ball3='{}{}{}'.format(start_delimiter,subpat_run_ball3,end_delimiter)
    #36 after 4 overs
    #215 in 40 overs
    subpat_run_over1='(?:([0-9]+) (?:after|in (?:the)) ([0-9]+)(?:th|rd|st|nd)? overs?)'
    PAT_run_over1='{}{}{}'.format(start_delimiter,subpat_run_over1,end_delimiter)
    subpat_run1='(?:([0-9]+) (?:runs?))'
    PAT_run1='{}{}{}'.format(start_delimiter,subpat_run1,end_delimiter)
    subpat_run2='([1-9][0-9][0-9]|[7-9][0-9])'
    PAT_run2='{}{}{}'.format(start_delimiter,subpat_run2,end_delimiter)
    subpat_run3='(?:<<C(?:1|2)P[0-9]+>>|<<C(?:1|2)>>) (?:to )?([0-9]{2,3})'
    PAT_run3='{}{}{}'.format(start_delimiter,subpat_run3,end_delimiter)
    subpat_run4='<<C(?:1|2)P[0-9]+>> (?:was on )?([0-9]+)'
    PAT_run4='{}{}{}'.format(start_delimiter,subpat_run4,end_delimiter)
    subpat_run5='(?:to|on|make|fluent) ([0-9]+)'
    PAT_run5='{}{}{}'.format(start_delimiter,subpat_run5,end_delimiter)
    subpat_run5_neg='(?:to|on|make|fluent) ([0-9]+) (?:[a-z]+ ){1}wickets?'
    PAT_run5_neg='{}{}{}'.format(start_delimiter,subpat_run5_neg,end_delimiter)
    subpat_over1='(?:([0-9]+)(?:th|rd|st|nd)? overs?)'
    PAT_over1='{}{}{}'.format(start_delimiter,subpat_over1,end_delimiter)
    #numeric_string=['first','second','third','fourth',]
    subpat_ball1='(?:([0-9]+)(?:th|rd|st|nd)? balls?)'
    PAT_ball1='{}{}{}'.format(start_delimiter,subpat_ball1,end_delimiter)
    PAT_run_wicket1_search = re.search(PAT_run_wicket1, canonical_mention_text)
    PAT_run_wicket2_search = re.search(PAT_run_wicket2, canonical_mention_text)
    PAT_halfcentury_search = re.search(PAT_halfcentury, canonical_mention_text)
    PAT_century_search = re.search(PAT_century, canonical_mention_text)
    PAT_run_ball1_search = re.search(PAT_run_ball1, canonical_mention_text)
    PAT_run_ball2_search = re.search(PAT_run_ball2, canonical_mention_text)
    PAT_wicket_ball1_search = re.search(PAT_wicket_ball1, canonical_mention_text)
    PAT_run_ball3_search = re.search(PAT_run_ball3, canonical_mention_text)
    PAT_run_over1_search = re.search(PAT_run_over1, canonical_mention_text)
    PAT_run1_search  = re.search(PAT_run1, canonical_mention_text)
    PAT_run2_search = re.search(PAT_run2, canonical_mention_text)
    PAT_run3_search = re.search(PAT_run3, tagged_mention_text)
    PAT_run4_search = re.search(PAT_run4, tagged_mention_text)
    PAT_run5_search = re.search(PAT_run5, tagged_mention_text)
    PAT_run5_neg_search = re.search(PAT_run5_neg, tagged_mention_text)
    PAT_over1_search = re.search(PAT_over1, canonical_mention_text1)
    PAT_ball1_search = re.search(PAT_ball1, canonical_mention_text)
    if PAT_run_wicket1_search:
        runs = PAT_run_wicket1_search.group(1)
        wickets=str(0)
    elif PAT_run_wicket2_search:
        runs = PAT_run_wicket2_search.group(1)
        wickets=PAT_run_wicket2_search.group(2)
    elif  PAT_century_search and runs is None:
        runs='100'
    elif PAT_halfcentury_search and runs is None:
        runs='50'
    elif PAT_run_ball1_search and runs is None:
        runs = PAT_run_ball1_search.group(1)
        balls=PAT_run_ball1_search.group(2)
    elif PAT_run_ball2_search and runs is None:
        runs = PAT_run_ball2_search.group(1)
        balls=PAT_run_ball2_search.group(2)
    elif PAT_wicket_ball1_search and wickets is None:
        wickets = PAT_wicket_ball1_search.group(1)
        balls=PAT_wicket_ball1_search.group(2)
    elif PAT_run_ball3_search and runs is None:
        runs = PAT_run_ball3_search.group(1)
        balls=PAT_run_ball3_search.group(2)
    elif PAT_run_over1_search and runs is None:
        runs = PAT_run_over1_search.group(1)
        overs=PAT_run_over1_search.group(2)
    if PAT_run1_search and runs is None:
        runs = PAT_run1_search.group(1)
    if PAT_run2_search and runs is None:
        runs = PAT_run2_search.group(1)
    if PAT_run3_search and runs is None:
        runs = PAT_run3_search.group(1)
    if PAT_run4_search and runs is None:
        runs = PAT_run4_search.group(1)
    if PAT_run5_search and runs is None and not PAT_run5_neg_search:
        runs = PAT_run5_search.group(1)
    if PAT_over1_search and overs is None:
        overs = PAT_over1_search.group(1)
    if PAT_ball1_search and balls is None:
        balls = PAT_ball1_search.group(1)
    if wickets is None and info_wickets is not None and info_wickets!='':
        info_wickets=t2d1.convert(info_wickets)
        PAT_wicket_info_search=re.search('([0-9]{1,2})', info_wickets)
        if PAT_wicket_info_search:
            wickets = PAT_wicket_info_search.group(1)
    if overs is None and info_overs is not None and info_overs!='':
        info_overs=t2d1.convert(info_overs)
        PAT_over_info_search=re.search('([0-9]{1,2})', info_overs)
        if PAT_over_info_search:
            overs = PAT_over_info_search.group(1)
    out_dict={}
    out_dict['runs']=runs
    out_dict['overs']=overs
    out_dict['wickets']=wickets
    out_dict['balls']=balls
    return out_dict
# i=298
# canonical_mention_text=train_data1.iloc[i]['canonical_mention_text']
# info_wickets=train_data1.iloc[i]['info_wickets']
# info_overs=train_data1.iloc[i]['info_overs']
# tagged_mention_text=train_data1.iloc[i]['tagged_mention_text']
# score=train_data1.iloc[i]['score']
# print(canonical_mention_text)
# get_facts(canonical_mention_text,info_wickets,info_overs,tagged_mention_text,score)

def resolve_pronouns_by_context(fact_subjects, country1, country2,
                                c1_player_ids, c2_player_ids, add_c1_player_ids, add_c2_player_ids,
                                contains_country1, contains_country2
                                ):
    #     print(fact_subjects,country1,country2,
    #           c1_player_ids,c2_player_ids,add_c1_player_ids,add_c2_player_ids,
    #           contains_country1,contains_country2)
    c1_player_ids = set(c1_player_ids)
    c2_player_ids = set(c2_player_ids)
    add_c1_player_ids = list(set(add_c1_player_ids).difference(c1_player_ids))
    add_c2_player_ids = list(set(add_c2_player_ids).difference(c2_player_ids))
    fact_subjects1 = []
    for subject in fact_subjects:
        if subject in ['c1', 'c2']:
            fact_subjects1.append(subject)
        elif re.search('^[0-9_]+$', subject):
            fact_subjects1.append(subject)
        elif subject.lower() in ['i', 'he', 'his', 'my']:
            if len(list(c1_player_ids)) == 0 and len(list(c2_player_ids)) == 0:
                if len(add_c1_player_ids) == 0 and len(add_c2_player_ids) == 1 and add_c2_player_ids[
                    0] not in fact_subjects:
                    fact_subjects1.append(add_c2_player_ids[0])
                elif len(add_c1_player_ids) == 1 and len(add_c2_player_ids) == 0 and add_c1_player_ids[
                    0] not in fact_subjects:
                    fact_subjects1.append(add_c1_player_ids[0])
                else:
                    fact_subjects1.append(subject)
            else:
                fact_subjects1.append(subject)
        elif subject.lower() in ['they', 'we']:
            #             if contains_country1==1:
            #                 fact_subjects1.append('c2')
            #             elif contains_country2==1:
            #                 fact_subjects1.append('c1')
            #             else:
            fact_subjects1.append(subject)
    return fact_subjects1


def get_nlp_tags(input_text, country1, country2,
                 c1_player_ids, c2_player_ids, add_c1_player_ids, add_c2_player_ids,
                 contains_country1, contains_country2,
                 dict_players_name, dict_players_name_token, dict_player_names):
    input_text_doc = nlp(input_text)
    token_list = []
    token_dep_list = []
    token_pos_list = []
    subject_cands = []
    subjects = []
    subject_players = []
    subject_players_names = []
    nsubj_count = 0
    for token in input_text_doc:
        token_list.append(token.text)
        token_dep_list.append(token.dep_)
        token_pos_list.append(token.pos_)
        if token.dep_ == 'nsubj':
            nsubj_count = nsubj_count + 1
    subject_cands = get_chunks1(input_text_doc, kw=None)

    subject_cands1 = []
    subject_cands1_lower = []
    for subject_cand in subject_cands:
        if subject_cand.lower() in ['he', 'his', 'i', 'my', 'they']:
            if len(subject_cands1) == 0:
                subject_cands1.append(subject_cand)
                subject_cands1_lower.append(subject_cand.lower())
            elif len(
                    subject_cands1) > 0 and subject_cand not in subject_cands1 and subject_cand.lower() not in subject_cands1_lower:
                subject_cands1.append(subject_cand)
                subject_cands1_lower.append(subject_cand.lower())
        elif not re.search(subject_cand, ' '.join(subject_cands1)):
            subject_cands1.append(subject_cand)
            subject_cands1_lower.append(subject_cand.lower())
    subject_cands = subject_cands1
    if (token_list[0] == country1 or token_list[0] == country2) \
            and token_dep_list[0] == 'ROOT' and token_list[0] not in subject_cands:
        subject_cands.append(token_list[0])
    else:
        if (len(token_list) >= 2):
            look_up_token = ' '.join([token_list[0], token_list[1]])
            look_up_dep = ' '.join([token_dep_list[0], token_dep_list[1]])
            if look_up_token == country1 or look_up_token == country2:
                if look_up_dep in ['compound ROOT', 'compound POSS'] and look_up_token not in subject_cands:
                    subject_cands.append(look_up_token)
            elif token_dep_list[0] == 'ROOT':

                lookup_token1 = token_list[0] + '__' + country1.lower()
                lookup_token2 = token_list[0] + '__' + country2.lower()

                lookup_token1_val = dict_players_name_token.get(lookup_token1, None)
                lookup_token2_val = dict_players_name_token.get(lookup_token2, None)

                if lookup_token1_val is not None:
                    subject_cands.append(token_list[0])
                elif lookup_token2_val is not None:
                    subject_cands.append(token_list[0])

    if token_list[0].lower() in ['his'] and token_dep_list[0] == 'poss':
        subject_cands.append(token_list[0])

    for subject_cand in subject_cands:
        if subject_cand.lower() in ['he', 'they', 'his', 'my']:
            subjects.append(subject_cand)
        elif subject_cand in ['I']:
            subjects.append(subject_cand)

        elif subject_cand == country1:
            subjects.append('c1')
        elif subject_cand == country2:
            subjects.append('c2')
        else:
            if ' ' in subject_cand:
                lookup1 = subject_cand.lower() + '__' + country1.lower()
                lookup2 = subject_cand.lower() + '__' + country2.lower()
                lookup1_val = dict_players_name.get(lookup1, None)
                lookup2_val = dict_players_name.get(lookup2, None)
                if lookup1_val is not None:
                    subjects.append(lookup1_val)
                elif lookup2_val is not None:
                    subjects.append(lookup2_val)
            else:
                lookup1 = subject_cand + '__' + country1.lower()
                lookup2 = subject_cand + '__' + country2.lower()
                lookup1_val = dict_players_name_token.get(lookup1, None)
                lookup2_val = dict_players_name_token.get(lookup2, None)
                if lookup1_val is not None:
                    subjects.append(lookup1_val)
                elif lookup2_val is not None:
                    subjects.append(lookup2_val)

    subjects = resolve_pronouns_by_context(subjects, country1, country2,
                                           c1_player_ids, c2_player_ids, add_c1_player_ids, add_c2_player_ids,
                                           contains_country1, contains_country2
                                           )

    subjects = list(set(subjects))
    for subject in subjects:
        name_val = dict_player_names.get(subject)
        if name_val is not None:
            subject_players.append(subject)
            subject_players_names.append(name_val)
    out_dict = {}
    out_dict['subjects'] = subjects
    out_dict['subject_players'] = subject_players
    out_dict['subject_players_names'] = subject_players_names
    out_dict['subject_cands'] = subject_cands
    out_dict['token_list'] = token_list
    out_dict['token_dep_list'] = token_dep_list
    out_dict['token_pos_list'] = token_pos_list
    out_dict['nsubj_count'] = nsubj_count
    out_dict['subjects_count'] = len(subjects)
    return out_dict
# i=1080
# input_text=train_data1.iloc[i]['canonical_mention_text']
# country1=train_data1.iloc[i]['country1']
# country2=train_data1.iloc[i]['country2']
# c1_player_ids=train_data1.iloc[i]['c1_player_ids']
# c2_player_ids=train_data1.iloc[i]['c2_player_ids']
# add_c1_player_ids=train_data1.iloc[i]['add_c1_player_ids']
# add_c2_player_ids=train_data1.iloc[i]['add_c2_player_ids']
# contains_country1=train_data1.iloc[i]['contains_country1']
# contains_country2=train_data1.iloc[i]['contains_country2']
# input_text
# get_nlp_tags1(input_text,country1,country2,
#              c1_player_ids,c2_player_ids,add_c1_player_ids,add_c2_player_ids,
#              contains_country1,contains_country2,
#              dict_players_name,dict_players_name_token,dict_player_names)

def tag_subject_type(fact_subjects, fact_subject_players):
    status = None
    if 'c1' in fact_subjects or 'c2' in fact_subjects:
        status = 'country'
    elif len(fact_subject_players) > 0:
        status = 'players'
    else:
        fact_subjects = [i.lower() for i in fact_subjects]

        if not re.search('[0-9]', ' '.join(fact_subjects)):
            if set(['i', 'he', 'his', 'my']).intersection(set(fact_subjects)):
                status = 'player_pronoun'
            elif set(['they']).intersection(set(fact_subjects)):
                status = 'country_pronoun'
    return status


def gen_fact_status(label_runs, fact_runs, score):
    status = None

    if label_runs == str(fact_runs) and label_runs is not None:
        status = 'ok'
    elif score is not None:
        status = 'ok'
    return status
# i=7
# label_runs=train_data1.iloc[i]['label_runs']
# fact_runs=train_data1.iloc[i]['fact_runs']
# score=train_data1.iloc[i]['score']
# print(label_runs,fact_runs,score)
# print(gen_fact_status(label_runs,fact_runs,score))

def get_chunks1(doc, kw=None):
    # doc=nlp(input_text)
    matcher = Matcher(nlp.vocab)
    pattern = [{'DEP': 'compound', 'OP': '?'}, {'DEP': 'compound', 'OP': '?'}, {'DEP': 'nsubj', 'POS': 'PROPN'}]
    matcher.add('pat1', None, pattern)
    pattern = [{'DEP': 'compound', 'OP': '?'}, {'DEP': 'compound', 'OP': '?'}, {'DEP': 'nsubj', 'POS': 'PRON'}]
    matcher.add('pat2', None, pattern)
    pattern = [{'DEP': 'poss', 'POS': 'PROPN'}]
    matcher.add('pat3', None, pattern)

    pattern = [{'DEP': 'compound', 'OP': '?'}, {'DEP': 'nsubjpass', 'POS': 'PROPN'}]
    matcher.add('pat4', None, pattern)
    pattern = [{'DEP': 'compound', 'OP': '?'}, {'DEP': 'nsubjpass', 'POS': 'PRON'}]
    matcher.add('pat5', None, pattern)
    pattern = [{'DEP': 'compound', 'OP': '?'}, {'DEP': 'poss', 'POS': 'PROPN'}]
    matcher.add('pat6', None, pattern)
    pattern = [{'DEP': 'compound', 'OP': '?'}, {'DEP': 'poss', 'POS': 'PRON'}]
    matcher.add('pat7', None, pattern)
    pattern = [{'DEP': 'compound', 'POS': 'PROPN'}, {'DEP': 'ROOT', 'POS': 'PROPN'}]
    matcher.add('pat8', None, pattern)

    pattern = [{'DEP': 'compound', 'POS': 'PROPN'}, {'DEP': 'pobj', 'POS': 'PROPN'}]
    matcher.add('pat9', None, pattern)
    pattern = [{'DEP': 'nsubj'}, {'DEP': 'punct'}]
    matcher.add('pat10', None, pattern)

    matches = matcher(doc)
    # print('Total matches found:', len(matches))
    match_list = []
    for match_id, start, end in matches:
        matched_text = doc[start:end].text
        matched_text = re.sub('(?i)[^a-z]$', '', matched_text)
        match_list.append(matched_text)
    if len(match_list) > 0:
        return sorted(match_list, key=len, reverse=True)
    else:
        return []
# i=2
# input_text=train_data1_sel.iloc[i]['canonical_mention_text']
# input_text_doc = nlp(input_text)
# train_data1_sel.iloc[i]['token_list']
# train_data1_sel.iloc[i]['token_dep_list']
# train_data1_sel.iloc[i]['token_pos_list']
# get_chunks1(input_text_doc,kw=None)

## Labeled Data Specific
taxonomy_player_pronoun_list=['he','his','him','my','i']
#newchange
taxonomy_country_pronoun_list=['they','their']
taxonomy_pronoun_list=taxonomy_player_pronoun_list+taxonomy_country_pronoun_list

def get_tags(doc):
    token_list = []
    token_pos_list = []
    token_dict = {}
    token_str = ''
    token_dep_list = []
    pronoun_count = 0
    pronoun_list = []
    player_pronoun_list = []
    country_pronoun_list = []

    for token in doc:
        token_lower = token.text.lower()
        token_list.append(token.text)
        token_pos_list.append(token.pos_)
        if (token.pos_ == 'PRON' or token_lower in taxonomy_pronoun_list) and token.text.lower() not in pronoun_list:
            pronoun_list.append(token.text.lower())
            if token.text.lower() in taxonomy_player_pronoun_list and token.text.lower() not in player_pronoun_list:
                player_pronoun_list.append(token.text.lower())
            if token.text.lower() in taxonomy_country_pronoun_list and token.text.lower() not in country_pronoun_list:
                country_pronoun_list.append(token.text.lower())

        token_dep_list.append(token.dep_)
        token_dict[token] = token_dict.get(token, '') + token.dep_
    pronoun_count = len(pronoun_list)
    player_pronoun_count = len(player_pronoun_list)
    country_pronoun_count = len(country_pronoun_list)
    out_dict = {}
    out_dict['token_list'] = token_list
    out_dict['token_pos_list'] = token_pos_list
    out_dict['token_dep_list'] = token_dep_list
    out_dict['token_dict'] = token_dict
    out_dict['pronoun_count'] = pronoun_count
    out_dict['pronoun_list'] = pronoun_list
    out_dict['player_pronoun_count'] = player_pronoun_count
    out_dict['player_pronoun_list'] = player_pronoun_list
    out_dict['country_pronoun_count'] = country_pronoun_count
    out_dict['country_pronoun_list'] = country_pronoun_list
    return out_dict


# i=6
# tagged_mention_text1=nlp(train_data1.iloc[i]['mention_text'])
# get_tags(tagged_mention_text1)['player_pronoun_count']

def feature_generation1(df,other_prams):
    [mention_dict,dict_players_name,dict_players_name_token,ent_tagg_taxonomy] = other_prams
    df['mention_key'] = df.apply(lambda x: str(x['article_number']) + '_' + str(x['mention_id'][1:]), axis=1)
    df['mention_obj'] = df.apply(lambda x: mention_dict[x['mention_key']], axis=1)
    df['article_pera_num'] = df.apply(lambda x: x['mention_obj']['pera_number'], axis=1)
    df['article_line_num'] = df.apply(lambda x: x['mention_obj']['line_number'], axis=1)
    df['article_mention_line'] = df.apply(lambda x: x['mention_obj']['line_text'], axis=1)
    df['article_mention_previous_line'] = df.apply(lambda x: x['mention_obj']['previous_line_text'], axis=1)
    df['article_mention_line'] = df.apply(lambda x: norm_mention_text(x['article_mention_line']), axis=1)
    df['article_mention_previous_line'] = df.apply(lambda x: norm_mention_text(x['article_mention_previous_line']),
                                                   axis=1)
    df.drop(columns=['mention_obj'], inplace=True)
    df['canonical_mention_text'] = df.apply(lambda x: norm_mention_text(x['mention_text']), axis=1)
    df['doc'] = df.apply(lambda x: nlp(x['canonical_mention_text']), axis=1)
    df['tags'] = df.apply(lambda x: get_tags(x['doc']), axis=1)
    df['pronoun_count'] = df.apply(lambda x: x['tags']['pronoun_count'], axis=1)
    df['pronoun_list'] = df.apply(lambda x: x['tags']['pronoun_list'], axis=1)
    df['player_pronoun_count'] = df.apply(lambda x: x['tags']['player_pronoun_count'], axis=1)
    df['player_pronoun_list'] = df.apply(lambda x: x['tags']['player_pronoun_list'], axis=1)
    df['country_pronoun_count'] = df.apply(lambda x: x['tags']['country_pronoun_count'], axis=1)
    df['country_pronoun_list'] = df.apply(lambda x: x['tags']['country_pronoun_list'], axis=1)
    df['mention_text_length'] = df.apply(lambda x: len(x['mention_text']), axis=1)
    df['mention_text_contains_number'] = \
        df.apply(lambda x: 1 if re.search('[0-9]', x['mention_text']) is not None else 0, axis=1)

    df['tagged_mention_text_obj'] = df.apply(
        lambda x: get_additional_country_players(x['canonical_mention_text'], x['article_mention_line'],
                                                 x['article_mention_previous_line'], x['country1'], x['country2'],
                                                 dict_players_name, dict_players_name_token), axis=1)
    df['tagged_mention_text'] = df.apply(lambda x: x['tagged_mention_text_obj']['mention_text'], axis=1)
    df['canonical_mention_text1'] = df.apply(lambda x: clean_br(x['tagged_mention_text']), axis=1)
    df['count_country'] = df.apply(lambda x: x['tagged_mention_text_obj']['count_country'], axis=1)
    df['count_players'] = df.apply(lambda x: x['tagged_mention_text_obj']['count_players'], axis=1)
    df['count_players_country'] = df.apply(lambda x: x['tagged_mention_text_obj']['count_players_country'], axis=1)
    df['contains_country1'] = df.apply(lambda x: x['tagged_mention_text_obj']['contains_country1'], axis=1)
    df['contains_country2'] = df.apply(lambda x: x['tagged_mention_text_obj']['contains_country2'], axis=1)
    df['contains_country1_player'] = df.apply(lambda x: x['tagged_mention_text_obj']['contains_country1_player'],
                                              axis=1)
    df['contains_country2_player'] = df.apply(lambda x: x['tagged_mention_text_obj']['contains_country2_player'],
                                              axis=1)
    df['player_name'] = df.apply(lambda x: x['tagged_mention_text_obj']['maching_name_grams'], axis=1)
    df['c1_player_ids'] = df.apply(lambda x: x['tagged_mention_text_obj']['c1_player_ids'], axis=1)
    df['c2_player_ids'] = df.apply(lambda x: x['tagged_mention_text_obj']['c2_player_ids'], axis=1)

    df['add_count_country'] = df.apply(lambda x: x['tagged_mention_text_obj']['additional_count_country'], axis=1)
    df['add_count_players'] = df.apply(lambda x: x['tagged_mention_text_obj']['additional_count_players'], axis=1)
    df['add_count_players_country'] = df.apply(
        lambda x: x['tagged_mention_text_obj']['additional_count_players_country'], axis=1)
    df['add_contains_country1'] = df.apply(lambda x: x['tagged_mention_text_obj']['additional_contains_country1'],
                                           axis=1)
    df['add_contains_country2'] = df.apply(lambda x: x['tagged_mention_text_obj']['additional_contains_country2'],
                                           axis=1)
    df['add_contains_country1_player'] = df.apply(
        lambda x: x['tagged_mention_text_obj']['additional_contains_country1_player'], axis=1)
    df['add_contains_country2_player'] = df.apply(
        lambda x: x['tagged_mention_text_obj']['additional_contains_country2_player'], axis=1)
    # df['add_maching_name_grams']=df.apply(lambda x: x['tagged_mention_text_obj']['additional_maching_name_grams'],axis=1)
    df['add_c1_player_ids'] = df.apply(lambda x: x['tagged_mention_text_obj']['additional_c1_player_ids'], axis=1)
    df['add_c2_player_ids'] = df.apply(lambda x: x['tagged_mention_text_obj']['additional_c2_player_ids'], axis=1)
    df.drop(columns=['tagged_mention_text_obj'], inplace=True)

    df['entity_tag_obj'] = df.apply(lambda x: entity_tagging(
        x['tagged_mention_text'],
        x['country1'], x['country2'],
        x['doc'], x['c1_player_ids'], x['c2_player_ids'], ent_tagg_taxonomy), axis=1)
    df['tagged_mention_text1'] = df.apply(lambda x: x['entity_tag_obj']['tagged_mention_text1'], axis=1)
    df['tagged_mention_text2'] = df.apply(lambda x: x['entity_tag_obj']['tagged_mention_text2'], axis=1)
    df['contains_shot_kw'] = df.apply(lambda x: x['entity_tag_obj']['contains_shot_kw'], axis=1)
    df['contains_ball_kw'] = df.apply(lambda x: x['entity_tag_obj']['contains_ball_kw'], axis=1)
    df['contains_fieldpostion_kw'] = df.apply(lambda x: x['entity_tag_obj']['contains_fieldpostion_kw'], axis=1)
    df['contains_other_kw'] = df.apply(lambda x: x['entity_tag_obj']['contains_other_kw'], axis=1)
    df['contains_other_mb_kw'] = df.apply(lambda x: x['entity_tag_obj']['contains_other_mb_kw'], axis=1)
    df['contains_other_sb_kw'] = df.apply(lambda x: x['entity_tag_obj']['contains_other_sb_kw'], axis=1)
    df['contains_conclusion_kw'] = df.apply(lambda x: x['entity_tag_obj']['contains_conclusion_kw'], axis=1)
    df['contains_score'] = df.apply(lambda x: x['entity_tag_obj']['contains_score'], axis=1)
    df['contains_country_score'] = df.apply(lambda x: x['entity_tag_obj']['contains_country_score'], axis=1)
    df['contains_player_score'] = df.apply(lambda x: x['entity_tag_obj']['contains_player_score'], axis=1)
    df['score'] = df.apply(lambda x: x['entity_tag_obj']['score'], axis=1)
    df['contains_balls'] = df.apply(lambda x: x['entity_tag_obj']['contains_balls'], axis=1)
    df['contains_overs'] = df.apply(lambda x: x['entity_tag_obj']['contains_overs'], axis=1)
    df['contains_fours'] = df.apply(lambda x: x['entity_tag_obj']['contains_fours'], axis=1)
    df['contains_sixes'] = df.apply(lambda x: x['entity_tag_obj']['contains_sixes'], axis=1)
    df['contains_powerplay'] = df.apply(lambda x: x['entity_tag_obj']['contains_powerplay'], axis=1)
    df['contains_wickets'] = df.apply(lambda x: x['entity_tag_obj']['contains_wickets'], axis=1)
    df['contains_victims'] = df.apply(lambda x: x['entity_tag_obj']['contains_victims'], axis=1)
    df['contains_RS'] = df.apply(lambda x: x['entity_tag_obj']['contains_RS'], axis=1)
    df['info_balls'] = df.apply(lambda x: x['entity_tag_obj']['info_balls'], axis=1)
    df['info_overs'] = df.apply(lambda x: x['entity_tag_obj']['info_overs'], axis=1)
    df['info_over'] = df.apply(lambda x: x['entity_tag_obj']['info_over'], axis=1)
    df['info_fours'] = df.apply(lambda x: x['entity_tag_obj']['info_fours'], axis=1)
    df['info_sixes'] = df.apply(lambda x: x['entity_tag_obj']['info_sixes'], axis=1)
    df['info_powerplay'] = df.apply(lambda x: x['entity_tag_obj']['info_powerplay'], axis=1)
    df['info_wickets'] = df.apply(lambda x: x['entity_tag_obj']['info_wickets'], axis=1)
    df['info_wicket'] = df.apply(lambda x: x['entity_tag_obj']['info_wicket'], axis=1)
    df['info_overs'] = df.apply(lambda x: choose_out_of2(x['info_overs'], x['info_over']), axis=1)
    df['info_wickets'] = df.apply(lambda x: choose_out_of2(x['info_wickets'], x['info_wicket']), axis=1)
    df.drop(columns=['info_over', 'info_wicket'], inplace=True)
    df['info_victims'] = df.apply(lambda x: x['entity_tag_obj']['info_victims'], axis=1)
    df['contains_4'] = df.apply(lambda x: x['entity_tag_obj']['contains_4'], axis=1)
    df['contains_6'] = df.apply(lambda x: x['entity_tag_obj']['contains_6'], axis=1)
    df['contains_R'] = df.apply(lambda x: x['entity_tag_obj']['contains_R'], axis=1)
    df['contains_W'] = df.apply(lambda x: x['entity_tag_obj']['contains_W'], axis=1)
    df['contains_OV'] = df.apply(lambda x: x['entity_tag_obj']['contains_OV'], axis=1)
    df['contains_B'] = df.apply(lambda x: x['entity_tag_obj']['contains_B'], axis=1)
    df['contains_O'] = df.apply(lambda x: x['entity_tag_obj']['contains_O'], axis=1)
    df['contains_PS'] = df.apply(lambda x: x['entity_tag_obj']['contains_PS'], axis=1)
    df['contains_SP'] = df.apply(lambda x: x['entity_tag_obj']['contains_SP'], axis=1)
    df['count_MBK'] = df['contains_balls'] + df['contains_overs'] + df['contains_fours'] + \
                      df['contains_sixes'] + df['contains_powerplay'] + df['contains_wickets'] + \
                      df['contains_victims'] + df['contains_PS'] + df['contains_SP'] + \
                      +df['contains_other_mb_kw']
    df['count_SBK'] = df['contains_B'] + df['contains_4'] + df['contains_6'] + \
                      df['contains_W'] + df['contains_O'] + df['contains_other_sb_kw'] + +df['contains_conclusion_kw']
    df['contains_balls_info'] = df.apply(lambda x: 1 if len(x['info_balls']) > 0 else 0, axis=1)
    df['contains_overs_info'] = df.apply(lambda x: 1 if len(x['info_overs']) > 0 else 0, axis=1)
    df['contains_fours_info'] = df.apply(lambda x: 1 if len(x['info_fours']) > 0 else 0, axis=1)
    df['contains_sixes_info'] = df.apply(lambda x: 1 if len(x['info_sixes']) > 0 else 0, axis=1)
    df['contains_powerplay_info'] = df.apply(lambda x: 1 if len(x['info_powerplay']) > 0 else 0, axis=1)
    df['contains_wickets_info'] = df.apply(lambda x: 1 if len(x['info_wickets']) > 0 else 0, axis=1)
    df['contains_victims_info'] = df.apply(lambda x: 1 if len(x['info_victims']) > 0 else 0, axis=1)
    df['country_mention'] = df.apply(lambda x: country_mention(x['contains_country1'], x['contains_country2']), axis=1)
    df['c1_player_count'] = df.apply(lambda x: len(x['c1_player_ids']), axis=1)
    df['c2_player_count'] = df.apply(lambda x: len(x['c2_player_ids']), axis=1)
    df['player_mention'] = df.apply(lambda x: player_mention(x['c1_player_count'], x['c2_player_count']), axis=1)
    df['sentiment_score'] = df['mention_text'].apply(lambda x: get_sentiment_score(x))
    df.drop(columns=['entity_tag_obj'], inplace=True)
    out_dict = {}
    out_dict['df'] = df
    return out_dict

def feature_generation2(df,other_prams):
    # get facts
    [dict_players_name, dict_players_name_token,dict_player_names] = other_prams
    df['fact_obj'] = df.apply(lambda x: get_facts(
        x['canonical_mention_text'], x['info_wickets'], x['info_overs'], x['tagged_mention_text'], x['score']), axis=1)
    df['fact_runs'] = df.apply(lambda x: x['fact_obj']['runs'], axis=1)
    df['fact_wickets'] = df.apply(lambda x: x['fact_obj']['wickets'], axis=1)
    df['fact_wickets_int'] = df['fact_wickets'].fillna('-100').astype('int')
    df['fact_overs'] = df.apply(lambda x: x['fact_obj']['overs'], axis=1)
    df['fact_balls'] = df.apply(lambda x: x['fact_obj']['balls'], axis=1)
    df.drop(columns=['fact_obj'], inplace=True)

    # get nlp subject
    df['nlp_obj'] = df.apply(
        lambda x: get_nlp_tags(x['canonical_mention_text'], x['country1'], x['country2'], x['c1_player_ids'],
                               x['c2_player_ids'], x['add_c1_player_ids'], x['add_c2_player_ids'],
                               x['contains_country1'], x['contains_country2'], dict_players_name,
                               dict_players_name_token, dict_player_names), axis=1)
    df['fact_subjects'] = df.apply(lambda x: x['nlp_obj']['subjects'], axis=1)
    df['fact_subjects_count'] = df.apply(lambda x: x['nlp_obj']['subjects_count'], axis=1)
    df['fact_subject_players'] = df.apply(lambda x: x['nlp_obj']['subject_players'], axis=1)
    df['subject_players_names'] = df.apply(lambda x: x['nlp_obj']['subject_players_names'], axis=1)
    df['fact_subject_cands'] = df.apply(lambda x: x['nlp_obj']['subject_cands'], axis=1)
    df['fact_nsubj_count'] = df.apply(lambda x: x['nlp_obj']['nsubj_count'], axis=1)
    df['token_list'] = df.apply(lambda x: x['nlp_obj']['token_list'], axis=1)
    df['token_dep_list'] = df.apply(lambda x: x['nlp_obj']['token_dep_list'], axis=1)
    df['token_pos_list'] = df.apply(lambda x: x['nlp_obj']['token_list'], axis=1)
    df.drop(columns=['nlp_obj'], inplace=True)
    df['fact_subject_type'] = df.apply(lambda x: (tag_subject_type(x['fact_subjects'], x['fact_subject_players'])),
                                       axis=1)
    df['fact_status'] = df.apply(
        lambda x: gen_fact_status(x['label_runs'],
                                  x['fact_runs'],
                                  x['score']
                                  ), axis=1)
    out_dict = {}
    out_dict['df'] = df
    return out_dict