import re
from fuzzywuzzy import fuzz
import numpy as np
#NLP
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
from spacy.matcher import Matcher


def gen_is_out_player_in_batting_player_ids(out_player_id, c1_player_ids, c2_player_ids):
    c1_player_ids1 = []
    c2_player_ids1 = []
    temp_list = [i.split('_') for i in c1_player_ids]
    for i in temp_list:
        for j in i:
            if j not in c1_player_ids1:
                c1_player_ids1.append(j)
    temp_list = [i.split('_') for i in c2_player_ids]
    for i in temp_list:
        for j in i:
            if j not in c2_player_ids1:
                c2_player_ids1.append(j)
    if out_player_id in c1_player_ids1 or out_player_id in c2_player_ids1:
        return 1
    else:
        return 0


# out_player_id=train_data_sb_OUT_candidates.iloc[0]['comm_out_player_id']
# c1_player_ids=train_data_sb_OUT_candidates.iloc[0]['c1_player_ids']
# c2_player_ids=train_data_sb_OUT_candidates.iloc[0]['c2_player_ids']
# print(out_player_id,c1_player_ids,c2_player_ids)
# gen_is_out_player_in_batting_player_ids(out_player_id,c1_player_ids,c2_player_ids)
def gen_is_bowler_in_bowling_player_ids(bowler_id, c1_player_ids, c2_player_ids):
    c1_player_ids1 = []
    c2_player_ids1 = []
    temp_list = [i.split('_') for i in c1_player_ids]
    for i in temp_list:
        for j in i:
            if j not in c1_player_ids1:
                c1_player_ids1.append(j)
    temp_list = [i.split('_') for i in c2_player_ids]
    for i in temp_list:
        for j in i:
            if j not in c2_player_ids1:
                c2_player_ids1.append(j)

    if bowler_id in c1_player_ids1 or bowler_id in c2_player_ids1:
        return 1
    else:
        return 0


def gen_is_other_batsmasn_in_batting_player_ids(other_batsman_id, c1_player_ids, c2_player_ids):
    c1_player_ids1 = []
    c2_player_ids1 = []
    temp_list = [i.split('_') for i in c1_player_ids]
    for i in temp_list:
        for j in i:
            if j not in c1_player_ids1:
                c1_player_ids1.append(j)
    temp_list = [i.split('_') for i in c2_player_ids]
    for i in temp_list:
        for j in i:
            if j not in c2_player_ids1:
                c2_player_ids1.append(j)

    if other_batsman_id in c1_player_ids1 or other_batsman_id in c2_player_ids1:
        return 1
    else:
        return 0


def get_upper_case_tokens(input_string):
    uppertokens = []
    PAT_ENT = '^([A-Z]+|C(?:1|2)P[0-9]{1,3})$'
    doc = nlp(input_string)
    for t in doc:
        PAT_ENT_search = re.search(PAT_ENT, t.text)
        if PAT_ENT_search:
            uppertokens.append(t.text)
    return set(uppertokens)


# i=17
# input_string=train_data_sb_OUT.iloc[17]['canonical_mention_text1']
# print(input_string)
# print(get_upper_case_tokens(input_string))

# i=607
# input_string=train_data_sb_OUT_candidates.iloc[i]['canonical_mention_text']
# print(input_string)
# print(get_upper_case_tokens(input_string))

def jaccard_similarity(s1, s2):
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def get_men_to_comm_sim(mention_text, commentary_line):
    uppertokens_mention = get_upper_case_tokens(mention_text)
    uppertokens_commenatary = get_upper_case_tokens(commentary_line)
    sim_jaccard_upper = round(jaccard_similarity(uppertokens_mention, uppertokens_commenatary), 2)
    sim_token_set = fuzz.token_set_ratio(mention_text, commentary_line) / 100
    sim_score = round((2 * sim_jaccard_upper + sim_token_set) / 3, 2)
    out_dict = {}
    out_dict['uppertokens_mention'] = uppertokens_mention
    out_dict['uppertokens_commenatary'] = uppertokens_commenatary
    out_dict['sim_jaccard_upper'] = sim_jaccard_upper
    out_dict['sim_token_set'] = sim_token_set
    out_dict['sim_score'] = sim_score
    return out_dict


# i=604
# mention_text=train_data_sb_OUT_candidates.iloc[i]['canonical_mention_text']
# commentary_line=train_data_sb_OUT_candidates.iloc[i]['comm_canonical_commentary_line']
# get_men_to_comm_sim(mention_text,commentary_line)

def gen_candidate_score_out(has_out_batsman, has_bowler, has_other_batsmans, sim_score):
    return 4 * has_out_batsman + 2 * has_bowler + 1 * has_other_batsmans + sim_score


def is_run_match(fact_runs, runs_c1, runs_c2, comm_total_tuns):
    status = 0
    if fact_runs is not None:
        if int(fact_runs) == int(comm_total_tuns):
            status = 1
    return status


def is_over_match(fact_overs, comm_over):
    status = 0
    if fact_overs is not None:
        fact_overs = int(fact_overs)
        comm_over = int(comm_over)
        if np.abs(fact_overs - comm_over) <= 1:
            status = 1
    return status


def gen_candidate_score_lastball(contains_conclusion_kw, innings_number, is_run_match, is_over_match, sim_score):
    last_ball_match = 0
    if contains_conclusion_kw and innings_number == 2:
        last_ball_match = 1
    return 3 * last_ball_match + 2 * is_run_match + 2 * is_over_match + sim_score



