# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:29:35 2022

@author: Alessandra
"""

import pandas as pd
import random
import math
from scipy.stats import entropy
from numpy.linalg import norm
import itertools

# =================== FUNCTIONS ===================

def harmonic_number(n):
    """Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)


def compute_distr(items, adjusted=False):
       """Compute the genre distribution for a given list of Items."""
       n = len(items)
       sum_one_over_ranks = harmonic_number(n)
       count = 0
       distr = {}
       for indx, item in enumerate(items):
           rank = indx + 1
           story_freq = distr.get(item, 0.)
           distr[item] = story_freq + 1 * 1 / rank / sum_one_over_ranks if adjusted else story_freq + 1 * 1 / n
           count += 1

       return distr

def opt_merge_max_mappings(dict1, dict2):
    """ Merges two dictionaries based on the largest value in a given mapping.
    Parameters
    ----------
    dict1 : Dict[Any, Comparable]
    dict2 : Dict[Any, Comparable]
    Returns
    -------
    Dict[Any, Comparable]
        The merged dictionary
    """
    # we will iterate over `other` to populate `merged`
    merged, other = (dict1, dict2) if len(dict1) > len(dict2) else (dict2, dict1)
    merged = dict(merged)

    for key in other:
        if key not in merged or other[key] > merged[key]:
            merged[key] = other[key]
    return merged

def compute_kl_divergence(s, q, alpha=0.001):
    """
    KL (p || q), the lower the better.
    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    try:
        assert 0.99 <= sum(s.values()) <= 1.01
        assert 0.99 <= sum(q.values()) <= 1.01
    except AssertionError:
        print("Assertion Error")
        pass
    #kl_div = 0.
    ss = []
    qq = []
    merged_dic = opt_merge_max_mappings(s, q)
    for key in sorted(merged_dic.keys()):
        q_score = q.get(key, 0.)
        s_score = s.get(key, 0.)
        ss.append((1 - alpha) * s_score + alpha * q_score)
        qq.append((1 - alpha) * q_score + alpha * s_score)
        # by contruction they cannot be both 0
        # if s_score == 0 and q_score == 0:
        #     pass
        #     # raise Exception('Something is wrong in compute_kl_divergence')
        # elif s_score == 0:
        #     ss_score = (1 - alpha) * s_score + alpha * q_score
        #     ss.append(ss_score)
        #     qq.append(q_score)
        # elif q_score == 0:
        #     qq_score = (1 - alpha) * q_score + alpha * s_score
        #     ss.append(s_score)
        #     qq.append(qq_score)
        # else:
        #     ss.append(s_score)
        #     qq.append(q_score)
    kl = entropy(ss, qq, base=2)
    jsd = JSD(ss,qq)
    return [kl, jsd]


def KL_symmetric(a, b):
    return (entropy(a, b, base=2) + entropy(b, a, base=2))/2


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    # return 0.5 * (KL(_P, _M) + KL(_Q, _M))
    # added the abs to catch situations where the disocunting causes a very small <0 value, check this more!!!!
    try:
        jsd_root = math.sqrt(abs(0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2))))
    except ZeroDivisionError:
        print(P)
        print(Q)
        print()
        jsd_root = None
    return jsd_root



def compare_recommendations(x, y):
        # output = rbo(x, y, 0.9)
        freq_x = compute_distr(x, adjusted=True)
        freq_y = compute_distr(y, adjusted=True)
        divergence_with_discount = compute_kl_divergence(freq_x, freq_y)

        freq_x = compute_distr(x, adjusted=False)
        freq_y = compute_distr(y, adjusted=False)
        divergence_without_discount = compute_kl_divergence(freq_x, freq_y)
        # xy = compute_kl_divergence(freq_x, freq_y)
        # yx = compute_kl_divergence(freq_y, freq_x)
        # kl = 1/2*(xy[0] + yx[0])
        # jsd = 1/2*(xy[1] + yx[1])
        #return [divergence_with_discount, divergence_without_discount]
        return divergence_with_discount


    
    
    
# ======================== GENERATE RANDOM RECOMMENDATIONS ========================
users = [i for i in range(717)] # 100 users 

random_recs = pd.DataFrame()
random_recs["user_id"] = users

recs = []

chains = [2,4,5,6,7,8,9]

for user in users: 
    rec = random.choices(chains, k = 20) # 15 recs per user 
    recs.append(rec)
    
random_recs["chains"] = recs



# get combinations for all users 
#for L in range(0, len(recs)+1):
#    for subset in itertools.combinations(recs, L):
#       print(subset)

combinations = list(itertools.combinations(recs, 2))


rec1 = recs[0]
rec2 = recs[1]

all_divergences = []

for comb in combinations: 
    divergence = compare_recommendations(comb[0], comb[1])
    all_divergences.append(divergence)
    
    
# ======================== CALCULATE DIVERSION ========================
divergence = compare_recommendations(rec1, rec2)
