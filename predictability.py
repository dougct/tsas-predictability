# -*- coding: utf-8 -*-

import math
import numpy as np
import re
from collections import defaultdict


def entropy(Y):
    """
    H(X)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count / len(Y)
    return np.sum((-1) * prob * np.log2(prob))


def joint_entropy(Y,X):
    """
    H(X,Y)
    """
    XY = np.c_[X,Y]
    return entropy(XY)


def conditional_entropy(X, Y):
    """
    H(X | Y) = H(X, Y) - H(Y)
    """
    return joint_entropy(X, Y) - entropy(Y)


def rand_entropy(sequence):
    """
    Compute the "random entropy", that is, the entropy of a uniform distribution.

    Equation:
        S_{rand} = \log_{2}(n), where n is the number of unique symbols in the input sequence.

    Args:
        sequence: 1-D array-like sequence of symbols.

    Returns:
        A float representing the random entropy of the input sequence.

    Reference: 
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu, 
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    alphabet_size = np.unique(sequence).size
    return np.log2(alphabet_size)


def unc_entropy(sequence):
    """
    Compute temporal-uncorrelated entropy (Shannon entropy).

    Equation:
    S_{unc} = - \sum p(i) \log_2{p(i)}, for each symbol i in the input sequence.

    Args:
        sequence: the input sequence of symbols.

    Returns:
        temporal-uncorrelated entropy of the input sequence.

    Reference: 
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu, 
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    _, counts = np.unique(sequence, return_counts = True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))


def entropy_kontoyiannis(sequence):
    """
    Estimate the entropy rate of the sequence using Kontoyiannis' estimator.
    
    Reference:
        Kontoyiannis, I., Algoet, P. H., Suhov, Y. M., & Wyner, A. J. (1998).
        Nonparametric entropy estimation for stationary processes and random
        fields, with applications to English text. IEEE Transactions on Information
        Theory, 44(3), 1319-1327.
    
    Equation:
        S_{real} = \left( \frac{1}{n} \sum \Lambda_{i} \right)^{-1}\log_{2}(n)
    
    Args:
        sequence: the input sequence of symbols.

    Returns:
        A float representing an estimate of the entropy rate of the sequence.
    """
    if not sequence:
        return 0.0    
    lambdas = 0
    n = len(sequence)
    for i in range(n):
        current_sequence = ''.join(sequence[0:i])
        match = True
        k = i
        while match and k < n:
            k += 1
            match = ''.join(sequence[i:k]) in current_sequence
        lambdas += (k - i)
    return (1.0 * len(sequence) / lambdas) * np.log2(len(sequence))


def max_predictability(S, N):
    """
    Estimate the maximum predictability of a sequence with 
    entropy S and alphabet size N.

    Equation:
    $S = - H(\Pi) + (1 - \Pi)\log(N - 1),$
        where $H(\Pi)$ is given by
    $H(\Pi) = \Pi \log_2(\Pi) + (1 - \Pi) \log_2(1 - \Pi)$

    Args:
        S: the entropy of the input sequence of symbols.
        N: the size of the alphabet (number of unique symbols)

    Returns:
        the maximum predictability of the sequence.

    Reference: 
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu, 
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    if S == 0.0 or N <= 1:
        return 1.0
    for p in np.arange(0.0001, 1.0000, 0.0001):
        h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        pi_max = h + (1 - p) * math.log2(N - 1) - S
        if pi_max <= 0.001:
            return round(p, 3)
    return 0.0


def longest_match_length(s, i):
    sequence = ''.join(s[0:i])
    k = i
    while k < len(s) and ''.join(s[i:k]) in sequence:
        k += 1
    return k - i


def entropy_kontoyiannis_longest_match(sequence):
    """
    Estimate the entropy rate of the sequence using Kontoyiannis' estimator.
    
    Reference:
        Kontoyiannis, I., Algoet, P. H., Suhov, Y. M., & Wyner, A. J. (1998).
        Nonparametric entropy estimation for stationary processes and random
        fields, with applications to English text. IEEE Transactions on Information
        Theory, 44(3), 1319-1327.
    
    Equation:
        S_{real} = \left( \frac{1}{n} \sum \Lambda_{i} \right)^{-1}\log_{2}(n)
    
    Args:
        sequence: the input sequence of symbols.

    Returns:
        A float representing an estimate of the entropy rate of the sequence.
    """
    if not sequence:
        return 0.0    
    lambdas = 0
    n = len(sequence)
    for i in range(n):
        match_length = longest_match_length(sequence, i)
        print(sequence[0:i], sequence[i:i+match_length])
        lambdas += (match_length + 1)
    print(lambdas)
    return (1.0 * len(sequence) / lambdas) * np.log2(len(sequence))


def baseline_entropy(sequence):
    n = len(sequence)
    m = len(set(sequence)) - 1
    k = n - m
    baseline_routine = (k * k) / 4 + k / 2
    baseline_novelty = m
    return n * np.log2(n) / (baseline_routine + baseline_novelty)


def baseline_entropy_kontoyiannis(sequence):
    if not sequence:
        return 0.0
    n, n_unique = len(sequence), len(set(sequence))
    baseline_sequence = [sequence[0]] * (n - n_unique) + list(set(sequence))
    return entropy_kontoyiannis(baseline_sequence)


# Context

def sequence_splitting(X, C):
    ents = []
    w = []
    for context in set(C):
        sequence = [str(X[i]) for i in range(len(X)) if C[i] == context]
        ents.append(entropy_kontoyiannis(sequence))
        w.append(len(sequence) / len(X))
    return np.average(ents, weights=w)


def sequence_merging(X, Y):
    assert(len(X) == len(Y))
    XY = [str(X[i]) + str(Y[i]) for i in range(len(X))]
    return entropy_kontoyiannis(XY) - entropy_kontoyiannis(Y)


def sequence_concatenating(X, Y):
    assert(len(X) == len(Y))
    return entropy_kontoyiannis(Y + X) - entropy_kontoyiannis(Y)



# Metrics

def regularity(sequence):
    """
    Compute the regularity of a sequence.

    The regularity basically measures what percentage of a user's
    visits are to a previously visited place.

    Parameters
    ----------
    sequence : list
        A list of symbols.

    Returns
    -------
    float
        100 minus the percentage of the symbols in the sequence that are unique.
    """
    if len(set(sequence)) <= 1:
        return 100.0

    if len(set(sequence)) == len(sequence):
        return .0

    return 100.0 - len(set(sequence)) * 100 / len(sequence)


def stationarity(sequence):
    """
    Compute the stationarity of a sequence.

    A stationary transition is one whose source and destination symbols
    are the same. The stationarity measures the percentage of transitions
    to the same location.

    Parameters
    ----------
    sequence : list
        A list of symbols.

    Returns
    -------
    float
        Percentage of the sequence that is stationary.
    """
    if len(sequence) <= 1:
        return 100.0

    if len(sequence) == len(set(sequence)):
        return .0

    stationary_transitions = 0
    for i in range(1, len(sequence)):
        if sequence[i - 1] == sequence[i]:
            stationary_transitions += 1
    return stationary_transitions * 100 / (len(sequence) - 1)


def _suffix_array_manber_myers(s):
    """
    Reference: http://algorithmicalley.com/archive/2013/06/30/suffix-arrays.aspx
    Reference: https://louisabraham.github.io/notebooks/suffix_arrays.html
    """
    def sort_bucket(s, bucket, order):
        d = defaultdict(list)
        for i in bucket:
            key = ''.join(s[i + order // 2:i + order])
            d[key].append(i)
        result = []
        for k, v in sorted(d.items()):
            if len(v) > 1:
                result += sort_bucket(s, v, 2 * order)
            else:
                result.append(v[0])
        return result

    return sort_bucket(s, range(len(s)), 1)


def _kasai(s, sa):
    """
    Reference: https://web.stanford.edu/class/cs166/lectures/03/Small03.pdf
    Reference: https://web.stanford.edu/class/archive/cs/cs166/cs166.1146/
    """
    n = len(s)
    k = 0
    lcp = [0] * n
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    for i in range(n):
        k = k - 1 if k > 0 else 0
        if rank[i] == n - 1:
            k = 0
            continue
        j = sa[rank[i] + 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
    return lcp


def diversity(sequence): 
    """
    Returns the ratio of distinct substrings over the total number of substrings in the sequence.
    Reference: https://www.youtube.com/watch?v=m2lZRmMjebw
    """
    if not sequence:
        return 0.0

    suffix_array = _suffix_array_manber_myers(sequence)
    lcp = _kasai(sequence, suffix_array)
    
    n = len(sequence)
    total_substrs = (n * (n + 1)) // 2
    distinct_substrs = total_substrs - sum(lcp) 
    return distinct_substrs / total_substrs

