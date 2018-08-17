import numpy as np
import math

def orig_log_choose(n, k):
    r = 0.0
    # swap for efficiency if k is more than half of n
    if k * 2 > n:
        k = n - k

    for  d in xrange(1,k+1):
        r += math.log(n, 10)
        r -= math.log(d, 10)
        n -= 1

    return r


# return the genotype and log10 p-value
def orig_bayes_gt(ref, alt, is_dup):
    # probability of seeing an alt read with true genotype of of hom_ref, het, hom_alt respectively
    if is_dup: # specialized logic to handle non-destructive events such as duplications
        p_alt = [1e-2, 1/5.0, 1/3.0]
    else:
        p_alt = [1e-3, 0.5, 0.9]

    total = ref + alt
    log_combo = orig_log_choose(total, alt)

    lp_homref = log_combo + alt * math.log(p_alt[0], 10) + ref * math.log(1 - p_alt[0], 10)
    lp_het = log_combo + alt * math.log(p_alt[1], 10) + ref * math.log(1 - p_alt[1], 10)
    lp_homalt = log_combo + alt * math.log(p_alt[2], 10) + ref * math.log(1 - p_alt[2], 10)

    return (lp_homref, lp_het, lp_homalt)

def orig_total(lk):
    gt_sum = 0
    for gt in lk:
        try:
            gt_sum += 10**gt
        except OverflowError:
            gt_sum += 0
    return gt_sum


def np_indiv_log_choose(n_minus_k, k):
    n = n_minus_k + k
    r = 0.0
    # swap for efficiency if k is more than half of n
    if k * 2 > n:
        k = n - k
    for  d in xrange(1,k+1):
        r += np.log10(n, dtype='float128')
        r -= np.log10(d, dtype='float128')
        n -= 1
    return r
    
np_log_choose = np.vectorize(np_indiv_log_choose)

def np_bayes_gt(ref, alt, is_dup=False):
    if is_dup:
        p_alt = np.array([1e-2, 1/5.0, 1/3.0], dtype='float128').reshape(1,3)
    else:
        p_alt = np.array([1e-3, 0.5, 0.9], dtype='float128').reshape(1,3)
    log_combo = np_log_choose(ref, alt)
    lk = log_combo + alt * np.log10(p_alt, dtype='float128') + ref * np.log10(1. - p_alt, dtype='float128')
    return lk

olk = orig_bayes_gt(39, 14, False)
nlk = np_bayes_gt(np.array([39.], dtype='float128'), np.array([14.], dtype='float128'), False)

n_total = np.logaddexp.reduce(nlk * np.log(10., dtype='float128'), axis=1, dtype='float128') * np.log10(np.e, dtype='float128')
o_total = math.log(orig_total(olk), 10)
orig_gt_idx = olk.index(max(olk))
np_gt_idx = nlk.argmax(axis=1)

orig_genotype_prob = 1 - (10**olk[orig_gt_idx] / 10**o_total)
np_genotype_prob = 1 - (10.**(nlk[:,np_gt_idx] - n_total))

orig_gq = -10 * math.log(orig_genotype_prob, 10)
np_gq = -10 * np.log10(np_genotype_prob, dtype='float128')


#with np.errstate(divide='ignore', invalid='ignore'):
#    genotype_quals = abs(-10 * np.log10(genotype_probs))
#    genotype_quals[ ~ np.isfinite(genotype_quals)] =  200
#
print n_total, o_total
