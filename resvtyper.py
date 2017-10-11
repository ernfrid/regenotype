#!/usr/bin/env python

import signal

import click
from cyvcf2 import VCF, Writer
import numpy as np

def indiv_log_choose(n_minus_k, k):
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
    
log_choose = np.vectorize(indiv_log_choose)
genotype_strings = np.array([[0, 0, False], [0, 1, False], [1, 1, False], [-1,-1,False]])
dup_priors = np.array([1e-2, 1/5.0, 1/3.0], dtype='float128').reshape(1,3)
nondup_priors = np.array([1e-3, 0.5, 0.9], dtype='float128').reshape(1,3)


def regenotype(writer, variant):
    if variant.INFO.get('SVTYPE') == 'DUP': # specialized logic to handle non-destructive events such as duplications
        p_alt = dup_priors
    else:
        p_alt = nondup_priors

    ref = variant.format('QR', int)
    alt = variant.format('QA', int)

    # store which data have zero counts
    # These look like they will progress through below as homozygous reference sites
    no_data = np.logical_and(ref==0, alt==0).ravel()
    log_combo = log_choose(ref, alt)

    lk = log_combo + alt * np.log10(p_alt, dtype='float128') + ref * np.log10(1. - p_alt, dtype='float128') 
    best_gt = lk.argmax(axis=1)
    total_quals = np.logaddexp.reduce(lk * np.log(10., dtype='float128'), axis=1) * np.log10(np.e, dtype='float128')
    second_best_gt = np.ma.array(lk, mask=False, dtype='float128')
    second_best_gt.mask[np.arange(len(lk.argmax(axis=1))), lk.argmax(axis=1)] = True
    genotype_quals = -10. * (second_best_gt.max(axis=1) - lk.max(axis=1))
    sample_quals = -10. * (lk[:, 0] - total_quals)
    variant_sample_quals = np.ma.array(sample_quals, mask=(np.logical_or(best_gt == 0, no_data)))
    msq = np.mean(variant_sample_quals, dtype='float')
    nsamp = np.count_nonzero(best_gt)
    variant_qual = np.sum(sample_quals[~no_data]).round(2)

    # Set sites with no data to null genotypes
    # cyvcf2 doesn't provide a facility for setting null values so set to -1 below
    best_gt[no_data] = -1
    genotype_quals[no_data] = -1
    sample_quals[no_data] = -1
    variant.genotypes = genotype_strings[best_gt]
    variant.set_format('GQ', genotype_quals.astype(np.int))
    variant.set_format('SQ', sample_quals.astype(np.float64).round(2))
    variant.set_format('GL', lk.round(0).astype(np.int))
    variant.QUAL = variant_qual
    if 'MSQ' in variant.INFO:
        variant.INFO['MSQ'] = float(msq.round(2))
    if 'NSAMP' in variant.INFO:
        variant.INFO['NSAMP'] = nsamp
    writer.write_record(variant)

@click.command()
@click.argument('vcf', type=click.Path())
def main(vcf):
    reader = VCF(vcf)
    writer = Writer('-', reader)

    for variant in reader:
        regenotype(writer, variant)
    writer.close()

if __name__ == '__main__':
    # Setting this allows the program to work
    # properly in a unix pipeline
    # unclear to me why it doesn't without this.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
