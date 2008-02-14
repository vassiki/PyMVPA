#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA recursive feature elimination"""

import unittest
import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.algorithms.anova import OneWayAnova
from mvpa.algorithms.featsel import FractionTailSelector
from mvpa.algorithms.optthreshold import OptimalOverlapThresholder
from mvpa.clfs.transerror import TransferError

from tests_warehouse import normalFeatureDataset, sweepargs
from tests_warehouse_clfs import *


class OptimalOverlapThresholderTests(unittest.TestCase):

    def getData(self):
        data = N.random.standard_normal((100, 20))
        labels = N.concatenate((N.repeat(0, 50),
                                N.repeat(1, 50)))
        chunks = N.repeat(range(5), 10)
        chunks = N.concatenate((chunks, chunks))
        return Dataset(samples=data, labels=labels, chunks=chunks)


    @sweepargs(l_clf=clfs['LinearSVMC'])
    def testFullThresholder(self, l_clf):
        # test full range of thresholding with 0.1 steps
        fractions = N.arange(0.0, 1.1, 0.1).tolist()
        thresholders = [ FractionTailSelector(i, mode='select', tail='upper')
                         for i in fractions ]

        state_keys = ['ovscores', 'ovstatmaps', 'sensitivities', 
                      'selected_ids']
        terr_keys = ['terr_ov', 'terr_nthr', 'terr_spread', 'terr_sthr',
                     'terr_nsthr']

        # configure OptimalOverlapThresholder with nfold + anova
        # fully state-enabled
        othr = OptimalOverlapThresholder(
                    OneWayAnova(), thresholders, NFoldSplitter(cvtype=1),
                    TransferError(l_clf),
                    enable_states=state_keys + terr_keys)

        # run on dataset
        data = self.getData()
        sdataset, stdataset = othr(data)

        # by default no test dataset
        self.failUnlessEqual(stdataset, None)

        # check score keys
        for k in othr.ovscores.keys():
            self.failUnless(k in ['fspread', 'fselected', 'fov'])
        # one overlap map per thresholder
        self.failUnlessEqual(N.array(othr.ovstatmaps).shape,
                             (len(thresholders), data.nfeatures))

#        # by definition full threshold gives full overlap
#        self.failUnlessEqual(othr.best_thresholder.felements, 1.0)
#        self.failUnlessEqual(othr.best_thresholder_id, len(thresholders)-1)
        self.failUnlessEqual(N.array(othr.sensitivities).shape,
                             (len(data.uniquechunks), data.nfeatures))
#        self.failUnless((othr.selected_ids == range(data.nfeatures)).all())

        # check score keys
        for k in terr_keys:
            terr = othr.states.get(k)
            # proper shape: one per thresholder
            self.failUnlessEqual(terr.shape, (len(thresholders),))
            # proper range
            self.failUnless((terr >= 0.0).all())
            self.failUnless((terr <= 1.0).all())



def suite():
    return unittest.makeSuite(OptimalOverlapThresholderTests)


if __name__ == '__main__':
    import test_runner

