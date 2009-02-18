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

from mvpa.datasets.base import Dataset
from mvpa.datasets.splitters import NFoldSplitter, NoneSplitter
from mvpa.measures.anova import OneWayAnova
from mvpa.featsel.helpers import FractionTailSelector
from mvpa.featsel.optthreshold import OptimalOverlapThresholder
from mvpa.clfs.transerror import TransferError

from tests_warehouse import *
from tests_warehouse_clfs import *


class OptimalOverlapThresholderTests(unittest.TestCase):

    def getData(self):
        data = N.random.standard_normal((100, 10))
        labels = N.concatenate((N.repeat(0, 50),
                                N.repeat(1, 50)))
        chunks = N.repeat(range(5), 10)
        chunks = N.concatenate((chunks, chunks))
        return Dataset(samples=data, labels=labels, chunks=chunks)


    @sweepargs(l_clf=clfswh['linear', 'svm', '!meta'])
    def testFullThresholder(self, l_clf):
        # test full range of thresholding with 0.1 steps
        fractions = N.arange(0.0, 1.1, 0.1).tolist()
        thresholders = [ FractionTailSelector(i, mode='select', tail='upper')
                         for i in fractions ]

        ovscore_keys = ['fov', 'fselected', 'fspread']
        state_keys = ['ovstatmaps', 'sensitivities', 'selected_ids']
        terr_keys = ['terr_ov', 'terr_nthr', 'terr_spread', 'terr_sthr',
                     'terr_nsthr']

        # configure OptimalOverlapThresholder with nfold + anova
        # fully state-enabled
        othr = OptimalOverlapThresholder(
                    OneWayAnova(), thresholders, NFoldSplitter(cvtype=1),
                    TransferError(l_clf),
                    enable_states=state_keys + terr_keys + ovscore_keys)

        # run on dataset
        data = self.getData()
        sdataset, stdataset = othr(data)

        # by default no test dataset
        self.failUnlessEqual(stdataset, None)

        # one overlap map per thresholder
        self.failUnlessEqual(N.array(othr.ovstatmaps).shape,
                             (len(thresholders), data.nfeatures))

        self.failUnlessEqual(N.array(othr.sensitivities).shape,
                             (len(data.uniquechunks), data.nfeatures))

        # check score keys
        for k in terr_keys + ovscore_keys:
            state = othr.states.get(k)
            # proper shape: one per thresholder
            self.failUnlessEqual(state.shape, (len(thresholders),))
            # proper range
            self.failUnless((state >= 0.0).all())
            self.failUnless((state <= 1.0).all())



def suite():
    return unittest.makeSuite(OptimalOverlapThresholderTests)


if __name__ == '__main__':
    import runner

