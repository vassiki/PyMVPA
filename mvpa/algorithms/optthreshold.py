#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Hmm, what is it?"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.algorithms.featsel import FeatureSelection
from mvpa.misc.state import StateVariable


if __debug__:
    from mvpa.misc import debug


class OptimalOverlapThresholder(FeatureSelection):
    """
    """
    sensitivities = StateVariable(enabled=False)
    overlap_scores = StateVariable(enabled=False)
    overlap_maps = StateVariable(enabled=False)
    best_thresholder = StateVariable()
    best_thresholder_id = StateVariable()

    def __init__(self,
                 sensitivity_analyzer,
                 thresholders,
                 splitter,
                 overlap_thr=1.0,
                 overlap_crit='rel',
                 **kargs):
        """
        :Parameters:
            sensitivity_analyzer : SensitivityAnalyzer instance
            thresholders: sequence of ElementSelector instances
            splitter: Splitter instance
            overlap_thr: float [0,1]
                Minimum fraction of selections across splits to define an
                overlap. Default: 1.0, i.e. a feature has to be selected in
                each split.
            overlap_crit: str ('frel', 'fspread')
        """
        # base init first
        FeatureSelection.__init__(self, **kargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        self.__thresholders = thresholders
        self.__splitter = splitter
        self.__overlap_thr = overlap_thr
        self.__overlap_crit = overlap_crit


    def __call__(self, dataset, testdataset=None, callables=[]):
        """
        :Parameters:
            dataset: Dataset instance
            testdataset: Dataset instance
            callables: (not yet implemented)
        """
        fold_maps = []

        if self.states.isEnabled("sensitivities"):
            self.sensitivities = []

        for nfold, (working_ds, validation_ds) in \
                enumerate(self.__splitter(dataset)):

            # compute sensitivities for the working dataset
            sensitivities = self.__sensitivity_analyzer(working_ds)

            if self.states.isEnabled("sensitivities"):
                self.sensitivities.append(sensitivities)

            # compute selection maps for each thresholder
            thr_maps = \
                [ dataset.convertFeatureIds2FeatureMask(thr(sensitivities))
                    for thr in self.__thresholders ]

            fold_maps.append(thr_maps)

        # need array for easy access
        # (splits x thresholders x features)
        fold_maps = N.array(fold_maps, dtype='bool')

        # maps of overlapping features (for each thresholder)
        overlap_maps = []

        overlap_scores = {'spread': [],
                          'rel': [],
                          'fspread': [],
                          'frel': []}

        # computer feature overlap per thresholder and across splits
        for i, thr in enumerate(self.__thresholders):
            # fraction of selected features in each split
            # computed as mean over splits, but just to deal with e.g. tricky
            # fractions
            fselected_eachsplit = N.mean(fold_maps[:,i])

            # fraction of selections per feature
            feature_overlap_stats = N.mean(fold_maps[:,i], axis=0)

            # fraction of features selected in any of the splits
            fselected_acrosssplits = N.mean(feature_overlap_stats > 0.0)

            # boolean map of overlapping features (as defined by threshold)
            overlap_map = feature_overlap_stats >= self.__overlap_thr

            # fraction of overlapping features
            foverlap = N.mean(overlap_map)

            # store stuff
            overlap_maps.append(overlap_map)
            # overlap wrt mean of selected features
            overlap_scores['rel'].append(fselected_eachsplit)
            overlap_scores['frel'].append(foverlap / fselected_eachsplit)
            # overlap wrt fraction of features selected in any split
            overlap_scores['spread'].append(fselected_acrosssplits)
            overlap_scores['fspread'].append(foverlap / fselected_acrosssplits)


        # determine largest overlap and associated thresholder
        best_thr_id = N.array(overlap_scores[self.__overlap_crit]).argmax()
        selected_ids = overlap_maps[best_thr_id].nonzero()[0]

        # charge state
        self.overlap_scores = overlap_scores
        self.overlap_maps = overlap_maps

        self.best_thresholder_id = best_thr_id
        self.best_thresholder = self.__thresholders[best_thr_id]

        self.selected_ids = selected_ids

        # and select overlapping feature from original dataset(s)
        if testdataset:
            results = (dataset.selectFeatures(selected_ids),
                       testdataset.selectFeatures(selected_ids))
        else:
            results = (dataset.selectFeatures(selected_ids), None)

        return results
