#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Feature selection by maximizing feature selection overlap across dataset
splits."""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.algorithms.featsel import FeatureSelection
from mvpa.misc.state import StateVariable


if __debug__:
    from mvpa.misc import debug


class OptimalOverlapThresholder(FeatureSelection):
    """Feature selection by maximizing feature selection overlap across splits.

    A `Splitter` is used to generate multiple splits of a dataset. For each
    split a `SensitivityAnalyzer` is used to compute sensitivity maps. All
    sensitivity maps are thresholded with some given thresholders (instances
    of `ElementSelector`) and the respective feature selection overlap across
    dataset splits is then computed for each thresholder.
    """
    sensitivities = StateVariable(enabled=False,
                                  doc="List of sensitivity maps for all splits")
    overlap_scores = StateVariable(enabled=False,
                                   doc="dictionary with several overlap scores")
    overlap_maps = StateVariable(enabled=False,
            doc="List of boolean feature overlap maps for each threshold")
    best_thresholder = StateVariable(doc="Thresholder instance causing the "
                                         "largest feature overlap")
    best_thresholder_id = StateVariable(doc="Id of the best thresholder")


    def __init__(self,
                 sensitivity_analyzer,
                 thresholders,
                 splitter,
                 overlap_thr=1.0,
                 overlap_crit='rel',
                 **kargs):
        """Cheap init.

        :Parameters:
            sensitivity_analyzer : SensitivityAnalyzer instance
                Used to compute a sensitivity map for each dataset split.
            thresholders: sequence of ElementSelector instances
                Feature selection overlap across dataset splits will be
                computed for every thresholder in this sequence and the
                best performing overlap will be determined.
            splitter: Splitter instance
                Used to generate dataset splits.
            overlap_thr: float [0,1]
                Minimum fraction of selections across splits to define an
                overlap. Default: 1.0, i.e. a feature has to be selected in
                each split.
            overlap_crit: str ('frel', 'fspread')
                Criterion used to determine the 'best' threshold. Available
                scores are:

                    'frel': Fraction of overlapping features relative to the
                            number of features selected in each split
                            (determined as average across splits)
                    'fspread': Fraction of overlapping features releative to
                            the total number of features selected in ANY split.

                Please note, that all score are computed and are available
                from the state variables (if enabled). This argument is only
                used to determine which score is used to decide what thresholder
                is best.
        """
        # base init first
        FeatureSelection.__init__(self, **kargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        self.__thresholders = thresholders
        self.__splitter = splitter
        self.__overlap_thr = overlap_thr
        self.__overlap_crit = overlap_crit


    def __call__(self, dataset, testdataset=None, callables=[]):
        """Perform thresholding on a dataset.

        :Parameters:
            dataset: Dataset instance
                This dataset is used to compute the feature selection overlap
                scores for all configured thresholders.
            testdataset: Dataset instance
                The optimal thresholding is finally also applied to this
                dataset (if present).
            callables: (not yet implemented)

        :Returns:
            A 2-tuple with:

                * `Dataset` instances for all features from `dataset` selected
                   by the best performing thresholder.
                * `Dataset instance with the same thresholding applied to
                  `testdataset` or `None` if no testdataset was supplied.
        """
        fold_maps = []

        # need to precharge state
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
