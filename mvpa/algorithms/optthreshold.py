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
from mvpa.clfs.transerror import TransferError
from mvpa.misc.state import StateVariable
from mvpa.misc import verbose

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
    sensitivities = \
        StateVariable(enabled=False,
            doc="List of sensitivity maps for all splits")
    fselected = \
        StateVariable(enabled=True,
            doc="Fractions of features thresholded in each split for every " \
                "thresholder.")
    fspread = \
        StateVariable(enabled=True,
            doc="Fractions of features selected in any split, but not " \
                "overlapping for every thresholder.")
    fov = \
        StateVariable(enabled=True,
            doc="Fractions of features overlapping across splits for every " \
                "thresholder.")
    ovstatmaps = \
        StateVariable(enabled=False,
            doc="List of feature overlap maps for each threshold. Each "
                "element shows the fraction of selections for the "
                "respective feature across splits.")
    terr_ov = \
        StateVariable(enabled=False,
            doc="Thresholderwise transfer error for overlapping features.")
    terr_nthr = \
        StateVariable(enabled=False,
            doc="Thresholderwise transfer error for features never " \
                "thresholded.")
    terr_spread = \
        StateVariable(enabled=False,
            doc="Thresholderwise transfer error for features thresholded at " \
                "least once but not always.")
    terr_sthr = \
        StateVariable(enabled=False,
            doc="Thresholderwise transfer error for features thresholded in " \
                "the respective split.")
    terr_nsthr = \
        StateVariable(enabled=False,
            doc="Thresholderwise transfer error for features not " \
                "thresholded in the respective split.")
#    best_thresholder = StateVariable(doc="Thresholder instance causing the "
#                                         "largest feature overlap")
#    best_thresholder_id = StateVariable(doc="Id of the best thresholder")


    def __init__(self,
                 sensitivity_analyzer,
                 thresholders,
                 splitter,
                 transerror,
                 overlap_thr=1.0,
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
            transerror: TransferError instance
                For each datasplit the transfer error is computed for all and
                overlapping-only features using this object.
            overlap_thr: float [0,1]
                Minimum fraction of selections across splits to define an
                overlap. Default: 1.0, i.e. a feature has to be selected in
                each split.
        """
        # base init first
        FeatureSelection.__init__(self, **kargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        self.__thresholders = thresholders
        self.__splitter = splitter
        self.__transerror = transerror
        self.__overlap_thr = overlap_thr


    def computeSelectionMaps(self, dataset):
        """Compute the boolean selection maps for all splits and thresholders.

        Returns:
            ndarray: (splits x thresholders x features), dtype='bool'
        """
        fold_maps = []

        # need to precharge state
        self.sensitivities = []

        for nfold, (working_ds, validation_ds) in \
                enumerate(self.__splitter(dataset)):

            if __debug__:
                debug('OTHRC', 'Compute sensitivities for split %i.' %nfold)

            # compute sensitivities for the working dataset
            sensitivities = self.__sensitivity_analyzer(working_ds)

            if self.states.isEnabled("sensitivities"):
                self.sensitivities.append(sensitivities)

            # compute selection maps for each thresholder
            thr_maps = []

            for i, thr in enumerate(self.__thresholders):
                thr_maps.append(
                    dataset.convertFeatureIds2FeatureMask(thr(sensitivities)))

            fold_maps.append(thr_maps)

        # need array for easy access
        return N.array(fold_maps, dtype='bool')


    def computeOverlap(self, smaps):
        """Compute feature selection overlap across splits and associated
        scores.

        Returns:
            ndarray: (thresholders x features)
                Fraction of selections per thresholder and feature.
            dict: keys=['fselected', 'fspread', 'fov']
                Each key points to a list with one entry per thresholder.

                  :fselected: fraction of features selected in *each* split
                  :fspread: fraction of features selected in *any* split, but
                            not overlapping
                  :fov: fraction of features selected in *all* splits
        """
        # maps of overlapping features (thresholders x features)
        ovstatmaps = []

        ovscores = {'fselected': [],
                     'fspread': [],
                     'fov': []}

        # computer feature overlap per thresholder and across splits
        for i, thr in enumerate(self.__thresholders):
            # fraction of selected features in each split
            # computed as mean over splits, but just to deal with e.g. tricky
            # fractions
            fselected = N.mean(smaps[:,i])

            # fraction of selections per feature
            ovstatmap = N.mean(smaps[:,i], axis=0)

            # fraction of features selected in any of the splits, but not
            # overlapping
            fspread = N.mean(N.logical_and(ovstatmap > 0.0,
                                           ovstatmap < 1.0))

            # fraction of overlapping features
            foverlap = N.mean(ovstatmap >= self.__overlap_thr)

            # store stuff
            ovstatmaps.append(ovstatmap)
            ovscores['fselected'].append(fselected)
            ovscores['fspread'].append(fspread)
            ovscores['fov'].append(foverlap)

        return ovstatmaps, ovscores


    def __storeTransferError(self, container, key, split, wds, vds,
                             feature_ids):
        """Incrementally store transfer error values. Evil! Do not use from
        outside.
        """
        if not container.has_key(key):
            container[key] = []

        if len(container[key]) == split:
            container[key].append([])

        container[key][split].append(
            self.__transerror(vds.selectFeatures(feature_ids),
                              wds.selectFeatures(feature_ids)))


    def computeTransferErrors(self, dataset, smaps, ovstatmaps, ovscores):
        """Compute cross-validated transfer errors for various feature set.

        Each transfer error has to be enable via the `Stateful` interface.

        :Returns:
            dict: keys == state names
        """
        # transfer errors: (measure: split x thresholder)
        terrs = {}

        # split
        for nfold, (wds, vds) in enumerate(self.__splitter(dataset)):
            if __debug__:
                debug('OTHRC', "Compute transfer error(s) for split (%i/%i)" \
                               % (nfold, smaps.shape[0]))

            # for every thresholder
            for i, thr in enumerate(self.__thresholders):
                if self.states.isEnabled("terr_sthr"):
                    self.__storeTransferError(
                        terrs, 'terr_sthr', nfold, wds, vds,
                        dataset.convertFeatureMask2FeatureIds(
                                            smaps[nfold, i] == True))
                if self.states.isEnabled("terr_nsthr"):
                    self.__storeTransferError(
                        terrs, 'terr_nsthr', nfold, wds, vds,
                        dataset.convertFeatureMask2FeatureIds(
                                            smaps[nfold, i] == False))
                if self.states.isEnabled("terr_ov"):
                    self.__storeTransferError(
                        terrs, 'terr_ov', nfold, wds, vds,
                        dataset.convertFeatureMask2FeatureIds(
                                    ovstatmaps[i] >= self.__overlap_thr))
                if self.states.isEnabled("terr_nthr"):
                    self.__storeTransferError(
                        terrs, 'terr_nthr', nfold, wds, vds,
                        dataset.convertFeatureMask2FeatureIds(
                                    ovstatmaps[i] == 0))
                if self.states.isEnabled("terr_spread"):
                    self.__storeTransferError(
                        terrs, 'terr_spread', nfold, wds, vds,
                        dataset.convertFeatureMask2FeatureIds(
                            N.logical_and(ovstatmaps[i] > 0.0,
                                          ovstatmaps[i] < 1.0)))

        # mean across splits for all computed transfer errors
        for k, v in terrs.iteritems():
            avg = N.mean(v, axis=0)

            # store to state
            self.states.set(k, avg)

            # prepare return value
            terrs[k] = avg

        return terrs


    def __call__(self, dataset, testdataset=None):
        """Perform thresholding on a dataset.

        :Parameters:
            dataset: Dataset instance
                This dataset is used to compute the feature selection overlap
                scores for all configured thresholders.
            testdataset: Dataset instance
                The optimal thresholding is finally also applied to this
                dataset (if present).

        :Returns:
            A 2-tuple with:

                * `Dataset` instances for all features from `dataset` selected
                   by the best performing thresholder.
                * `Dataset instance with the same thresholding applied to
                  `testdataset` or `None` if no testdataset was supplied.
        """
        verbose(1, "Determine optimal overlap threshold.")
        verbose(4, "Compute sensitivities and store selection maps for all " \
                   "thresholders.")
        # (splits x thresholders x features)
        smaps = self.computeSelectionMaps(dataset)

        verbose(4, "Determine overlapping features and associated stats.")
        ovstatmaps, ovscores = self.computeOverlap(smaps)

        verbose(4, "Compute various transfer errors.")
        # (thresholders x (a,b,c,d))
        terrs = self.computeTransferErrors(dataset,
                                           smaps,
                                           ovstatmaps,
                                           ovscores)

        # determine largest overlap and associated thresholder
#        best_thr_id = N.array(thr_scores[self.__overlap_crit]).argmax()
#        selected_ids = overlap_maps[best_thr_id].nonzero()[0]

        # charge state
        self.fov = N.array(ovscores['fov'])
        self.fspread = N.array(ovscores['fspread'])
        self.fselected = N.array(ovscores['fselected'])
        self.ovstatmaps = N.array(ovstatmaps)
#
#        self.best_thresholder_id = best_thr_id
#        self.best_thresholder = self.__thresholders[best_thr_id]
#
#        self.selected_ids = selected_ids
#
#        # and select overlapping feature from original dataset(s)
#        if testdataset:
#            results = (dataset.selectFeatures(selected_ids),
#                       testdataset.selectFeatures(selected_ids))
#        else:
#            results = (dataset.selectFeatures(selected_ids), None)
#
#        return results
        # for now just return original input
        return dataset, testdataset
