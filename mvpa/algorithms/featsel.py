#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Feature selection base class and related stuff base classes and helpers."""

__docformat__ = 'restructuredtext'

import numpy as N

from math import floor
from numpy import arange

from mvpa.misc.vproperty import VProperty
from mvpa.misc import warning
from mvpa.misc.state import StateVariable, Statefull
from mvpa.misc.exceptions import UnknownStateError

if __debug__:
    from mvpa.misc import debug


class FeatureTransformation(Statefull):
    """Base class for anything which modifies each sample in some way

    XXX it sounds like there is some ambigouse overlap at least in
    definition with what mappers are there for... think about it. For
    now I am just adding mapper state variable since that is actually
    the tool which is 'crafted' by any FeatureTransformation

    Convention is that all `FeatureTransformation`'s are functors which
    are transforming the data given in the call arguments: 'dataset' and
    'testdataset'. But besides that they can create and store mapper
    which would apply computed transformation (previousely on
    'dataset' and might be using 'testdataset') on some new data.
    """

    mapper = StateVariable(enabled=False,
        doc="Generated mapper")

    def __call__(self, dataset, testdataset=None, callables=[]):
        """Functor function: all transformations must be functors
        """
        raise NotImplementedError


class RecursiveFeatureProjection(FeatureTransformation):
    """Transform features for each sample by projecting on a hyperplane

    XXX Should it be named RFP (to follow RFE naming ? ;-))
    XXX This is a sketch now -- it might transform wildly -- we might
    like to allow not only orthogonal/orthonormal spaces... may be
    eventually we make it work for non-linear boundaries (manifolds),
    so keep that in mind
    """

    errors = StateVariable(enabled=False,
                            doc="List of errors (if any) at each step")

    offsets = StateVariable(enabled=False,
                            doc="List of offsets (if any) at each step")

    normals = StateVariable(enabled=False,
                            doc="List of normals at each step")

    projections = StateVariable(enabled=False,
                            doc="List of projections done at each step")

    def __init__(self,
                 normal,
                 transfer_error=None,
                 **kwargs):
        """Initialize `FeatureProjection`

        :Parameters:
          normal
            Some functor which given the data would provide us with a
            normal to the hyperplane to project on. Optionally that
            functor might provide an offset fot the hyperplane (if it
            doesn't pass through the origin of coordinates) via state
            variable 'offset'.
            It might be simply a sensitivity analyzer of smth else
            (any functor pretty much)
            XXX might need to add bool flag to make that explicite
          transfer_error : TransferError
            used to compute the transfer error of a classifier based on a
            certain feature set on the test dataset.
            NOTE: If sensitivity analyzer is based on the same
            classifier as transfer_error is using, make sure you
            initialize transfer_error with train=False, otherwise
            it would train classifier twice without any necessity.
            NOTE: Default is None, so we stop only whenever we exhausted
            the space
          stopping_criterion : Functor
            Given a list of error values it has to return whether the
            criterion is fulfilled.
          train_clf : bool
            Flag whether the classifier in `transfer_error` should be
            trained before computing the error. In general this is
            required, but if the `sensitivity_analyzer` and
            `transfer_error` share and make use of the same classifier it
            can be switched off to save CPU cycles.

        """
        FeatureTransformation.__init__(self, **kargs)

        self.__normal = normal
        """Functor to define normal to the hyperplane"""

        self.__transfer_error = transfer_error
        """If defined - can be used by stopping_criterion"""

        self.__train_clf = train_clf
        """If defined - can be used by stopping_criterion"""


    def __call__(self, dataset, testdataset=None, callables=[]):
        """Functor call of `RecursiveFeatureProjection`

        Iteratevely obtain new normal, project data, repeat

        XXX It reminds RFE call very much (see below for more food for
        thoughts), so it might be that they could be kids of the same
        RecursiveFeatureTransformation of some kind?  It might be even
        closer to RFE than coded now -- we might have the same logic
        for selection of the best stopping point -- who knows if any
        given transfer_error is monotonically decreasing... might be
        not...
        YYY damn me... sure there is lots of common between RFE and
        this projection thingie! RFE is simply projecting to dimension
        orthogonal to those selected to be removed features, thus
        eliminating those features completely. With current projection
        scheme we alter all features which got involved in
        classification, ie which sensitivities are different from 0....
        """

        errors = []
        """Computed error for each tested projection. If no transfer_error
        were provided -- simply store None's so StoppingCriterion reacting
        on number of step could do its job."""

        self.normals = []
        """Computed normals"""

        self.projections = []
        """Computed normals"""


        self.offsets = []
        """If offsets are available"""

        wdataset = dataset
        """Operate on working dataset initially identical."""

        wtestdataset = testdataset
        """Same projection has to be performs on test dataset as well.
        This will hold the current testdataset."""

        # XXX some time we could come up with some projections which
        # wouldn't be creating basis, thus there could be more of them than
        # number of dimensions (ie # of features)
        while len(errors) < wdataset.nfeatures:
            # compute new normal
            normal = self.__normal(wdataset)

            if hasattr(normal, 'offset'):
                offset = normal.offset

            # we dedicate assumption about its normalization
            # if needed to itself

            if self.__transfer_error:
                # do not retrain clf if not necessary
                if self.__train_clf:
                    error = self.__transfer_error(wtestdataset, wdataset)
                else:
                    error = self.__transfer_error(wtestdataset, None)
            else:
                error = None

            # Check if it is time to stop
            stop = self.__stopping_criterion(errors)

            if __debug__:
                debug('RFP',
                      "Step %d: error=%s stop=%d " %
                      (len(errors), nfeatures, error, isthebest, stop,
                       len(selected_ids)))

            # stop if it is time to finish
            if stop: break

            # projection
            projection = 

            # compute projection of all samples to the hyperplane
            wdataset =


            # Record the error
            
            errors += error
            offsets += offset



        raise NotImplementedError       # yet not finished


class FeatureSelection(FeatureTransformation):
    """Base class for any feature selection - transformatioin where
    features are selected

    Base class for Functors which implement feature selection on the
    datasets.
    """

    selected_ids = StateVariable(enabled=False,
        doc="List of selected Ids")

#    def __init__(self, **kargs):
#        # base init first
#        FeatureTransformation.__init__(self, **kargs)


    def __call__(self, dataset, testdataset=None, callables=[]):
        """Invocation of the feature selection

        :Parameters:
          dataset : Dataset
            dataset used to select features
          testdataset : Dataset
            dataset the might be used to compute a stopping criterion
          callables : sequence
            a list of functors to be called with locals()

        Returns a tuple with the dataset containing the selected features.
        If present the tuple also contains the selected features of the
        test dataset. Derived classes must provide interface to access other
        relevant to the feature selection process information (e.g. mask,
        elimination step (in RFE), etc)
        """
        raise NotImplementedError



#
# Functors to be used for FeatureSelection
#

class BestDetector(object):
    """Determine whether the last value in a sequence is the best one given
    some criterion.
    """
    def __init__(self, func=min, lastminimum=False):
        """Initialize with number of steps

        :Parameters:
            fun : functor
                Functor to select the best results. Defaults to min
            lastminimum : bool
                Toggle whether the latest or the earliest minimum is used as
                optimal value to determine the stopping criterion.
        """
        self.__func = func
        self.__lastminimum = lastminimum
        self.__bestindex = None
        """Stores the index of the last detected best value."""


    def __call__(self, errors):
        """Returns True if the last value in `errors` is the best or False
        otherwise.
        """
        isbest = False

        # just to prevent ValueError
        if len(errors)==0:
            return isbest

        minerror = self.__func(errors)

        if self.__lastminimum:
            # make sure it is an array
            errors = N.array(errors)
            # to find out the location of the minimum but starting from the
            # end!
            minindex = N.array((errors == minerror).nonzero()).max()
        else:
            minindex = errors.index(minerror)

        self.__bestindex = minindex

        # if minimal is the last one reported -- it is the best
        if minindex == len(errors)-1:
            isbest = True

        return isbest

    bestindex = property(fget=lambda self:self.__bestindex)



class StoppingCriterion(object):
    """Base class for all functors to decide when to stop RFE (or may
    be general optimization... so it probably will be moved out into
    some other module
    """

    def __call__(self, errors):
        """Instruct when to stop.

        Every implementation should return `False` when an empty list is
        passed as argument.

        Returns tuple `stop`.
        """
        raise NotImplementedError



class MultiStopCrit(StoppingCriterion):
    """Stop computation if the latest error drops below a certain threshold.
    """
    def __init__(self, crits, mode='or'):
        """
        :Parameters:
            crits : list of StoppingCriterion instances
                For each call to MultiStopCrit all of these criterions will
                be evaluated.
            mode : any of ('and', 'or')
                Logical function to determine the multi criterion from the set
                of base criteria.
        """
        if not mode in ('and', 'or'):
            raise ValueError, \
                  "A mode '%s' is not supported." % `mode`

        self.__mode = mode
        self.__crits = crits


    def __call__(self, errors):
        """Evaluate all criteria to determine the value of the multi criterion.
        """
        # evaluate all crits
        crits = [ c(errors) for c in self.__crits ]

        if self.__mode == 'and':
            return N.all(crits)
        else:
            return N.any(crits)



class FixedErrorThresholdStopCrit(StoppingCriterion):
    """Stop computation if the latest error drops below a certain threshold.
    """
    def __init__(self, threshold):
        """Initialize with threshold.

        :Parameters:
            threshold : float [0,1]
                Error threshold.
        """
        StoppingCriterion.__init__(self)
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError, \
                  "Threshold %f is out of a reasonable range [0,1]." \
                    % `threshold`
        self.__threshold = threshold


    def __call__(self, errors):
        """Nothing special."""
        if len(errors)==0:
            return False
        if errors[-1] < self.__threshold:
            return True
        else:
            return False


    threshold = property(fget=lambda x:x.__threshold)



class NStepsStopCrit(StoppingCriterion):
    """Stop computation after a certain number of steps.
    """
    def __init__(self, steps):
        """Initialize with number of steps.

        :Parameters:
            steps : int
                Number of steps after which to stop.
        """
        StoppingCriterion.__init__(self)
        if steps < 0:
            raise ValueError, \
                  "Number of steps %i is out of a reasonable range." \
                    % `steps`
        self.__steps = steps


    def __call__(self, errors):
        """Nothing special."""
        if len(errors) >= self.__steps:
            return True
        else:
            return False


    steps = property(fget=lambda x:x.__steps)



class NBackHistoryStopCrit(StoppingCriterion):
    """Stop computation if for a number of steps error was increasing
    """

    def __init__(self, bestdetector=BestDetector(), steps=10):
        """Initialize with number of steps

        :Parameters:
            bestdetector : BestDetector instance
                used to determine where the best error is located.
            steps : int
                How many steps to check after optimal value.
        """
        StoppingCriterion.__init__(self)
        if steps < 0:
            raise ValueError, \
                  "Number of steps (got %d) should be non-negative" % steps
        self.__bestdetector = bestdetector
        self.__steps = steps


    def __call__(self, errors):
        stop = False

        # just to prevent ValueError
        if len(errors)==0:
            return stop

        # charge best detector
        self.__bestdetector(errors)

        # if number of elements after the min >= len -- stop
        if len(errors) - self.__bestdetector.bestindex > self.__steps:
            stop = True

        return stop

    steps = property(fget=lambda x:x.__steps)



class ElementSelector(Statefull):
    """Base class to implement functors to select some elements based on a
    sequence of values.
    """
    def __init__(self):
        """Cheap initialization.
        """
        Statefull.__init__(self)


    def __call__(self, seq):
        """Implementations in derived classed have to return a list of selected
        element IDs based on the given sequence.
        """
        raise NotImplementedError



class TailSelector(ElementSelector):
    """Select elements from a tail of a distribution.

    The default behaviour is to discard the lower tail of a given distribution.
    """

    ndiscarded = StateVariable(True,
        doc="Store number of discarded elements.")


    # TODO: 'both' to select from both tails
    def __init__(self, tail='lower', mode='discard', sort=True):
        """Initialize TailSelector

        :Parameters:
           tail : ['lower', 'upper']
              Choose the tail to be processed.
           mode : ['discard', 'select']
              Decides whether to `select` or to `discard` features.
           sort : bool
              Flag whether selected IDs will be sorted. Disable if not
              necessary to save some CPU cycles.

        """
        ElementSelector.__init__(self)  # init State before registering anything

        self._setTail(tail)
        """Know which tail to select."""

        self._setMode(mode)
        """Flag whether to select or to discard elements."""

        self.__sort = sort


    def _setTail(self, tail):
        """Set the tail to be processed."""
        if not tail in ['lower', 'upper']:
            raise ValueError, "Unkown tail argument [%s]. Can only be one " \
                              "of 'lower' or 'upper'." % tail

        self.__tail = tail


    def _setMode(self, mode):
        """Choose `select` or `discard` mode."""

        if not mode in ['discard', 'select']:
            raise ValueError, "Unkown selection mode [%s]. Can only be one " \
                              "of 'select' or 'discard'." % mode

        self.__mode = mode


    def _getNElements(self, seq):
        """In derived classes has to return the number of elements to be
        processed given a sequence values forming the distribution.
        """
        raise NotImplementedError


    def __call__(self, seq):
        """Returns selected IDs.
        """
        # TODO: Think about selecting features which have equal values but
        #       some are selected and some are not
        len_seq = len(seq)
        # how many to select (cannot select more than available)
        nelements = min(self._getNElements(seq), len_seq)

        # make sure that data is ndarray and compute a sequence rank matrix
        # lowest value is first
        seqrank = N.array(seq).argsort()

        if self.__mode == 'discard' and self.__tail == 'upper':
            good_ids = seqrank[:-1*nelements]
            self.ndiscarded = nelements
        elif self.__mode == 'discard' and self.__tail == 'lower':
            good_ids = seqrank[nelements:]
            self.ndiscarded = nelements
        elif self.__mode == 'select' and self.__tail == 'upper':
            good_ids = seqrank[-1*nelements:]
            self.ndiscarded = len_seq - nelements
        else: # select lower tail
            good_ids = seqrank[:nelements]
            self.ndiscarded = len_seq - nelements

        # sort ids to keep order
        # XXX should we do here are leave to other place
        if self.__sort:
            good_ids.sort()

        return good_ids



class FixedNElementTailSelector(TailSelector):
    """Given a sequence, provide set of IDs for a fixed number of to be selected
    elements.
    """

    def __init__(self, nelements, *args, **kwargs):
        """Cheap initialization.

        :Parameters:
          nselect : int
            Number of elements to select/discard.
        """
        TailSelector.__init__(self, *args, **kwargs)
        self._setNElements(nelements)


    def __repr__(self):
        return "%s number=%f" % (
            TailSelector.__repr__(self), self.__nselect)


    def _getNElements(self, seq):
        return self.__nelements


    def _setNElements(self, nelements):
        if __debug__:
            if nelements <= 0:
                raise ValueError, "Number of elements less or equal to zero " \
                                  "does not make sense."

        self.__nelements = nelements


    nelements = property(fget=lambda x:x.__nelements,
                         fset=_setNElements)



class FractionTailSelector(TailSelector):
    """Given a sequence, provide Ids for a fraction of elements
    """

    def __init__(self, felements, **kargs):
        """Cheap initialization.

        :Parameters:
           felements : float (0,1.0]
              Fraction of elements to select/discard.
        """
        TailSelector.__init__(self, **kargs)
        self._setFElements(felements)


    def __repr__(self):
        return "%s fraction=%f" % (
            TailSelector.__repr__(self), self.__felements)


    def _getNElements(self, seq):
        num = int(floor(self.__felements * len(seq)))
        num = max(1, num)               # remove at least 1
        # no need for checks as base class will do anyway
        #return min(num, nselect)
        return num


    def _setFElements(self, felements):
        """What fraction to discard"""
        if felements > 1.0 or felements < 0.0:
            raise ValueError, \
                  "Fraction (%f) cannot be outside of [0.0,1.0]" \
                  % felements

        self.__felements = felements


    felements = property(fget=lambda x:x.__felements,
                         fset=_setFElements)


#
# Particular implementations of FeatureSelection
#
# RFE  lives in a separate file rfe.py for now ;-)
#
class SensitivityBasedFeatureSelection(FeatureSelection):
    """Feature elimination.

    A `SensitivityAnalyzer` is used to compute sensitivity maps given a certain
    dataset. These sensitivity maps are in turn used to discard unimportant
    features.
    """

    sensitivity = StateVariable(enabled=False)

    def __init__(self,
                 sensitivity_analyzer,
                 feature_selector=FractionTailSelector(0.05),
                 **kargs
                 ):
        """Initialize feature selection

        :Parameters:
          sensitivity_analyzer : SensitivityAnalyzer
            sensitivity analyzer to come up with sensitivity
          feature_selector : Functor
            Given a sensitivity map it has to return the ids of those
            features that should be kept.

        """

        # base init first
        FeatureSelection.__init__(self, **kargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        """Sensitivity analyzer to use once"""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features."""



    def __call__(self, dataset, testdataset=None, callables=[]):
        """Select the most important features

        :Parameters:
          dataset : Dataset
            used to compute sensitivity maps
          testdataset: Dataset
            optional dataset to select features on

        Returns a tuple of two new datasets with selected feature
        subset of `dataset`.
        """

        sensitivity = self.__sensitivity_analyzer(dataset)
        """Compute the sensitivity map."""

        self.sensitivity = sensitivity

        # Select features to preserve
        selected_ids = self.__feature_selector(sensitivity)

        # Create a dataset only with selected features
        wdataset = dataset.selectFeatures(selected_ids)

        if not testdataset is None:
            wtestdataset = testdataset.selectFeatures(selected_ids)
        else:
            wtestdataset = None

        # Differ from the order in RFE when actually error reported is for
        results = (wdataset, wtestdataset)

        # provide evil access to internals :)
        for callable_ in callables:
            callable_(locals())

        # WARNING: THIS MUST BE THE LAST THING TO DO ON selected_ids
        selected_ids.sort()
        self.selected_ids = selected_ids

        # dataset with selected features is returned
        return results



class FeatureSelectionPipeline(FeatureSelection):
    """Feature elimination through the list of FeatureSelection's.

    Given as list of FeatureSelections it applies them in turn.
    """

    nfeatures = StateVariable(
        doc="Number of features before each step in pipeline")
    # TODO: may be we should also append resultant number of features?

    def __init__(self,
                 feature_selections,
                 **kargs
                 ):
        """Initialize feature selection pipeline

        :Parameters:
          feature_selections : lisf of FeatureSelection
            selections which to use. Order matters
        """
        # base init first
        FeatureSelection.__init__(self, **kargs)

        self.__feature_selections = feature_selections
        """Selectors to use in turn"""


    def __call__(self, dataset, testdataset=None, **kwargs):
        """Invocation of the feature selection

        TODO: not clear what was to do with callables -- pass inside
              or process locally. For now just pass inside
        """
        wdataset = dataset
        wtestdataset = testdataset

        self.selected_ids = None

        self.nfeatures = []
        """Number of features at each step (before running selection)"""

        for fs in self.__feature_selections:

            # enable selected_ids state if it was requested from this class
            fs.states._changeTemporarily(
                enable_states=["selected_ids"], other=self)
            if self.states.isEnabled("nfeatures"):
                self.nfeatures.append(wdataset.nfeatures)

            if __debug__:
                debug('FSPL', 'Invoking %s on (%s, %s)' %
                      (fs, wdataset, wtestdataset))
            wdataset, wtestdataset = fs(wdataset, wtestdataset, **kwargs)

            if self.states.isEnabled("selected_ids"):
                if self.selected_ids == None:
                    self.selected_ids = fs.selected_ids
                else:
                    self.selected_ids = self.selected_ids[fs.selected_ids]

            fs.states._resetEnabledTemporarily()

        return (wdataset, wtestdataset)

    feature_selections = property(fget=lambda self:self.__feature_selections,
                                  doc="List of `FeatureSelections`")
