#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Optimized classifier."""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.clfs.base import Classifier
from mvpa.misc.state import StateVariable
from mvpa.misc.param import Parameter
from mvpa.base import warning

if __debug__:
    from mvpa.base import debug

class OptimizedClassifier(Classifier):  # should be ProxyClassifier?
    """classifier which chooses hyper-parameters according to some criterion.

    """

    _clf_internals = [ 'meta' ]

    def __init__(self, clf, clfopt=None, params=None,
                 errorfx=None, train_optclf=False,
                 **kwargs):
        """Initialize OptimizedClassifier.

        :Parameters:
          clf : Classifier
            Target classifier to use
          clfopt : Classifier
            Parameters of which classifier to optimize. If None, use `clf`
          params : list of strings or tuples (string, dict)
            Definition of what parameters and how to optimize. If None,
            use pre-assigned default params and ranges.
            XXX: provide format + examples. Preliminary
            params=('C', ('tube_epsilon', {'grid' : [0.01, 0.1]}), ('gamma', {'init': 1.0}))
          errorfx : string or function (no arguments)
            Function to evaluate to obtain the value to optimize. It might
            need to carry action to trigger reevaluation (ie training of
            classifier) XXX
          errorfx_df : string or function (no arguments)
            Function to evaluate to obtain derivative of errorfx
          train_optclf : bool
            Either to call train for optclt # XXX may be no need and incorp
            into errorfx_df
        """

        """
        TODO: we might want to optimize parameters of multiple
        classifiers at once, ie clfopt+params should be joined in a single argument

        Add parameters for ways to optimize: openopt vs simple grid search etc
        """
        # init base class first
        super(OptimizedClassifier, self).__init__(**kwargs)

        self._clf = clf

        if clfopt is None:
            clfopt = clf
        self._clfopt = clfopt

        if errorfx is None:
            if clfopt.states.isKnown('confusion'):
                errorfx = 'lambda *args:clfopt.states.confusion.error'
                clfopt.states.enable('confusion')
            else:
                # XXX probably we would better want some CrossValidatedTransferError
                errorfx = 'lambda *args:clfopt.states.training_confusion.error'
                clfopt.states.enable('training_confusion')
        self._errorfx = errorfx

        self._train_optclf = train_optclf

        # last but not least ;)
        self._params = params


    def _train(self, dataset):
        """Train the classifier using `dataset` (`Dataset`).
        """

        

        raise NotImplementedError

    def _predict(self, data):
        """Predict the output for the provided data.
        """

        raise NotImplementedError
        return predictions


if __name__ == '__main__':
    from mvpa.clfs.svm import *
    from mvpa.clfs.transerror import TransferError
    from mvpa.datasets.splitter import NFoldSplitter
#    from mvpa.clfs import SplitClassifier
    from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
    from mvpa.misc.data_generators import *

    svm = sg.SVM(C=1.0)
    #sclf = SplitClassifier(clf_,
    cve = CrossValidatedTransferError(TransferError(svm), NFoldSplitter())
    oc = OptimizedClassifier(clf=svm,
                             errorfx='cve(dataset)',
                             params=('C'))
    ds = normalFeatureDataset(
        nlabels=2,
        nonbogus_features=[1,3],
        perlabel=30,
        nchunks=3,
        snr=4)

    oc.train(ds)

