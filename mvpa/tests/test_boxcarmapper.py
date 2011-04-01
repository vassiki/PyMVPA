# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Boxcar mapper"""


import numpy as np

from mvpa.testing.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true, assert_array_equal, assert_not_equal, SkipTest

from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.datasets import Dataset
from mvpa.mappers.flatten import FlattenMapper
from mvpa.mappers.base import ChainMapper
from mvpa.mappers.fx import mean_feature



def test_simpleboxcar():
    data = np.atleast_2d(np.arange(10)).T
    sp = np.arange(10)

    # check if stupid thing don't work
    assert_raises(ValueError, BoxcarMapper, sp, 0)

    # now do an identity transformation
    bcm = BoxcarMapper(sp, 1)
    trans = bcm.forward(data)
    # ,0 is a feature below, so we get explicit 2D out of 1D
    assert_array_equal(trans[:,0], data)

    # now check for illegal boxes
    if __debug__:
        # condition is checked only in __debug__
        assert_raises(ValueError, BoxcarMapper(sp, 2).train, data)

    # now something that should work
    nbox = 9
    boxlength = 2
    sp = np.arange(nbox)
    bcm = BoxcarMapper(sp, boxlength)
    trans = bcm.forward(data)
    # check that is properly upcasts the dimensionality
    assert_equal(trans.shape, (nbox, boxlength) + data.shape[1:])
    # check actual values, squeezing the last dim for simplicity
    assert_array_equal(trans.squeeze(), np.vstack((np.arange(9), np.arange(9)+1)).T)


    # now test for proper data shape
    data = np.ones((10,3,4,2))
    sp = [ 2, 4, 3, 5 ]
    trans = BoxcarMapper(sp, 4).forward(data)
    assert_equal(trans.shape, (4,4,3,4,2))

    # test reverse
    data = np.arange(240).reshape(10, 3, 4, 2)
    sp = [ 2, 4, 3, 5 ]
    boxlength = 2
    m = BoxcarMapper(sp, boxlength)
    m.train(data)
    mp = m.forward(data)
    assert_equal(mp.shape, (4, 2, 3, 4, 2))

    # try full reconstruct
    mr = m.reverse(mp)
    # shape has to match
    assert_equal(mr.shape, (len(sp) * boxlength,) + data.shape[1:])
    # only known samples are part of the results
    assert_true((mr >= 24).all())
    assert_true((mr < 168).all())

    # check proper reconstruction of non-conflicting sample
    assert_array_equal(mr[0].ravel(), np.arange(48, 72))

    # check proper reconstruction of samples being part of multiple
    # mapped samples
    assert_array_equal(mr[1].ravel(), np.arange(72, 96))

    # test reverse of a single sample
    singlesample = np.arange(48).reshape(2, 3, 4, 2)
    assert_array_equal(singlesample, m.reverse1(singlesample))
    # now in a dataset
    ds = Dataset([singlesample])
    assert_equal(ds.shape, (1,) + singlesample.shape)
    # after reverse mapping the 'sample axis' should vanish and the original 3d
    # shape of the samples should be restored
    assert_equal(ds.shape[1:], m.reverse(ds).shape)
    # multiple samples should just be concatenated along the samples axis
    ds = Dataset([singlesample, singlesample])
    assert_equal((np.prod(ds.shape[:2]),) + singlesample.shape[1:],
                 m.reverse(ds).shape)
    # should not work for shape mismatch, but it does work and is useful when
    # reverse mapping sample attributes
    #assert_raises(ValueError, m.reverse, singlesample[0])

    # check broadcasting of 'raw' samples into proper boxcars on forward()
    bc = m.forward1(np.arange(24).reshape(3, 4, 2))
    assert_array_equal(bc, np.array(2 * [np.arange(24).reshape(3, 4, 2)]))


def test_datasetmapping():
    # 6 samples, 4X2 features
    data = np.arange(48).reshape(6,4,2)
    ds = Dataset(data,
                 sa={'timepoints': np.arange(6),
                     'multidim': data.copy()},
                 fa={'fid': np.arange(4)})
    # with overlapping and non-overlapping boxcars
    startpoints = [0, 1, 4]
    boxlength = 2
    bm = BoxcarMapper(startpoints, boxlength, space='boxy')
    # train is critical
    bm.train(ds)
    mds = bm.forward(ds)
    assert_equal(len(mds), len(startpoints))
    assert_equal(mds.nfeatures, boxlength)
    # all samples attributes remain, but the can rotated/compressed into
    # multidimensional attributes
    assert_equal(sorted(mds.sa.keys()), ['boxy_onsetidx'] + sorted(ds.sa.keys()))
    assert_equal(mds.sa.multidim.shape,
            (len(startpoints), boxlength) + ds.shape[1:])
    assert_equal(mds.sa.timepoints.shape, (len(startpoints), boxlength))
    assert_array_equal(mds.sa.timepoints.flatten(),
                       np.array([(s, s+1) for s in startpoints]).flatten())
    assert_array_equal(mds.sa.boxy_onsetidx, startpoints)
    # feature attributes also get rotated and broadcasted
    assert_array_equal(mds.fa.fid, [ds.fa.fid, ds.fa.fid])
    # and finally there is a new one
    assert_array_equal(mds.fa.boxy_offsetidx, range(boxlength))

    # now see how it works on reverse()
    rds = bm.reverse(mds)
    # we got at least something of all original attributes back
    assert_equal(sorted(rds.sa.keys()), sorted(ds.sa.keys()))
    assert_equal(sorted(rds.fa.keys()), sorted(ds.fa.keys()))
    # it is not possible to reconstruct the full samples array
    # some samples even might show up multiple times (when there are overlapping
    # boxcars
    assert_array_equal(rds.samples,
                       np.array([[[ 0,  1], [ 2,  3], [ 4,  5], [ 6,  7]],
                                 [[ 8,  9], [10, 11], [12, 13], [14, 15]],
                                 [[ 8,  9], [10, 11], [12, 13], [14, 15]],
                                 [[16, 17], [18, 19], [20, 21], [22, 23]],
                                 [[32, 33], [34, 35], [36, 37], [38, 39]],
                                 [[40, 41], [42, 43], [44, 45], [46, 47]]]))
    assert_array_equal(rds.sa.timepoints, [0, 1, 1, 2, 4, 5])
    assert_array_equal(rds.sa.multidim, ds.sa.multidim[rds.sa.timepoints])
    # but feature attributes should be fully recovered
    assert_array_equal(rds.fa.fid, ds.fa.fid)

    # popular dataset configuration (double flatten + boxcar)
    cm= ChainMapper([FlattenMapper(), bm, FlattenMapper()])
    cm.train(ds)
    bflat = ds.get_mapped(cm)
    assert_equal(bflat.shape, (len(startpoints), boxlength * np.prod(ds.shape[1:])))
    # add attributes
    bflat.fa['testfa'] = np.arange(bflat.nfeatures)
    bflat.sa['testsa'] = np.arange(bflat.nsamples)
    # now try to go back
    bflatrev = bflat.mapper.reverse(bflat)
    # data should be same again, as far as the boxcars match
    assert_array_equal(ds.samples[:2], bflatrev.samples[:2])
    assert_array_equal(ds.samples[-2:], bflatrev.samples[-2:])
    # feature axis should match
    assert_equal(ds.shape[1:], bflatrev.shape[1:])


def test_boxcar_with_postproc():
    data = np.arange(20).reshape(5,4)
    ds = Dataset(data)
    ds.fa['fa_int'] = xrange(ds.nfeatures)
    startpoints = [0,2]
    boxlength = 2
    args = [startpoints, boxlength]
    kwargs = dict(space='boxy')
    bm = BoxcarMapper(*args, **kwargs)
    bm_mean = BoxcarMapper(*args,
                           postproc=mean_feature(), **kwargs)
    bm_mean_pt = BoxcarMapper(*args,
                              postproc=mean_feature(),
                              passthrough=True, **kwargs)
    bm.train(ds)
    ds_bm = ds.get_mapped(bm)
    #print `bm`, `bm_mean`, `bm_mean_pt`

    bm_mean.train(ds)
    # causes to fail!
    #ds_bm_mean = ds.get_mapped(bm_mean)
    ds_bm_mean = bm_mean(ds)

    bm_mean_pt.train(ds)
    #ds_bm_mean_pt = ds.get_mapped(bm_mean_pt)
    ds_bm_mean_pt = bm_mean_pt(ds)

    for ds_ in ds_bm, ds_bm_mean, ds_bm_mean_pt:
        continue
        print "---", ds_
        if 'mapper' in ds_.a:
            print ds_.O

    # basic tests
    for ds_ in ds_bm_mean, ds_bm_mean_pt:
        assert_array_equal(ds_.shape, (len(startpoints), ds.nfeatures))
        assert_array_equal(ds_.samples,
                           [[  2,   3,   4,   5,],
                            [ 10,  11,  12,  13,]])

    # weak atm due to always Falsing due to below TODOs
    assert_not_equal(ds.fa, ds_bm.fa)
    assert_not_equal(ds.fa, ds_bm_mean.fa)
    # TODO: fix up
    #  proper copying of collectables so  __doc__ is not lost
    #ds_bm_mean_pt.fa['fa_int'].__doc__ = ds.fa['fa_int'].__doc__
    # TODO: comparisons among collectables/collections
    #assert_equal(ds.fa, ds_bm_mean_pt.fa)
    # Basic manual tests
    def av(col):
        return [x.value for x in col.values()]
    # FAs
    assert_array_equal(ds.fa.keys(), ds_bm_mean_pt.fa.keys())
    assert_array_equal(av(ds.fa), av(ds_bm_mean_pt.fa))
    # SAs
    assert_array_equal(ds_bm_mean.sa.keys(), ds_bm_mean_pt.sa.keys())
    assert_array_equal(av(ds_bm_mean.sa), av(ds_bm_mean_pt.sa))
    assert_array_equal(av(ds_bm.sa), av(ds_bm_mean_pt.sa))
    assert_not_equal(av(ds.sa), av(ds_bm_mean_pt.sa))

    raise SkipTest("We need more")
