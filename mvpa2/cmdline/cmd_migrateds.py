# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Migrate PyMVPA datasets from previous versions

Different actions might need to be performed while migrating datasets if
they were created with previous versions of PyMVPA and external dependencies.
Multiple migration commands might become available in the future.  ATM only
"strip_nibabel" action is available:

strip_nibabel
-------------

This action converts attributes into a portable form.  PyMVPA datasets that
contain NiBabel objects, such as image header instances,
as it was default prior PyMVPA 2.4, can suffer from NiBabel API changes that
result in the inability to re-load serialized datasets (e.g. from HDF5) with
newer versions of NiBabel.

This script converts these NiBabel internals into a simpler form that helps
process such datasets with a much wider range of NiBabel Versions, and
removes the need to have NiBabel installed for simply loading such a dataset.

If you run into this problem, you need access to NiBabel source code that
can still process your dataset. A reasonable guess is to try NiBabel
version 1.1. Here is a recipe:

Get the NiBabel repository from github and checkout the version you need::

  % git clone  https://github.com/nipy/nibabel /tmp/nibabel
  % cd /tmp/nibabel
  % git checkout 1.1.0

Now call this script with the path to your old NiBabel, the HDF5 file with
the to-be-converted datasets, and the output filename.

  % pymvpa2 migrateds -a strip_nibabel \
    --paths=/tmp/nibabel -i olddata.hdf5 -o newdata.hdf5

This tools support HDF5 files containing a single dataset, as well as
files with sequences of datasets.

If you have a more complicated conversion task, please inspect the function
``convert_dataset()`` in this script to aid a custom conversion.
"""

# magic line for manpage summary
# man: -*- % migrate PyMVPA datasets from previous versions

import sys

__docformat__ = 'restructuredtext'


def setup_parser(parser):
    from .helpers import parser_add_common_opt

    parser.add_argument('-p', '--paths',
                        help='colon-separated additional paths to be added to '
                             'sys.path')
    if __debug__:
        parser.add_argument('--debug', action='store_true',
                            help='list available debug channels')

    parser.add_argument(
            '-a', '--action', choices=['strip_nibabel'],
            default=None, required=True,
            help="""Available migration actions.  'strip_nibabel' -- converts
                    dataset attributes into portable form""")

    parser.add_argument(
            '-i', '--input', required=True,
            help="""path to a PyMVPA dataset file.  It can contain a list of
                    datasets to be migrated""")

    parser_add_common_opt(parser, 'output_file', required=True)

    return parser


def action_strip_nibabel(ds):
    from mvpa2.datasets.mri import _hdr2dict
    # only str class name is stored
    ds.a['imgtype'] = ds.a.imgtype.__name__
    # new dataset store the affine directly
    ds.a['imgaffine'] = ds.a.imghdr.get_best_affine()
    # dict(array) stores header info
    ds.a['imghdr'] = _hdr2dict(ds.a.imghdr)
    return ds


def run(args):
    old_sys_path = sys.path

    if args.paths:
        # need to happen before any other import
        sys.path = args.paths.split(':') + sys.path
        import nibabel
        reload(nibabel)

    try:
        from mvpa2.base.hdf5 import h5load, h5save
        src = h5load(args.input)

        if args.action == 'strip_nibabel':
            if isinstance(src, list) or isinstance(src, tuple):
                out = [action_strip_nibabel(s) for s in src]
            else:
                out = action_strip_nibabel(src)
        else:
            raise ValueError(args.action)

        h5save(args.output, out, compression=9)

    finally:
        # revert back happen this one was invoked from within a test or smth
        sys.path = old_sys_path
