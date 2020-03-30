# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os


extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxcontrib.bibtex',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
source_suffix = '.rst'
master_doc = 'index'
project = 'parma'
year = '2019-2020'
author = 'Douglas Machado Vieira'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.1.2'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/dougmvieira/parma/issues/%s', '#'),
    'pr': ('https://github.com/dougmvieira/parma/pull/%s', 'PR #'),
}
html_theme = 'alabaster'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
