# -*- coding: utf-8 -*-

"""Code for the Drug Explorer (DrugEx)."""
from pkg_resources import resource_filename

from drugex.util import Voc

VOC_DEFAULT = Voc(resource_filename('drugex', 'voc.txt'))
