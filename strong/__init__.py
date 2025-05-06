#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. include:: ../README.md
"""

__version__ = "0.2.1"
__author__ = "Naruki Ichihara"
__email__ = "ichihara.naruki@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Naruki Ichihara"
__status__ = "Development"

from strong.analysis import drop_edges_3D, compute_structure_tensor, compute_orientation, compute_static_data
from strong.dehom import Fibers, generate_fiber_stl
from strong.io import import_image, import_dicom, import_image_sequence, trim_image
from strong.simulation import estimate_compression_strength, estimate_compression_strength_from_profile, MaterialParams