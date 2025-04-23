from sta.simulation import MaterialParams, estimate_compression_strength, estimate_compression_strength_from_profile
from sta.io import import_image_sequence
from sta.analysis import compute_structure_tesnsor, compute_orientation
import numpy as np

def test_estimate_compression_strength():
    
    # Create MaterialParams object
    material_params = MaterialParams(1, 1, 1, 1, 1, 1, 1)
    
    # Estimate compression strength
    compression_strength, ultimate_strain, superposition_axial_stress_array, axial_strain_array = estimate_compression_strength(1, 1, material_params)
    assert compression_strength > 0 and type(compression_strength) == np.float64
    assert ultimate_strain > 0 and type(ultimate_strain) == np.float64
    assert superposition_axial_stress_array.shape == (1001,)
    assert axial_strain_array.shape == (1001,)

def test_estimate_compression_strength_from_profile():
    volume = import_image_sequence("tests/test_images/test_",
                                           0,
                                           4,
                                           4,
                                           "tif")
    print("\nstarting to compute structure tensor...")
    material_params = MaterialParams(1, 1, 1, 1, 1, 1, 1)
    structure_tensor = compute_structure_tesnsor(volume, 10)
    axes = [1., 0., 0.]
    theta = compute_orientation(structure_tensor, axes)
    print("Done")
    compression_strength, ultimate_strain, superposition_axial_stress_array, axial_strain_array = estimate_compression_strength_from_profile(theta, material_params)
    assert compression_strength > 0 and type(compression_strength) == np.float64
    assert ultimate_strain > 0 and type(ultimate_strain) == np.float64
    assert superposition_axial_stress_array.shape == (1001,)
    assert axial_strain_array.shape == (1001,)