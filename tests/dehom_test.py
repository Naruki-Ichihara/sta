from sta.dehom import Fibers, generate_fiber_stl
from sta.analysis import compute_structure_tesnsor, compute_orientation
from sta.io import import_image_sequence
import numpy as np
import os

def test_generate_fiber_stl():
    volume = import_image_sequence("tests/test_images/test_",
                                           0,
                                           4,
                                           4,
                                           "tif")
    structure_tensor = compute_structure_tesnsor(volume, 10)
    theta, phi = compute_orientation(structure_tensor)

    fibers = Fibers()
    fibers.initialize(volume.shape, 10, 0.1)
    step_size = 10
    directions_x = step_size * np.tan(np.deg2rad(theta))[0]
    directions_y = step_size * np.tan(np.deg2rad(phi))[0]
    fibers.move_points(directions_x, directions_y)
    fibers.update_fiber(step_size, fibers.points)
    generate_fiber_stl(fibers, "tests/fibers.stl")
    assert os.path.exists("tests/fibers.stl"), "STL file was not created."
    os.remove("tests/fibers.stl")  # Clean up the generated STL file
    assert True  # If no exception is raised, the test passes