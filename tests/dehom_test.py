import strong as st
import numpy as np
import os

def test_generate_fiber_stl():
    volume = st.import_image_sequence("tests/test_images/test_",
                                           0,
                                           4,
                                           4,
                                           "tif")
    structure_tensor = st.compute_structure_tensor(volume, 10)
    theta, phi = st.compute_orientation(structure_tensor)

    fibers = st.Fibers()
    fibers.initialize(volume.shape, 10, 0.1)
    step_size = 10
    directions_x = step_size * np.tan(np.deg2rad(theta))[0]
    directions_y = step_size * np.tan(np.deg2rad(phi))[0]
    fibers.move_points(directions_x, directions_y)
    fibers.update_fiber(step_size, fibers.points)
    mesh = st.generate_fiber_stl(fibers)
    mesh.save("tests/fibers.stl")
    # Check if the STL file was created successfully
    assert os.path.exists("tests/fibers.stl"), "STL file was not created."
    os.remove("tests/fibers.stl")  # Clean up the generated STL file
    assert True  # If no exception is raised, the test passes