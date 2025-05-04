# -*- coding: utf-8 -*-
import strong as st
import time

def test_compute_structure_tensor():
    volume = st.import_image_sequence("tests/test_images/test_",
                                           0,
                                           4,
                                           4,
                                           "tif")
    structure_tensor = st.compute_structure_tensor(volume, 10)
    assert structure_tensor.shape == (6, 5, 1024, 1024)

def test_compute_orientation():
    volume = st.import_image_sequence("tests/test_images/test_",
                                           0,
                                           4,
                                           4,
                                           "tif")
    print("\nstarting to compute structure tensor...")
    start = time.time()
    structure_tensor = st.compute_structure_tensor(volume, 10)
    end = time.time()
    print(f"Time taken to compute structure tensor: {end - start} seconds")
    print("starting to compute orientation...")
    start = time.time()
    theta, phi = st.compute_orientation(structure_tensor)
    end = time.time()
    print(f"Time taken to compute orientation: {end - start} seconds")
    assert theta.shape == (5, 1024, 1024)
    assert phi.shape == (5, 1024, 1024)

    axes = [1., 0., 0.]
    print("starting to compute orientation for axial...")
    theta = st.compute_orientation(structure_tensor, axes)
    print("Done")
    assert theta.shape == (5, 1024, 1024)