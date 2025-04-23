import numpy as np
import numba
import numba_progress
from skimage.feature import structure_tensor

def compute_structure_tesnsor(volume: np.ndarray, noise_scale: int, mode: str = 'nearest') -> np.ndarray:
    """
    Computes the structure tensor of a 3D volume. Based on the skimage feature structure_tensor function.

    Args:
        volume (np.ndarray): The input 3D volume.
        noise_scale (int): The scale for the Gaussian filter.
        mode (str): The mode for the Gaussian filter. Default is 'nearest'.

    Returns:
        np.ndarray: The structure tensor of the input volume.
    
    Raises:
        ValueError: If the input volume is not 3D.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be 3D.")
    tensor_list = structure_tensor(volume, sigma=noise_scale, mode=mode)
    tensors = np.empty((6, *tensor_list[0].shape), dtype=np.float32)
    for i, tensor in enumerate(tensor_list):
        tensors[i] = tensor
    return tensors

@numba.njit(parallel=True, cache=True)
def _orientation_function(structureTensor, progressProxy):
    symmetricComponents3d = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    theta = np.zeros(structureTensor.shape[1:], dtype="<f4")
    phi = np.zeros(structureTensor.shape[1:], dtype="<f4")

    for z in numba.prange(0, structureTensor.shape[1]):
        for y in range(0, structureTensor.shape[2]):
            for x in range(0, structureTensor.shape[3]):
                structureTensorLocal = np.empty((3, 3), dtype="<f4")
                for n, [i, j] in enumerate(symmetricComponents3d):
                    structureTensorLocal[i, j] = structureTensor[n, z, y, x]
                    if i != j:
                        structureTensorLocal[j, i] = structureTensor[n, z, y, x]

                w, v = np.linalg.eig(structureTensorLocal)
                m = np.argmin(w)

                selectedEigenVector = v[:, m]

                if selectedEigenVector[0] < 0:
                    selectedEigenVector *= -1

                theta[z, y, x] = np.rad2deg(np.arctan2(selectedEigenVector[2], selectedEigenVector[0]))
                phi[z, y, x] = np.rad2deg(np.arctan2(selectedEigenVector[1], selectedEigenVector[0]))

        progressProxy.update(1)

    return theta, phi

@numba.njit(parallel=True, cache=True)
def _orientation_function_reference(structureTensor, progressProxy, reference_vector):

    symmetricComponents3d = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    theta = np.zeros(structureTensor.shape[1:], dtype="<f4")
    axial_vec = np.array(reference_vector, dtype="<f4")

    for z in numba.prange(0, structureTensor.shape[1]):
        for y in range(0, structureTensor.shape[2]):
            for x in range(0, structureTensor.shape[3]):
                structureTensorLocal = np.empty((3, 3), dtype="<f4")
                for n, [i, j] in enumerate(symmetricComponents3d):
                    structureTensorLocal[i, j] = structureTensor[n, z, y, x]
                    if i != j:
                        structureTensorLocal[j, i] = structureTensor[n, z, y, x]

                w, v = np.linalg.eig(structureTensorLocal)
                m = np.argmin(w)

                selectedEigenVector = v[:, m]

                if selectedEigenVector[0] < 0:
                    selectedEigenVector *= -1

                theta[z, y, x] = np.rad2deg(np.arccos(np.dot(selectedEigenVector, axial_vec)))

        progressProxy.update(1)

    return theta

def compute_orientation(structure_tensor, reference_vector=None):
    """ Compute orientation function.
    Args:
        structureTensor (np.ndarray): Structure tensor.
    Returns:
        tuple: Orientation angles.
    """
    if reference_vector is None:
        with numba_progress.ProgressBar(total=structure_tensor.shape[1]) as progress:
            theta, phi = _orientation_function(structure_tensor, progress)
        return theta, phi
    else:
        with numba_progress.ProgressBar(total=structure_tensor.shape[1]) as progress:
            theta = _orientation_function_reference(structure_tensor, progress, reference_vector)
        return theta