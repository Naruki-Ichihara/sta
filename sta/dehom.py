import numpy as np
from scipy.stats.qmc import PoissonDisk
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree
import pyvista as pv

class Fibers:
    def __init__(self):
        """
        Initialize the Fibers class with default attributes.
        """
        self.points = None
        self.bounds = None
        self.fiber_diameter = None
        self.fiber_volume_fraction = None
        self.fiber = []  # [z position, points]
        self.trajectory = []

    def initialize(self, shape, fiber_diameter, fiber_volume_fraction, scale=1.0, seed=42):
        """
        Initialize the Fibers class with a given shape, fiber diameter, and volume fraction.
        The centers of the fibers are generated using a Poisson disk sampling method.

        Args:
            shape (tuple): Shape of the domain (z, y, x).
            fiber_diameter (float): Diameter of the fibers.
            fiber_volume_fraction (float): Volume fraction of the fibers.
            scale (float): Scale factor for the fiber diameter.
            seed (int): Random seed for reproducibility.
        """
        if not shape[1] == shape[2]:
            raise ValueError("Shape must be square")
        self.bounds = shape
        self.fiber_diameter = fiber_diameter
        self.fiber_volume_fraction = fiber_volume_fraction

        total_area = shape[1] * shape[2]
        fiber_area = np.pi / 4 * fiber_diameter**2
        num_fibers = int(total_area * fiber_volume_fraction / fiber_area)

        max_dim = shape[1]
        normalized_radius = fiber_diameter*scale / max_dim

        sampler = PoissonDisk(d=2, radius=normalized_radius, seed=seed)
        points = sampler.random(num_fibers) * max_dim
        self.points = points
        self.fiber.append([0, self.points])
        self.trajectory.append(self.points.copy())

    def update_fiber(self, position, points):
        """
        Update the fiber data with a new position and points.
        This is used to add new layers of fibers at different z positions.
        Args:
            position (float): The z position of the new layer.
            points (np.ndarray): The points of the new layer.
        """
        self.fiber.append([position, points])
        self.trajectory.append(points.copy())

    def move_points(self, directions_x, directions_y, update=True):
        """
        Move the points in the x and y directions based on the provided direction arrays.
        The points are then relaxed to avoid overlaps.

        Args:
            directions_x (np.ndarray): Array of x direction vectors.
            directions_y (np.ndarray): Array of y direction vectors.
            update (bool): Whether to update the points in the object.

        Returns:
            np.ndarray: The new positions of the points after moving and relaxing.
        """
        x = self.points[:, 0]
        y = self.points[:, 1]
        coords = np.array([y, x])
        dir_x = map_coordinates(directions_x, coords, order=1, mode='nearest')
        dir_y = map_coordinates(directions_y, coords, order=1, mode='nearest')
        directions = np.stack([dir_x, dir_y], axis=1)
        new_points = self.points + directions
        relaxed_points = self._relax_points(new_points, self.fiber_diameter)
        if update:
            self.points = relaxed_points
        return relaxed_points
    
    def _relax_points(self, points, d, iterations=100):
        points = np.array(points, dtype=float)
        for _ in range(iterations):
            tree = KDTree(points)
            pairs = tree.query_pairs(r=d)
            moved = np.zeros_like(points)
            for i, j in pairs:
                delta = points[j] - points[i]
                dist = np.linalg.norm(delta)
                if dist < 1e-5:
                    continue
                overlap = d - dist
                shift = (delta / dist) * (overlap / 2)
                moved[i] -= shift
                moved[j] += shift
            points += moved
            if np.all(np.linalg.norm(moved, axis=1) < 1e-3):
                break
        return points
    
def generate_fiber_stl(fibers: Fibers, path: str) -> None:
    """
    Generate an STL file for the fibers.

    Args:
        fibers (Fibers): The Fibers object containing fiber data.
        path (str): Path to save the STL file.
    """
    if fibers.points is None:
        raise ValueError("No points to generate STL")
    
    diameter = fibers.fiber_diameter
    
    n_fibers = fibers.fiber[0][1].shape[0]
    radius = diameter / 2

    tubes = []

    for i in range(n_fibers):
        fiber_path = []
        for z_index, (z, points) in enumerate(fibers.fiber):
            x, y = points[i]
            fiber_path.append([x, y, z])
        fiber_path = np.array(fiber_path)

        poly = pv.Spline(fiber_path, n_points=len(fiber_path))
        tube = poly.tube(radius=radius, n_sides=3, capping=True)
        tubes.append(tube)

    full_mesh = tubes[0]
    for tube in tubes[1:]:
        full_mesh += tube

    full_mesh.save(path)
    print(f"STL saved: {path}")