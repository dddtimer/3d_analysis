'''"""
Spatial Autocorrelation

This module contains functions for computing the spatial autocorrelation of a 3D mesh. The spatial
autocorrelation measures the similarity of local features in the mesh as a function of distance. 
It is calculated by computing the local color entropy at each vertex and then computing the 
semivariance of the entropy values between pairs of vertices at different distances.

This script requires the following packages to be installed:

    numpy
    vedo
    plotly
"""'''

# Standard library imports
import os
import sys
import traceback

# Third-party library imports
import numpy as np
from vedo import Mesh
import plotly.graph_objects as go

def single_bootstrap_iteration(args):
    """
    Compute the semivariances and geodesic distances for a single bootstrap iteration.

    Args:
        args (tuple): A tuple containing the following arguments:
            i (int): Index of the bootstrap iteration.
            points (np.ndarray): Array of shape (n_points, n_dimensions) containing the points.
            local_entropies (list): A list of local color entropies for each vertex in the mesh.
            radius (float): Radius used to compute the geodesic distances.
            subset_size (int): Size of the subset to use for the bootstrap.
        
        geodesic_distances (np.ndarray): A square matrix with geodesic distances between all pairs of vertices.

    Returns:
        tuple: A tuple containing the following elements:
            semivariances (numpy.ndarray): Array of shape (n_bins,) containing the semivariances.
            geodesic_distances (numpy.ndarray): Array of shape (n_distances,) containing the geodesic distances.
    """   
    i, points, faces, local_entropies, subset_size, seed = args
    np.random.seed(seed + i)
    subset_indices = np.random.choice(range(len(points)), size=subset_size, replace=False)
    subset_entropies = [local_entropies[idx] for idx in subset_indices]
    geodesic_distances=compute_geodesic_distances(subset_indices,points,faces)
    if len(geodesic_distances) == 0:
        print(f"Subset size of {subset_size} is too low, exiting...")
        sys.exit()
    semivariances_by_bin = compute_semivariance(geodesic_distances, subset_entropies, num_bins=10)
    return semivariances_by_bin

def compute_geodesic_distances(subset_indices, points, faces):   
    """
    Computes the geodesic distances between pairs of points in a subset of a mesh.

    This function takes a list of indices of points in a mesh and computes the pairwise
    geodesic distances between these points. The mesh is represented by a set of points
    and triangular faces that define the mesh's geometry. The function uses the vedo library
    to compute the geodesic distances between pairs of points.

    Args:
        subset_indices (list): A list of indices of points in the mesh.
        points (np.ndarray): An array of shape (n_points, n_dimensions) containing the coordinates
                             of all points in the mesh.
        faces (np.ndarray): An array of shape (n_faces, 3) containing the indices of the three points
                            that define each triangular face in the mesh.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_pairs, 3) containing the pairwise geodesic distances
                    between the points in the subset. Each row in the array represents a single pairwise
                    distance and contains three elements: the distance, the index of the first point,
                    and the index of the second point.
    """    
    def geodesic_length(geodesic_line):
        """
        Computes the length of a geodesic line between two points.

        This function takes a geodesic line between two points in a mesh and computes
        its length. The geodesic line is represented by a vedo PolyLine object.

        Args:
            geodesic_line (vedo.PolyLine): A PolyLine object representing the geodesic line.

        Returns:
            float: The length of the geodesic line.
        """       
        points = geodesic_line.points()
        length = 0
        for i in range(len(points) - 1):
            length += np.linalg.norm(points[i + 1] - points[i])
        return length
    mesh = Mesh([points, faces])

    # Initialize the geodesic distances list
    geodesic_distances = []

    # Iterate over pairs of points in the subset_indices to calculate pairwise geodesic distances
    for i in range(len(subset_indices)):
        for j in range(i + 1, len(subset_indices)):
            # Calculate the geodesic line between the two points
            geodesic_line = mesh.geodesic(subset_indices[i], subset_indices[j])
            # Calculate the length of the geodesic line
            dist = geodesic_length(geodesic_line)
            #print(f"Distance between {subset_indices[i]} and {subset_indices[j]}: {dist}")
            geodesic_distances.append((dist, i, j))

    # Convert the list of geodesic distances and indices to a 2D NumPy array
    geodesic_distances_array = np.array(geodesic_distances)

    return geodesic_distances_array

def compute_semivariance(geodesic_distances, subset_entropies, num_bins=10, min_points=5):
    """
    Computes the semivariance values for a given set of geodesic distances and local color entropies.

    Args:
        geodesic_distances (numpy.ndarray): Array of shape (n_distances, 3) containing the geodesic distances
                                            between all pairs of vertices in the subset.
        subset_entropies (list): A list of local color entropies for the subset of vertices.
        num_bins (int, optional): Number of bins to use for semivariance calculation. Defaults to 10.
        min_points (int, optional): Minimum number of pairs of points in each bin. Bins with fewer points are discarded.
                                    Defaults to 5.

    Returns:
        dict: A dictionary containing the semivariance values for each upper edge of the distance bins.
              Keys are the upper edge of the distance bin and values are the semivariance for that bin.
    """    
    max_distance = np.max(geodesic_distances[:, 0])
    min_distance = np.min(geodesic_distances[:, 0])

    range_diff = max_distance - min_distance
    bin_size = range_diff / num_bins

    semivariances_by_bin = [[] for _ in range(num_bins)]
    counts_by_bin = [0] * num_bins
    semivariance_count = 0
    semivariances_by_upper_bin = {}

    for row in geodesic_distances:
        distance, i, j = row
        semivariance = (subset_entropies[int(i)] - subset_entropies[int(j)])**2
        semivariance_count += 1
        bin_index = min(int((distance - min_distance) / bin_size), num_bins - 1)
        if bin_index < num_bins:
            semivariances_by_bin[bin_index].append(semivariance)
            counts_by_bin[bin_index] += 1

    semivariances = []
    for i in range(num_bins):
        if counts_by_bin[i] >= min_points:
            semivariance = np.sum(semivariances_by_bin[i]) / counts_by_bin[i]
            semivariances.append(semivariance)
            upper_edge = (i + 1) * bin_size + min_distance
            semivariances_by_upper_bin[upper_edge] = semivariance
    return semivariances_by_upper_bin


def plot_variogram(output_directory, mesh_file_prefix, bin_edges, semivariances, max_range=None, sem=None):
    """
    Generate a plot of the variogram.

    Args:
        output_directory (str): Path to the directory where the plot will be saved.
        mesh_file_prefix (str): Prefix of the mesh file name.
        bin_edges (list): A list containing the edges of each bin for the semivariances.
        semivariances (list): A list containing the semivariances for each bin.
        max_range (float, optional): Maximum range of the x-axis. Defaults to None.
        sem (list, optional): A list containing the standard errors of the mean for each bin. Defaults to None.

    Returns:
        plotly.graph_objs._figure.Figure: A plotly figure object.
    """    
    semivariances = np.array(semivariances)
    fig = go.Figure(data=go.Scatter(x=bin_edges, y=semivariances, mode='markers+lines'))

    #Add uncertainty estimate to the plot (SEM of bootstraps)
    if sem is not None:
        sem = np.array(sem)
        fig.add_trace(go.Scatter(x=bin_edges, y=semivariances - sem, 
                                 mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=bin_edges, y=semivariances + sem, 
                                 mode='lines', fill='tonexty', 
                                 fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), 
                                 showlegend=False))

    fig.update_layout(title="Variogram", xaxis_title="Distance", yaxis_title="Semivariance")
    
    if max_range is not None:
        fig.update_xaxes(range=[0, max_range])
        
    #Save the variogram to disk
    try:
        spatial_plots_dir = os.path.join(output_directory, "plots", "spatial")
        if not os.path.exists(spatial_plots_dir):
            os.makedirs(spatial_plots_dir)         
        output_plot = os.path.join(spatial_plots_dir, f"{mesh_file_prefix}.html")
        fig.write_html(output_plot)
        #fig.show()

    except (OSError, IOError)  as e:
        print(f"Error generating plot in plot_variogram() - skipping...")
        print(f"Error: {e}")
        traceback.print_exc()
    except RuntimeError as e:
        print(f"Cannot display variogram plot to default display - skipping...")
        print(f"Error: {e}")

    return fig
