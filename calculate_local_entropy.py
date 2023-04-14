'''"""
calculate_local_entropy.py contains functions for calculating the local color entropy of a 3D mesh.

The function calculate_local_entropy() calculates the local color entropy for a single
vertex in the mesh. It finds neighboring vertices within a given radius using a k-d tree
and computes the entropy of their color distribution in HSV space. The local entropy is
calculated as the average of the entropies of the hue, saturation, and value channels.

The function calculate_local_entropies_parallel() calculates the local color entropy for
each vertex in the mesh in parallel using a Pool of worker processes. It splits the list of
vertex indices into batches and submits tasks to the pool, one for each batch of vertices.
The function returns a list of local entropies, one for each vertex in the mesh.

To use these functions, import this script into your project and call the
calculate_local_entropies_parallel() function with the appropriate arguments.

This script requires the following packages to be installed:

numpy
scipy
multiprocessing
"""'''

#Standard library imports
import time 

#Third-party library imports
import numpy as np
from scipy.stats import entropy
from multiprocessing import Pool

def calculate_local_entropy(vertex_colors_hsv, points, kd_tree, vertex_index, radius):
#    vertex_index, vertex_colors_hsv, points, kd_tree, radius = args
    """
    Calculates the local color entropy of a given vertex in the mesh.

    The function finds neighboring vertices within the specified radius using a
    k-d tree and computes the entropy of their color distribution in HSV space.
    The k-d tree is a data structure that allows for efficient search of points
    in a multidimensional space. In this case, it is used to find vertices in the
    3D mesh that are close to the given vertex. The local entropy is calculated
    as the average of the entropies of the hue, saturation, and value channels.

    Search for neighbouring vertices uses scipy.spatial.cKDTree, which can handle 3d point
    and takes a search radius as an argument.

    Entropy calculation steps:
     (1) Convert the histogram into a probability density function. 
         ---Divide the count of data points in each bin by the total number of data points and by the width of the bin. 
     (2) Calculate the entropy of the dataset using the Shannon entropy formula H(X) = -sum(p(x) * log2(p(x)))

    Args:
        vertex_colors_hsv (list): A list of vertex colors in HSV format.
        points: A list of points in the 3D model.
        kd_tree (scipy.spatial.cKDTree): A k-d tree built from the mesh's points.
        vertex_index (int): The index of the vertex for which to calculate the local entropy.
        radius (float, optional): The radius to search for neighboring vertices. Defaults to 0.1.

    Returns:
        float: The local color entropy for the given vertex.
    """
    # Find neighboring vertices within the specified radius using the k-d tree
    neighbors = kd_tree.query_ball_point(points[vertex_index], r=radius)

    # Collect colors of neighboring vertices
    neighbor_colors = [vertex_colors_hsv[j] for j in neighbors]

    # Calculate entropy for each color channel (hue, saturation, and value)
    # Uses scipy.stats.entropy() to calculate Shannon entropy
    entropy_h = entropy(np.histogram([c[0] for c in neighbor_colors], bins=10, range=(0, 1), density=True)[0]) #density=true for pdf
    entropy_s = entropy(np.histogram([c[1] for c in neighbor_colors], bins=10, range=(0, 1), density=True)[0])
    entropy_v = entropy(np.histogram([c[2] for c in neighbor_colors], bins=10, range=(0, 1), density=True)[0])

    # Calculate the average entropy of the three color channels
    avg_entropy = (entropy_h + entropy_s + entropy_v) / 3
    #print(f"{vertex_index} {avg_entropy}")
    return avg_entropy

def process_batch(batch_args):
    """
    Calculates the local color entropy for a batch of vertices in the mesh.

    This function takes a batch of vertices and calculates their local color entropy in parallel.
    It uses the calculate_local_entropy() function to compute the entropy for each vertex.

    Args:
        batch_args (tuple): A tuple of arguments for the function, including:
            - vertex_colors_hsv (list): A list of vertex colors in HSV format.
            - points: A list of points in the 3D model.
            - kd_tree (scipy.spatial.cKDTree): A k-d tree built from the mesh's points.
            - radius (float): The radius to search for neighboring vertices.
            - batch (list): A list of indices of the vertices in the batch.

    Returns:
        list: A list of tuples, where each tuple contains the index of a vertex and its corresponding
        local color entropy.
    """
    vertex_colors_hsv, points, kd_tree, radius, batch = batch_args
    # Calculate the local color entropy for each vertex in the batch using calculate_local_entropy()
    batch_result = [(i, calculate_local_entropy(vertex_colors_hsv, points, kd_tree, i, radius)) for i in batch]
    return batch_result

def print_progress(progress_queue, total):
    progress = 0
    while progress < total:
        progress = progress_queue.get()
        print(f'\rProgress: {progress}/{total}', end='')

def calculate_local_entropies_parallel(vertex_colors_hsv, points, kd_tree, radius, num_workers):
    """
    Calculates the local color entropy of each vertex in the mesh in parallel.

    This function builds a k-d tree from the mesh points, which allows for efficient
    spatial searches in the 3D mesh. It then calculates the local color entropy for
    each vertex in parallel using a Pool of worker processes. The function returns a list
    of local entropies, one for each vertex in the mesh.

    Args:
        vertex_colors_hsv (list): A list of vertex colors in HSV format.
        points (list): A list of 3D points in the mesh.
        kd_tree (scipy.spatial.cKDTree): A k-d tree built from the mesh's points.
        radius (float, optional): The radius to search for neighboring vertices. Defaults to 0.1.
        num_workers (int, optional): The number of worker processes to use for parallel processing.
                                      If None, the number of workers will be set to the number
                                      of available CPU cores. Defaults to None.

    Returns:
        list: A list of local color entropies for each vertex in the mesh.
    """
    local_entropies = [None] * len(points)

    # Split the list of indices into batches for each worker
    indices = list(range(len(points)))
    batch_size = len(points) // num_workers
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    batch_args = [(vertex_colors_hsv, points, kd_tree, radius, batch) for batch in batches]
    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(process_batch, batch_args), 1):
            results.append(res)
            elapsed_time = time.time() - start_time
            print(f'\rProcessed {i}/{num_workers} batches | Elapsed time: {elapsed_time:.2f} seconds', end='')

    print()

    # # Create a Pool of worker processes for parallel processing
    # with Pool(processes=num_workers) as pool:
    #     # Submit tasks to the pool, one for each batch of vertices
    #     results = pool.map(process_batch, batch_args)

    # Flatten the results list and set the local entropy value in the correct order
    for batch_result in results:
        for i, local_entropy in batch_result:
            local_entropies[i] = local_entropy

    return local_entropies
