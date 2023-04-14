"""
Main.py calculates the average local entropy of a mesh using its texture.

The script first loads the mesh and texture files, extracts the vertex colors from the mesh
and texture, and converts the vertex colors from RGB to HSV. It then builds a k-d tree from
the mesh points, which enables efficient spatial searches in the 3D mesh. The script calculates
the local entropies for each vertex using a parallel implementation, and calculates the average
local entropy. Finally, the script prints the average local entropy and the execution time.
"""

#Standard library imports
import argparse
import colorsys
import concurrent.futures
import os
import pickle
import random
import sys
import time
import traceback
import warnings

#Third-party library imports
import numpy as np
from PIL import Image
import vedo
from scipy.spatial import cKDTree
import plotly.subplots as sp
import plotly.graph_objects as go

#Local imports
import calculate_local_entropy as entropy
import spatial_autocorrelation as autocorr
import utilities as util

def main(mesh_file, texture_file, output_directory, num_proccesses , entropy_search_radius, force_entropy, spatial_subset_size, spatial_bootstrap):
    """
    Calculates the average local entropy for a 3D mesh and performs a spatial autocorrelation analysis of the local entropy values.
    
    Args:
    - mesh_file (str): the path to the .obj file containing the mesh.
    - texture_file (str): the path to the texture file to be applied to the mesh.
    - output_directory (str): the path to the directory where output files will be saved.
    - num_proccesses (int): the number of processes to use for parallel computations.
    - entropy_search_radius (float): the radius for the spatial search used to calculate local entropies.
    - force_entropy (bool): whether or not to force the recalculation of vertex entropies.
    - spatial_subset_size (int): the number of points to use for the spatial subset analysis.
    - spatial_bootstrap (bool): whether or not to use bootstrapping to estimate the uncertainty of the semivariance.
    
    Returns:
    - avg_local_entropy (float): the average local entropy value for the mesh.
    - variogram (plotly.graph_objs.Figure): the plotly figure object containing the variogram plot.
    """    

    warnings.filterwarnings("default", category=RuntimeWarning)

    try:
        variogram=None
        avg_local_entropy=None
        start_time = time.time()

        # Load the mesh and texture files
        print(f"Loading mesh file: {mesh_file}")
        mesh = vedo.load(mesh_file)  # Load the mesh from the .obj file
        #mesh_file_prefix = os.path.splitext(mesh_file)[0]
        mesh_file_prefix = os.path.splitext(os.path.basename(mesh_file))[0]
        print(f"Applying texture file {texture_file}")
        mesh = mesh.texture(texture_file)  # Apply the texture to the mesh
        mesh = mesh.triangulate().clean()

        texture_img = Image.open(texture_file)  # Open the texture image using the PIL library

        # Extract the vertex colors from the mesh and texture image
        print(f"Extracting vertex colors")
        vertex_colors = util.get_vertex_colors(mesh, texture_img, mesh_file_prefix)

        # Convert the vertex colors from RGB to HSV
        print(f"Converting from RGB to HSV")
        vertex_colors_hsv = [colorsys.rgb_to_hsv(c[0] / 255, c[1] / 255, c[2] / 255) for c in vertex_colors]

        # Build a k-d tree from mesh points, which enables efficient spatial searches in the 3D mesh
        points=mesh.points()
        faces=mesh.faces()
        kd_tree = cKDTree(points)

        # Calculate the local entropies for each vertex using the parallel implementation
        print(f"Initiating local entropy module...")
        local_entropies = get_local_entropies(force_entropy,  mesh_file_prefix, vertex_colors_hsv, points, kd_tree, entropy_search_radius, num_proccesses)

        # Calculate the average local entropy
        avg_local_entropy = sum(local_entropies) / len(local_entropies)

        # Print the average local entropy
        print(f"Average local entropy: {avg_local_entropy:.4f}")

        # autocorrelation
        if not args.no_autocorr:
            try:
                print("Running spatial analysis of local entropy values...")
                variogram=run_spatial_analysis(output_directory, mesh_file_prefix, points, faces, local_entropies, entropy_search_radius, kd_tree, num_proccesses, spatial_subset_size, spatial_bootstrap)
            except Exception as e:
                traceback.print_exc()
                print(f"Error in spatial autocorrelation module: {e}")
                sys.exit(1)
            end_time = time.time()
            print(f"Execution Time: {end_time - start_time:.2f} seconds")
        return avg_local_entropy, variogram if variogram is not None else go.Figure()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up and exiting...")
        # Perform any necessary cleanup here before exiting the program
        exit()

def get_local_entropies(force_entropy,mesh_file_prefix,vertex_colors_hsv, points, kd_tree, radius, num_proccesses):
    """
    This function is responsible for calculating the local entropy value for each vertex in a mesh. 
    It takes as input the prefix of the mesh file name, a list of HSV tuples representing the color 
    of each vertex in the mesh, the coordinates of each vertex in the mesh, a cKDTree built from the 
    vertex coordinates for efficient spatial searching, the radius to use for calculating the local 
    entropy, and the number of processes to use when calculating the local entropies in parallel.

    The function first checks if a pre-existing file of local entropies exists, and if so, loads the 
    entropies from the file. If the file does not exist or the force_entropy flag is set to True, it 
    calculates the local entropies and saves them to a file
    
    Args:
        force_entropy (bool): If True, forces the calculation of local entropies and overwrites any pre-existing
            local_entropies.pkl file. If False, checks if the file exists and loads the entropies from the file if it
            exists. If the file does not exist, calculates the local entropies and saves them to the file.
        mesh_file_prefix (str): Prefix of the mesh file name.
        vertex_colors_hsv (List[Tuple[float]]): A list of HSV tuples representing the color of each vertex in the mesh.
        points (np.ndarray): A numpy array of shape (n, 3) containing the (x,y,z) coordinates of the n vertices in 
            the mesh.
        kd_tree (scipy.spatial.ckdtree.cKDTree): A cKDTree built from the points array for efficient spatial 
            searching.
        radius (float): The radius to use for calculating local entropy.
        num_proccesses (int): The number of processes to use when calculating local entropies in parallel.
    
    Returns:
        List[float]: A list of length n containing the local entropies for each vertex in the mesh.
    """

    # Check force_entropy is false and if the local_entropies.pkl file exists
    local_entropies_file = f"{mesh_file_prefix}_{str(radius)}_local_entropies.pkl"
    if not force_entropy and os.path.exists(local_entropies_file):
        # If the file exists, load the local entropies from the file
        print(f"Loading local entropies from {local_entropies_file}")
        try:
            with open(local_entropies_file, "rb") as f:
                local_entropies = pickle.load(f)
        except Exception as e:
            print(f"Error loading local entropies from file {local_entropies_file}: {e}")
            sys.exit(1)
        
        # Check that the number of items in local entropies matches the number of points
        if len(local_entropies) != len(points):
            print(f"Error: the number of saved entropies in {local_entropies_file} does not match the number of vertices in the mesh file, {mesh_file}")
            print(f"{local_entropies_file} may be corrupt or does not correspond to the mesh file you are loading")
            print(f"Use the flag --force-entropy to regenerate the entropy file")
            sys.exit(1)
            # Otherwise, recalculate local entropies and save them to a file so they can be reused
    else:
        # If the file does not exist, calculate the local entropies and save them to a file
        local_entropies = entropy.calculate_local_entropies_parallel(vertex_colors_hsv, points, kd_tree, radius, num_proccesses)
        print("Saving local entropies to file...")
        with open(local_entropies_file, "wb") as f:
            pickle.dump(local_entropies, f)
    return local_entropies  

def run_spatial_analysis(output_directory,mesh_file_prefix,  points, faces, local_entropies, radius, kd_tree, num_proccesses, spatial_subset_size, spatial_bootstrap):
    """
    This function calculates the spatial autocorrelation of a given mesh using a bootstrap method. The spatial 
    autocorrelation describes the degree of similarity between the values of a given variable (in this case, the local 
    entropy values of the mesh vertices) at different spatial locations. The spatial autocorrelation is commonly 
    measured using the semivariogram, which is a plot of the semivariance (a measure of the degree of dissimilarity 
    between two points at a given distance) against lag distance (the distance between the two points).

    The function takes as input the directory to save output files, the prefix of the mesh file name, a numpy array 
    of the (x,y,z) coordinates of the mesh vertices, a numpy array of the vertex indices that form each triangular 
    face of the mesh, a list of the local entropy value for each vertex in the mesh, the radius to use for calculating 
    the local entropy, a cKDTree built from the vertex coordinates for efficient spatial searching, the number of 
    processes to use when calculating local entropies in parallel, the size of the random subset of points used for 
    each iteration in the bootstrap method, and the number of bootstrap iterations to run.

    The function runs the bootstrap iterations in parallel and calculates the semivariance of the local entropy 
    values over a range of lags. The maximum bin edge is found across all iterations, and new bins are calculated 
    based on this value. The semivariance values are assigned to the best new bin based on the largest overlap, and 
    the semivariances for each bin across all iterations are stored in a dictionary. The average semivariance for each 
    non-empty bin and the standard error of the mean for each non-empty bin are then calculated. The variogram is 
    plotted using the average semivariances and non-empty bin edges, and the function returns a Plotly figure object of 
    the variogram.

    Args:
        output_directory (str): Directory to save output files.
        mesh_file_prefix (str): Prefix of the mesh file name.
        points (np.ndarray): A numpy array of shape (n, 3) containing the (x,y,z) coordinates of the n vertices in 
            the mesh.
        faces (np.ndarray): A numpy array of shape (m, 3) containing the vertex indices that form each triangular face 
            of the mesh.
        local_entropies (List[float]): A list of length n containing the local entropy value for each vertex in the mesh.
        radius (float): The radius to use for calculating the local entropy.
        kd_tree (scipy.spatial.ckdtree.cKDTree): A cKDTree built from the points array for efficient spatial searching.
        num_proccesses (int): The number of processes to use when calculating local entropies in parallel.
        spatial_subset_size (int): The size of the random subset of points used for each iteration in the bootstrap method.
        spatial_bootstrap (int): The number of bootstrap iterations to run.

    Returns:
        A Plotly figure object

    Raises:
        None
    """
    num_workers = num_proccesses  # Number of parallel workers
    N = spatial_bootstrap  # Number of bootstrap iterations
    subset_size=spatial_subset_size
    #subset_size = 20 # Size of the random subset of points
    # Set the seed value for each process
    def init_random(*seeds):
        random.seed(sum(seeds))
        
    # Run the bootstrap iterations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_random, initargs=[random.randint(0, 100000) for i in range(num_workers)]) as executor:
        # returns a numpy array containing the semi-variances, and the geodesic distances
        results = list(executor.map(autocorr.single_bootstrap_iteration, [(i,  points, faces, local_entropies, subset_size,random.randint(0, 100000)) for i in range(N)]))
    # Initialize a variable to store the maximum bin edge seen so far
    max_bin_edge = 0

    # Loop over the results from each iteration
    for result in results:
        semivariance_dict = result
        # Find the maximum bin edge for this iteration
        max_bin_edge_iter = max(semivariance_dict.keys())
        # Update the overall maximum bin edge if necessary
        if max_bin_edge_iter > max_bin_edge:
            max_bin_edge = max_bin_edge_iter

    # Once the overall maximum bin edge has been found, calculate new bins based on this value
    num_bins = 10
    bin_edges = np.linspace(0, max_bin_edge, num_bins+1)

    # Initialize a dictionary to store the semivariances for each bin across all iterations
    semivariances_by_bin = {i: [] for i in range(num_bins)}

    # Helper function to calculate overlap between old and new bins
    def calculate_overlap(old_bin, new_bin):
        overlap_start = max(old_bin[0], new_bin[0])
        overlap_end = min(old_bin[1], new_bin[1])
        overlap = max(0, overlap_end - overlap_start + 1)
        return overlap    
    
    # Main loop to assign semivariance values based on the largest overlap
    for result in results:
        semivariance_dict = result
        for bin_edge, semivariance in semivariance_dict.items():
            lower_bound = bin_edge - (bin_edge % (bin_edges[1] - bin_edges[0]))

            # Create a tuple representing the old bin (lower_bound, bin_edge)
            old_bin = (lower_bound, bin_edge)

            # Find the new bin with the largest overlap
            largest_overlap = 0
            best_new_bin_idx = 0
            for new_bin_idx in range(len(bin_edges) - 1):  # Iterate over bin_edges indices
                new_bin = (bin_edges[new_bin_idx], bin_edges[new_bin_idx + 1])
                overlap = calculate_overlap(old_bin, new_bin)
                if overlap > largest_overlap:
                    largest_overlap = overlap
                    best_new_bin_idx = new_bin_idx    
                elif overlap == largest_overlap and new_bin[0] < bin_edges[best_new_bin_idx]:
                    best_new_bin_idx = new_bin_idx
            # Assign the entire semivariance value to the best new bin
            semivariances_by_bin[best_new_bin_idx].append(semivariance)

    # Filter out bins with no values
    non_empty_bins = {i: v for i, v in semivariances_by_bin.items() if len(v) > 0}

    # Calculate the average semivariance for each non-empty bin
    avg_semivariances = [np.mean(non_empty_bins[i]) for i in non_empty_bins.keys()]

    # Calculate the standard error of the mean for each non-empty bin
    sem = [np.std(non_empty_bins[i]) / np.sqrt(len(non_empty_bins[i])) for i in non_empty_bins.keys()]

    # Create a list of non-empty bin edges
    non_empty_bin_edges = [bin_edges[i] for i in non_empty_bins.keys()]

    # Plot the variogram using the average semivariances and non-empty bin edges
    variogram=autocorr.plot_variogram(output_directory, mesh_file_prefix, non_empty_bin_edges, avg_semivariances, max_range=None, sem=sem)

    return variogram

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate the average local entropy of a mesh using its texture")
    parser.add_argument("mesh_file", nargs="?", help="Path to the input mesh file (.obj format)")
    parser.add_argument("texture_file", nargs="?", help="Path to the input texture file (.jpg format)")
    parser.add_argument("-o","--output-directory", default=os.getcwd(), help="Output directory for the results (default: current working directory)")
    parser.add_argument("-n","--num-processes", type=int, default=os.cpu_count(), help="Number of workers to use for parallel processing (default: number of CPU cores)")
    parser.add_argument("-r","--entropy-search-radius", type=float, default=0.05, help="Radius (in mesh units) used for finding neighboring vertices when calculating local entropies (default: 0.1)")
    parser.add_argument("--spatial_subset_size", type=int, default=30, help="Number of vertices to sample in the spatial analysis module. Should be at least 30 to produce reliable variograms (default=30)")
    parser.add_argument("--spatial_bootstrap", type=int, default=3, help="Number of bootstrap iterations in the spatial analysis module when estimating the variogram (default=3)")   
    parser.add_argument('--force-entropy', action='store_true', help="Re-calculates local entropy even when a previous session exists")
    parser.add_argument('--no-autocorr', action='store_true', help="Suppress autocorrelation module")

    args = parser.parse_args()
    num_proccesses = args.num_processes
    output_directory = args.output_directory
    spatial_subset_size = args.spatial_subset_size
    spatial_bootstrap = args.spatial_bootstrap
    output_directory = args.output_directory
    spatial_subset_size=100
    spatial_bootstrap=100
    fig_width = 800
    species_list = [
        ("fletcheri_JJ89-meshlab.obj", "fletcheri_JJ89-meshlab.jpg"),
        ("prezewalskii_JJ14-meshlab.obj", "prezewalskii_JJ14-meshlab.jpg")
        # Add more species here
    ]

    species_results = {}
    for i, (mesh_file, texture_file) in enumerate(species_list):
        species_name = os.path.splitext(os.path.basename(mesh_file))[0]
        avg_local_entropy, variogram = main(mesh_file, texture_file, output_directory, num_proccesses, args.entropy_search_radius, args.force_entropy, spatial_subset_size, spatial_bootstrap)
        species_results[species_name] = {"avg_local_entropy": avg_local_entropy, "variogram": variogram, "index": i}

    # create a new figure with the appropriate number of subplots
    num_species = len(species_list)
    num_rows = num_species
    fig = sp.make_subplots(rows=num_rows, cols=1, vertical_spacing=0.1, column_widths=[0.8])
    fig.update_layout(showlegend=False)

    max_x_value = max([np.array(trace.x).max() for species_name, species_data in species_results.items() for trace in species_data["variogram"].data])
    max_y_value = max([np.array(trace.y).max() for species_name, species_data in species_results.items() for trace in species_data["variogram"].data])

    # add the variograms for each species to the appropriate subplot
    print(species_results.keys())
    for i, species_name in enumerate(species_results.keys()):
        species_data = species_results[species_name]
        avg_local_entropy = species_data["avg_local_entropy"]
        variogram = species_data["variogram"]

        # add the variogram trace to the subplot and annotate with species name and avg_local_entropy
        subplot_idx = i + 1
        for trace in variogram.data:  # Loop through all the traces in the individual variogram
            fig.add_trace(trace, row=subplot_idx, col=1)
 
        fig.update_yaxes(title_text="Semivariance", row=subplot_idx, col=1, range=[0,max_y_value])
        fig.update_xaxes(title_text="Distance", row=subplot_idx, col=1, range=[0,max_x_value])

        # calculate the y-coordinate of the annotation based on the height of the subplot
        subplot_height = 1.0 / num_rows
        #annotation_y = (i + 0.5) * subplot_height
        annotation_y = (num_species - i - 0.5) * subplot_height
        #print(f"{species_name} i:{i} subplot_height:{subplot_height} annotation:{annotation_y} ")
        # add the annotation to the subplot 
        fig.add_annotation(
            x=1.05, y=annotation_y, xref="paper", yref="paper",
            text=f"3D model: <b>{species_name}</b><br>Average local entropy: <b>{avg_local_entropy:.3f}</b>",
            showarrow=False, font=dict(size=14), align="left", xanchor='left', yanchor='middle'
        )

    # update the layout and display the figure
    plot_title=f"Variograms for all species <br>Model parameters: entropy search radius: {args.entropy_search_radius}, sampled vertices:{spatial_subset_size}, variogram iterations: {spatial_bootstrap}"
    fig.update_layout(height=800, width=1200, title_text=plot_title, margin=dict(r=400))
    fig.show()





    mesh_file = args.mesh_file or "prezewalskii_JJ14-meshlab.obj"
    texture_file = args.texture_file or "prezewalskii_JJ14-meshlab.jpg"    

    #num_proccesses = args.num_processes
    #output_directory = args.output_directory
    #main(mesh_file, texture_file, output_directory, num_proccesses, args.entropy_search_radius, args.force_entropy)        