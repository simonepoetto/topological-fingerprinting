\
import numpy as np
from scipy.spatial.distance import squareform
from itertools import combinations
import random

def get_links_weights(links, weights_matrix):
    """
    Extracts weights for a given list of links from a weight matrix.

    Args:
        links (list): A list of links, where each link is a tuple or list of two node indices (1-based).
        weights_matrix (np.ndarray): The NxN matrix containing weights.

    Returns:
        list: A list of weights corresponding to the input links. Returns empty list if input links is empty.
              Returns NaN for links where indices might be out of bounds.
    """
    links_weights = []
    n_nodes = weights_matrix.shape[0]
    for link in links:
        # Adjust for 0-based indexing and ensure indices are within bounds
        node1, node2 = link[0] - 1, link[1] - 1
        if 0 <= node1 < n_nodes and 0 <= node2 < n_nodes:
            # Ensure order for upper/lower triangle consistency if matrix is symmetric
            # If not symmetric, the order matters as provided. Assuming symmetric for now.
            # Use max/min to handle potential order differences in link representation
            # links_weights.append(weights_matrix[min(node1, node2), max(node1, node2)]) # Assumes upper triangle if symmetric
            # Let's assume the matrix access handles the order correctly or the matrix is dense
            links_weights.append(weights_matrix[node1, node2])
        else:
            # Handle out-of-bounds indices, e.g., append NaN or raise error
            links_weights.append(np.nan)
            print(f"Warning: Link {link} indices are out of bounds for matrix shape {weights_matrix.shape}.")

    return links_weights

def get_cycle_nodes(cycle):
    """
    Finds all unique nodes involved in a cycle.

    Args:
        cycle (list): A list of links representing the cycle.

    Returns:
        list: Sorted list of unique node indices (1-based). Returns empty list if cycle is empty.
    """
    if not cycle:
        return []
    nodes = set()
    for link in cycle:
        if len(link) == 2:
            nodes.update(link)
        else:
            print(f"Warning: Invalid link format found in cycle: {link}. Skipping.")
            # Decide how to handle invalid links, e.g., skip or raise error
    return sorted(list(nodes))


def get_cycle_missing_links(cycle):
    """
    Finds the links that are missing to make the cycle's nodes form a clique.

    Args:
        cycle (list): A list of links representing the cycle.

    Returns:
        list: A list of missing links (each as a list of two 1-based node indices).
              Returns empty list if cycle has < 3 nodes or is already a clique.
    """
    nodes = get_cycle_nodes(cycle)
    if len(nodes) < 3:
        return [] # Cannot have missing links in a cycle with less than 3 nodes

    # Represent cycle links as sets for easier comparison
    cycle_links_set = {frozenset(link) for link in cycle if len(link) == 2}

    # Generate all possible links between the nodes (potential clique links)
    all_possible_links_set = {frozenset(link) for link in combinations(nodes, 2)}

    # Find links present in all_possible_links but not in cycle_links
    missing_links_set = all_possible_links_set - cycle_links_set

    # Convert back to list of lists format
    missing_links = [list(link) for link in missing_links_set]

    return missing_links

def calculate_mean_border_synergy(cycles, synergy_matrix):
    """
    Calculates the mean synergy of links forming the border of the cycles for a subject.

    Args:
        cycles (list): List of representative cycles for the subject. Each cycle is a list of links.
        synergy_matrix (np.ndarray): The NxN synergy matrix for the subject.

    Returns:
        float: The mean synergy value of all border links across all cycles. Returns NaN if no valid links.
    """
    all_border_links = [link for cycle in cycles for link in cycle]
    if not all_border_links:
        return np.nan
    border_synergies = get_links_weights(all_border_links, synergy_matrix)
    return np.nanmean(border_synergies) # Use nanmean to ignore potential NaNs from get_links_weights

def calculate_mean_internal_synergy(cycles, synergy_matrix):
    """
    Calculates the mean synergy of the internal (missing) links within the cycles for a subject.

    Args:
        cycles (list): List of representative cycles for the subject.
        synergy_matrix (np.ndarray): The NxN synergy matrix for the subject.

    Returns:
        float: The mean synergy value of all internal links across all cycles. Returns NaN if no internal links.
    """
    all_internal_links = [link for cycle in cycles for link in get_cycle_missing_links(cycle)]
    if not all_internal_links:
        return np.nan
    internal_synergies = get_links_weights(all_internal_links, synergy_matrix)
    return np.nanmean(internal_synergies)

def calculate_mean_random_synergy(num_links_to_sample, synergy_matrix):
    """
    Calculates the mean synergy of a specified number of randomly selected links.

    Args:
        num_links_to_sample (int): The number of random links to sample.
        synergy_matrix (np.ndarray): The NxN synergy matrix.

    Returns:
        float: The mean synergy of the randomly selected links. Returns NaN if sampling is not possible.
    """
    n_nodes = synergy_matrix.shape[0]
    if n_nodes < 2 or num_links_to_sample <= 0:
        return np.nan

    # Flatten the upper triangle (excluding diagonal) to get unique link synergies
    # Ensure the matrix is symmetric or adjust accordingly if not
    # Assuming symmetric for now
    if not np.allclose(synergy_matrix, synergy_matrix.T, equal_nan=True):
         print("Warning: Synergy matrix may not be symmetric. Flattening upper triangle.")
    synergy_flat = squareform(synergy_matrix, checks=False) # Disable checks for speed if matrix is known good

    if len(synergy_flat) < num_links_to_sample:
        print(f"Warning: Requested {num_links_to_sample} samples, but only {len(synergy_flat)} unique links exist. Sampling all.")
        num_links_to_sample = len(synergy_flat)

    if num_links_to_sample == 0:
        return np.nan

    random_synergies = np.random.choice(synergy_flat, size=num_links_to_sample, replace=False)
    return np.nanmean(random_synergies)


def calculate_mean_random_synergy_fc_matched(cycles, synergy_matrix, fc_matrix, tol=0.02):
    """
    Calculates the mean synergy of random links matched one-to-one to the FC of the internal links.

    For each internal link, one link is drawn at random among all links whose FC lies within
    +/- tol of that internal link's FC, so the null has the same FC distribution as the internal links.

    Args:
        cycles (list): List of representative cycles for the subject.
        synergy_matrix (np.ndarray): The NxN synergy matrix.
        fc_matrix (np.ndarray): The NxN functional connectivity matrix.
        tol (float): FC matching tolerance.

    Returns:
        float: Mean synergy of FC-matched random links. Returns NaN if not possible.
    """
    all_internal_links = [link for cycle in cycles for link in get_cycle_missing_links(cycle)]
    if synergy_matrix.shape[0] < 2 or not all_internal_links:
        return np.nan

    internal_fc_values = np.array(get_links_weights(all_internal_links, fc_matrix), dtype=float)
    internal_fc_values = internal_fc_values[~np.isnan(internal_fc_values)]
    if internal_fc_values.size == 0:
        return np.nan

    # Upper triangles, sorted by FC, so the candidates for each internal link are a contiguous range
    fc_flat = squareform(fc_matrix, checks=False)
    synergy_flat = squareform(synergy_matrix, checks=False)
    valid = ~np.isnan(fc_flat) & ~np.isnan(synergy_flat)
    order = np.argsort(fc_flat[valid])
    fc_sorted = fc_flat[valid][order]
    synergy_sorted = synergy_flat[valid][order]

    lo = np.searchsorted(fc_sorted, internal_fc_values - tol, side='left')
    hi = np.searchsorted(fc_sorted, internal_fc_values + tol, side='right')
    ok = hi > lo
    if not ok.any():
        return np.nan
    picks = lo[ok] + (np.random.random(ok.sum()) * (hi[ok] - lo[ok])).astype(int)
    return np.nanmean(synergy_sorted[picks])


def calculate_mean_random_cycle_internal_synergy(cycles, synergy_matrix):
    """
    Generates random cycles matching the number and lengths of the original cycles,
    then calculates the mean synergy of the internal links of these random cycles.

    Args:
        cycles (list): List of representative cycles for the subject.
        synergy_matrix (np.ndarray): The NxN synergy matrix.

    Returns:
        float: Mean internal synergy across all generated random cycles. Returns NaN if not possible.
    """
    n_nodes = synergy_matrix.shape[0]
    if n_nodes < 3 or not cycles:
        return np.nan

    all_random_internal_synergies = []

    for original_cycle in cycles:
        original_nodes = get_cycle_nodes(original_cycle)
        k = len(original_nodes) # Length (number of nodes) of the original cycle

        if k < 3 or k > n_nodes:
            # Skip if cycle length is invalid
            continue

        # Generate a random cycle of the same length k
        try:
            random_nodes = random.sample(range(1, n_nodes + 1), k)
        except ValueError:
            # Should not happen if k <= n_nodes, but as a safeguard
            continue

        # Construct the links for the random cycle border: [(n1, n2), (n2, n3), ..., (nk, n1)]
        random_cycle_links = []
        for i in range(k):
            node1 = random_nodes[i]
            node2 = random_nodes[(i + 1) % k] # Wrap around
            # Store links consistently, e.g., smaller index first (doesn't strictly matter for get_cycle_missing_links)
            link = tuple(sorted((node1, node2)))
            random_cycle_links.append(list(link)) # Store as list of lists

        # Get missing links for this random cycle structure
        missing_links_random = get_cycle_missing_links(random_cycle_links)

        if missing_links_random:
            # Calculate synergy values for these missing links using the *actual* synergy matrix
            missing_synergies = get_links_weights(missing_links_random, synergy_matrix)
            all_random_internal_synergies.extend(missing_synergies)

    if not all_random_internal_synergies:
        return np.nan # No internal links found in any random cycle

    return np.nanmean(all_random_internal_synergies)


def get_subject_synergy_metrics(cycles, synergy_matrix, fc_matrix):
    """
    Calculates the five requested synergy metrics for a single subject.

    Args:
        cycles (list): List of representative cycles for the subject.
        synergy_matrix (np.ndarray): The NxN synergy matrix for the subject.
        fc_matrix (np.ndarray): The NxN functional connectivity matrix for the subject.

    Returns:
        dict: A dictionary containing the five mean synergy values:
              'border': Mean synergy of cycle border links.
              'internal': Mean synergy of cycle internal links.
              'random': Mean synergy of random links (matched count to internal).
              'random_fc_matched': Mean synergy of random links matched one-to-one to the internal links' FC.
              'random_cycle_internal': Mean internal synergy from randomly generated cycles.
              Returns NaNs for values that cannot be calculated.
    """
    n_nodes = synergy_matrix.shape[0]
    if n_nodes < 3:
        return {
            'border': np.nan,
            'internal': np.nan,
            'random': np.nan,
            'random_fc_matched': np.nan,
            'random_cycle_internal': np.nan
        }

    # Calculate Border Synergy
    mean_border = calculate_mean_border_synergy(cycles, synergy_matrix)

    # Calculate Internal Synergy
    mean_internal = calculate_mean_internal_synergy(cycles, synergy_matrix)

    # Determine number of internal links for random sampling
    num_internal_links = sum(len(get_cycle_missing_links(cycle)) for cycle in cycles)
    num_border_links = sum(len(cycle) for cycle in cycles)

    # Calculate Mean Random Synergy
    if num_internal_links > 0:
        mean_random = calculate_mean_random_synergy(num_internal_links, synergy_matrix)
    else:
        mean_random = np.nan # Cannot sample 0 links

    # Calculate Mean Random Synergy (FC Matched)
    mean_random_fc = calculate_mean_random_synergy_fc_matched(cycles, synergy_matrix, fc_matrix)

    # Calculate Mean Random Cycle Internal Synergy
    mean_random_cycle = calculate_mean_random_cycle_internal_synergy(cycles, synergy_matrix)

    return {
        'border': mean_border,
        'internal': mean_internal,
        'random': mean_random,
        'random_fc_matched': mean_random_fc,
        'random_cycle_internal': mean_random_cycle
    }

# Example Usage (requires actual data loading)
if __name__ == '__main__':
    # This is placeholder data. Replace with actual loading mechanisms.
    print("Running example usage (with placeholder data)...")
    num_nodes = 30
    # Create dummy symmetric matrices
    dummy_synergy = np.random.rand(num_nodes, num_nodes)
    dummy_synergy = (dummy_synergy + dummy_synergy.T) / 2
    np.fill_diagonal(dummy_synergy, 0)

    dummy_fc = np.random.rand(num_nodes, num_nodes)
    dummy_fc = (dummy_fc + dummy_fc.T) / 2
    np.fill_diagonal(dummy_fc, 0)

    # Create dummy cycles (list of lists of links, 1-based indexing)
    dummy_cycles = [
        [[1, 2], [2, 3], [3, 4], [4, 1]], # 4-cycle
        [[5, 6], [6, 7], [7, 5]],         # 3-cycle (triangle)
        [[8, 9], [9, 10], [10, 11], [11, 12], [12, 8]] # 5-cycle
    ]

    # --- Test individual functions ---
    print("\nTesting individual functions:")
    cycle_4 = dummy_cycles[0]
    nodes_4 = get_cycle_nodes(cycle_4)
    missing_4 = get_cycle_missing_links(cycle_4)
    print(f"Cycle 4: {cycle_4}")
    print(f"Nodes: {nodes_4}")
    print(f"Missing Links: {missing_4}")
    if missing_4:
        print(f"Internal Synergy (Cycle 4): {np.nanmean(get_links_weights(missing_4, dummy_synergy))}")

    cycle_3 = dummy_cycles[1]
    nodes_3 = get_cycle_nodes(cycle_3)
    missing_3 = get_cycle_missing_links(cycle_3)
    print(f"\nCycle 3: {cycle_3}")
    print(f"Nodes: {nodes_3}")
    print(f"Missing Links: {missing_3}") # Should be empty for a triangle
    if missing_3:
         print(f"Internal Synergy (Cycle 3): {np.nanmean(get_links_weights(missing_3, dummy_synergy))}")
    else:
         print("Internal Synergy (Cycle 3): N/A (no missing links)")


    # --- Test main function ---
    print("\nTesting main get_subject_synergy_metrics function:")
    metrics = get_subject_synergy_metrics(dummy_cycles, dummy_synergy, dummy_fc)

    print("\nCalculated Synergy Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}" if not np.isnan(value) else f"- {key}: NaN")

    # Example: Test random cycle generation part
    print("\nTesting random cycle internal synergy calculation:")
    rand_cycle_internal_syn = calculate_mean_random_cycle_internal_synergy(dummy_cycles, dummy_synergy)
    print(f"Mean internal synergy for random cycles matching originals: {rand_cycle_internal_syn:.4f}" if not np.isnan(rand_cycle_internal_syn) else "NaN")

    # Example: Test FC matching part
    print("\nTesting FC-matched random synergy calculation:")
    rand_fc_match_syn = calculate_mean_random_synergy_fc_matched(dummy_cycles, dummy_synergy, dummy_fc)
    print(f"Mean synergy for random links matched by FC: {rand_fc_match_syn:.4f}" if not np.isnan(rand_fc_match_syn) else "NaN")

    # Example: Test simple random synergy
    print("\nTesting simple random synergy calculation:")
    num_internal = sum(len(get_cycle_missing_links(c)) for c in dummy_cycles)
    print(f"Total internal links across cycles: {num_internal}")
    if num_internal > 0:
        rand_syn = calculate_mean_random_synergy(num_internal, dummy_synergy)
        print(f"Mean synergy for {num_internal} random links: {rand_syn:.4f}" if not np.isnan(rand_syn) else "NaN")
    else:
        print("Mean synergy for random links: N/A (no internal links to match count)")

