import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.interpolate import CubicSpline, interp1d

import os
import glob
from pathlib import Path

import ast
import matplotlib.ticker as ticker

from .helperFunc import *
from scipy import stats


"""
Cluster-based permutation testing for time series data.
"""
def cluster_permutation_test(
    data_df,
    condition_column,
    condition_values,
    n_permutations=5000,
    data_columns=None,
    p_threshold=0.05,
    stat_function='ttest',
    tail='two-sided',
    cluster_stat='sum',
    seed=None,
    verbose=True
):
    """
    Perform cluster-based permutation test on time series data.
    
    Shuffles condition labels and computes null distribution of cluster statistics
    to test whether observed clusters are significant.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Data with time series in columns and a condition label column
    condition_column : str
        Name of column containing condition labels
    condition_values : tuple or list of length 2
        Two condition values to compare (e.g., ('day', 'night') or ('high', 'low'))
    n_permutations : int, default=5000
        Number of permutations for null distribution
    data_columns : list or slice, optional
        Columns to use for analysis. If None, uses all numeric columns.
        Can be list of column names or slice object (e.g., slice(0, 3500))
    p_threshold : float, default=0.05
        P-value threshold for forming clusters
    stat_function : str or callable, default='ttest'
        Statistical test to use. Options: 'ttest', 'mannwhitneyu', or custom function
    tail : str, default='two-sided'
        Test type: 'two-sided', 'greater', or 'less'
    cluster_stat : str, default='sum'
        How to aggregate statistics within clusters: 'sum', 'mean', 'max', 'mass'
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'null_distribution': array of null cluster statistics
        - 'observed_clusters': list of observed cluster indices
        - 'observed_stats': observed test statistics at each timepoint
        - 'observed_pvals': observed p-values at each timepoint
        - 'largest_cluster_stat': statistic of largest observed cluster
        - 'p_value': p-value of largest observed cluster
    
    Examples
    --------
    >>> result = cluster_permutation_test(
    ...     data_df=pupil_data,
    ...     condition_column='cuedItemBri',
    ...     condition_values=('day', 'night'),
    ...     n_permutations=5000,
    ...     data_columns=slice(0, 3500)
    ... )
    >>> print(f"Cluster p-value: {result['p_value']:.4f}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Extract condition data
    cond1_name, cond2_name = condition_values
    cond1_mask = data_df[condition_column] == cond1_name
    cond2_mask = data_df[condition_column] == cond2_name
    
    # Select data columns
    if data_columns is None:
        # Use all numeric columns except condition column
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns
        data_columns = [col for col in numeric_cols if col != condition_column]
    elif isinstance(data_columns, slice):
        # Convert slice to column list
        all_cols = data_df.columns.tolist()
        data_columns = all_cols[data_columns]
    
    n_timepoints = len(data_columns)
    
    if verbose:
        print(f"Running cluster permutation test:")
        print(f"  Conditions: {cond1_name} vs {cond2_name}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Permutations: {n_permutations}")
        print(f"  P-threshold: {p_threshold}")
    
    # Choose statistical test
    if stat_function == 'ttest':
        def compute_stats(group1, group2):
            return stats.ttest_ind(group1, group2, nan_policy='omit', 
                                   alternative=tail)
    elif stat_function == 'mannwhitneyu':
        def compute_stats(group1, group2):
            alternative = 'two-sided' if tail == 'two-sided' else tail
            return stats.mannwhitneyu(group1, group2, nan_policy='omit',
                                     alternative=alternative)
    elif callable(stat_function):
        compute_stats = stat_function
    else:
        raise ValueError(f"Unknown stat_function: {stat_function}")
    
    # Compute observed statistics
    if verbose:
        print("\nComputing observed statistics...")
    
    cond1_data = data_df[cond1_mask][data_columns]
    cond2_data = data_df[cond2_mask][data_columns]
    
    observed_results = np.array([
        compute_stats(cond1_data.iloc[:, i], cond2_data.iloc[:, i])[:2]
        for i in range(n_timepoints)
    ])
    
    observed_stats = observed_results[:, 0]
    observed_pvals = observed_results[:, 1]
    
    # Find observed clusters
    significant_mask = observed_pvals < p_threshold
    observed_clusters = find_consecutive_groups(
        np.where(significant_mask)[0],
        stepsize=1,
        find_same=False
    )

    # Filter out empty clusters
    observed_clusters = [c for c in observed_clusters if len(c) > 0]

    # Compute observed cluster statistic
    if len(observed_clusters) > 0:
        largest_obs_cluster = _get_largest_cluster_stat(
            observed_clusters, observed_stats, cluster_stat
        )
        
        if verbose:
            largest_idx = np.argmax([_compute_cluster_stat(observed_stats[c], cluster_stat) 
                                    for c in observed_clusters])
            largest_cluster = observed_clusters[largest_idx]
            print(f"  Largest observed cluster: {len(largest_cluster)} timepoints "
                f"(indices {largest_cluster[0]}-{largest_cluster[-1]}), "
                f"statistic={largest_obs_cluster:.3f}")
            
    else: # If no cluster found, terminate the process
        largest_obs_cluster = 0
        if verbose:
            print("  No significant clusters found in observed data")
            return None
    
    if verbose and len(observed_clusters) > 0:
        largest_idx = np.argmax([_compute_cluster_stat(observed_stats[c], cluster_stat) 
                                for c in observed_clusters])
        largest_cluster = observed_clusters[largest_idx]
        print(f"  Largest observed cluster: {len(largest_cluster)} timepoints "
              f"(indices {largest_cluster[0]}-{largest_cluster[-1]}), "
              f"statistic={largest_obs_cluster:.3f}")
    
    # Run permutations
    if verbose:
        print(f"\nRunning {n_permutations} permutations...")
    
    null_distribution = np.zeros(n_permutations)
    combined_data = data_df[cond1_mask | cond2_mask].copy()
    
    for perm_idx in range(n_permutations):
        if verbose and (perm_idx + 1) % 1000 == 0:
            print(f"  Progress: {perm_idx + 1}/{n_permutations}")
        
        # Shuffle condition labels
        shuffled_labels = np.random.permutation(
            combined_data[condition_column].values
        )
        combined_data['_shuffled_label'] = shuffled_labels
        
        # Split by shuffled labels
        pseudo_cond1 = combined_data[combined_data['_shuffled_label'] == cond1_name][data_columns]
        pseudo_cond2 = combined_data[combined_data['_shuffled_label'] == cond2_name][data_columns]
        
        # Compute statistics for this permutation
        perm_results = np.array([
            compute_stats(pseudo_cond1.iloc[:, i], pseudo_cond2.iloc[:, i])[:2]
            for i in range(n_timepoints)
        ])
        
        perm_stats = perm_results[:, 0]
        perm_pvals = perm_results[:, 1]
        
        # Find clusters in permuted data
        perm_significant = perm_pvals < p_threshold
        perm_clusters = find_consecutive_groups(
            np.where(perm_significant)[0],
            stepsize=1,
            find_same=False
        )
        
        # Store largest cluster statistic
        if len(perm_clusters) > 0:
            null_distribution[perm_idx] = _get_largest_cluster_stat(
                perm_clusters, perm_stats, cluster_stat
            )
        else:
            null_distribution[perm_idx] = 0
    
    # Calculate p-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(largest_obs_cluster))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Observed largest cluster statistic: {largest_obs_cluster:.3f}")
        print(f"  Permutation p-value: {p_value:.4f}")
        print(f"{'='*60}")
    
    return {
        'null_distribution': null_distribution,
        'observed_clusters': observed_clusters,
        'observed_stats': observed_stats,
        'observed_pvals': observed_pvals,
        'largest_cluster_stat': largest_obs_cluster,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'condition_values': condition_values
    }


def _compute_cluster_stat(cluster_stats, method='sum'):
    """Compute aggregate statistic for a cluster."""
    if method == 'sum':
        return np.sum(cluster_stats)
    elif method == 'mean':
        return np.mean(cluster_stats)
    elif method == 'max':
        return np.max(np.abs(cluster_stats))
    elif method == 'mass':
        # Cluster mass (sum of absolute values)
        return np.sum(np.abs(cluster_stats))
    else:
        raise ValueError(f"Unknown cluster_stat method: {method}")


def _get_largest_cluster_stat(clusters, stats, method='sum'):
    """Find the largest cluster statistic."""
    if len(clusters) == 0:
        return 0
    
    cluster_stats = [_compute_cluster_stat(stats[cluster], method) 
                    for cluster in clusters]
    return cluster_stats[np.argmax(np.abs(cluster_stats))]


def plot_permutation_results(result, times=None, data_df=None, 
                             condition_column=None, data_columns=None):
    """
    Visualize cluster permutation test results.
    
    Parameters
    ----------
    result : dict
        Output from cluster_permutation_test()
    times : array-like, optional
        Time values for x-axis. If None, uses indices.
    data_df : pd.DataFrame, optional
        Original data for plotting condition means
    condition_column : str, optional
        Condition column name (needed if plotting data)
    data_columns : list, optional
        Data columns (needed if plotting data)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    n_timepoints = len(result['observed_stats'])
    if times is None:
        times = np.arange(n_timepoints)
    
    # Plot 1: Null distribution
    ax = axes[0]
    ax.hist(result['null_distribution'], bins=50, alpha=0.7, color='gray',
            edgecolor='black')
    ax.axvline(result['largest_cluster_stat'], color='red', linestyle='--',
               linewidth=2, label=f"Observed (p={result['p_value']:.4f})")
    ax.set_xlabel('Cluster Statistic')
    ax.set_ylabel('Frequency')
    ax.set_title('Null Distribution from Permutation Test')
    ax.legend()
    
    # Plot 2: T-statistics with clusters
    ax = axes[1]
    ax.plot(times, result['observed_stats'], 'k-', linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Highlight significant clusters
    for cluster in result['observed_clusters']:
        if len(cluster) > 0:
            ax.axvspan(times[cluster[0]], times[cluster[-1]], 
                      alpha=0.3, color='red')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('T-statistic')
    ax.set_title('Observed Statistics with Significant Clusters')
    
    # Plot 3: Data means (if provided)
    if data_df is not None and condition_column is not None:
        ax = axes[2]
        
        cond1_name, cond2_name = result['condition_values']
        
        if data_columns is None:
            data_columns = data_df.select_dtypes(include=[np.number]).columns
        
        cond1_data = data_df[data_df[condition_column] == cond1_name][data_columns]
        cond2_data = data_df[data_df[condition_column] == cond2_name][data_columns]
        
        mean1 = cond1_data.mean(axis=0)
        sem1 = cond1_data.sem(axis=0)
        mean2 = cond2_data.mean(axis=0)
        sem2 = cond2_data.sem(axis=0)
        
        ax.plot(times, mean1, label=cond1_name, color='blue')
        ax.fill_between(times, mean1-sem1, mean1+sem1, alpha=0.3, color='blue')
        ax.plot(times, mean2, label=cond2_name, color='orange')
        ax.fill_between(times, mean2-sem2, mean2+sem2, alpha=0.3, color='orange')
        
        # Highlight clusters
        for cluster in result['observed_clusters']:
            if len(cluster) > 0:
                ax.axvspan(times[cluster[0]], times[cluster[-1]], 
                          alpha=0.2, color='red')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Value')
        ax.set_title(f'Condition Means (p={result["p_value"]:.4f})')
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_permutation_results(
    result, 
    times=None, 
    data_df=None, 
    condition_column=None, 
    data_columns=None,
    colors=None,
    labels=None,
    figsize=(20, 15),
    smoothing_window=11,
    linewidth=8,
    vline_positions=None,
    cluster_color='#359937',
    show_null_dist=True,
    show_tstats=True
):
    """
    Visualize cluster permutation test results with custom styling.
    
    Parameters
    ----------
    result : dict
        Output from cluster_permutation_test()
    times : array-like, optional
        Time values for x-axis. If None, uses indices.
    data_df : pd.DataFrame, optional
        Original data for plotting condition means
    condition_column : str, optional
        Condition column name (needed if plotting data)
    data_columns : list, optional
        Data columns (needed if plotting data)
    colors : list of str, optional
        Colors for each condition. Default: ['#1f77b4', '#ff7f0e']
    labels : list of str, optional
        Labels for conditions. Default: uses condition_values
    figsize : tuple, default=(20, 15)
        Figure size
    smoothing_window : int, default=11
        Window size for smoothing plotted data
    linewidth : float, default=8
        Line width for condition plots
    vline_positions : list, optional
        X positions for vertical dashed lines
    cluster_color : str, default='#359937'
        Color for highlighting significant clusters
    show_null_dist : bool, default=True
        Whether to show null distribution plot
    show_tstats : bool, default=True
        Whether to show t-statistics plot
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    # Determine number of subplots
    n_plots = sum([show_null_dist, show_tstats, data_df is not None])
    if n_plots == 0:
        raise ValueError("Must plot at least one panel")
    
    # Setup
    n_timepoints = len(result['observed_stats'])
    if times is None:
        times = np.arange(n_timepoints)
    
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e']
    
    if labels is None:
        labels = list(result['condition_values'])
    
    # Create figure
    set_figure_size(figsize[0], figsize[1])
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Null distribution (if requested)
    if show_null_dist:
        ax = axes[plot_idx]
        ax.hist(result['null_distribution'], bins=50, alpha=0.7, color='gray',
                edgecolor='black')
        ax.axvline(result['largest_cluster_stat'], color='red', linestyle='--',
                   linewidth=5, label=f"Observed (p={result['p_value']:.4f})")
        
        # Apply styling
        ax.tick_params(labelsize=40)
        ax.set_xlabel('Cluster Statistic', fontsize=50)
        ax.set_ylabel('Frequency', fontsize=50)
        ax.set_title('Null Distribution from Permutation Test', fontsize=50)
        ax.legend(fontsize=40)
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_color('#666666')
            spine.set_linewidth(5)
        
        plot_idx += 1
    
    # Plot 2: T-statistics with clusters (if requested)
    if show_tstats:
        ax = axes[plot_idx]
        ax.plot(times, result['observed_stats'], 'k-', linewidth=linewidth)
        ax.axhline(0, color='grey', linestyle='--', linewidth=5)
        
        # Highlight significant clusters
        for cluster in result['observed_clusters']:
            if len(cluster) > 0:
                y_min, y_max = ax.get_ylim()
                ax.fill_between(
                    [times[cluster[0]], times[cluster[-1]]],
                    y_min, y_max,
                    alpha=0.2, 
                    color=cluster_color
                )
        
        # Vertical lines
        if vline_positions is not None:
            y_min, y_max = ax.get_ylim()
            ax.vlines(vline_positions, y_min, y_max, 
                     colors='grey', linestyles='dashed', linewidth=5)
        
        # Apply styling
        ax.tick_params(labelsize=40)
        ax.set_xlabel('Time (s)', fontsize=50)
        ax.set_ylabel('T-statistic', fontsize=50)
        ax.set_title('Observed Statistics with Significant Clusters', fontsize=50)
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_color('#666666')
            spine.set_linewidth(5)
        
        plot_idx += 1
    
    # Plot 3: Data means (if provided)
    if data_df is not None and condition_column is not None:
        ax = axes[plot_idx]
        
        cond1_name, cond2_name = result['condition_values']
        
        if data_columns is None:
            data_columns = data_df.select_dtypes(include=[np.number]).columns
        
        cond1_data = data_df[data_df[condition_column] == cond1_name][data_columns]
        cond2_data = data_df[data_df[condition_column] == cond2_name][data_columns]
        
        # Calculate means and confidence intervals
        mean1 = cond1_data.mean(axis=0).values
        sem1 = cond1_data.sem(axis=0).values
        mean2 = cond2_data.mean(axis=0).values
        sem2 = cond2_data.sem(axis=0).values
        
        upper1 = mean1 + sem1
        lower1 = mean1 - sem1
        upper2 = mean2 + sem2
        lower2 = mean2 - sem2
        
        # Smooth if window provided
        if smoothing_window is not None and smoothing_window > 1:
            mean1 = smooth_signal(mean1, window_len=smoothing_window)
            mean2 = smooth_signal(mean2, window_len=smoothing_window)
            upper1 = smooth_signal(upper1, window_len=smoothing_window)
            lower1 = smooth_signal(lower1, window_len=smoothing_window)
            upper2 = smooth_signal(upper2, window_len=smoothing_window)
            lower2 = smooth_signal(lower2, window_len=smoothing_window)
        
        # Plot condition 1
        ax.plot(times, mean1, color=colors[0], label=labels[0], linewidth=linewidth)
        ax.fill_between(times, lower1, upper1, alpha=0.2, color=colors[0])
        
        # Plot condition 2
        ax.plot(times, mean2, color=colors[1], label=labels[1], linewidth=linewidth)
        ax.fill_between(times, lower2, upper2, alpha=0.2, color=colors[1])
        
        # Highlight significant clusters
        for cluster in result['observed_clusters']:
            if len(cluster) > 0:
                y_min, y_max = ax.get_ylim()
                ax.fill_between(
                    [times[cluster[0]], times[cluster[-1]]],
                    y_min, y_max,
                    alpha=0.2,
                    color=cluster_color
                )
        
        # Vertical lines
        if vline_positions is not None:
            y_min, y_max = ax.get_ylim()
            ax.vlines(vline_positions, y_min, y_max,
                     colors='grey', linestyles='dashed', linewidth=5)
        
        # Apply styling
        ax.tick_params(labelsize=40)
        ax.set_xlabel('Time (s)', fontsize=50)
        ax.set_ylabel('Pupil Size', fontsize=50)
        ax.set_title(f'Condition Means (p={result["p_value"]:.4f})', fontsize=50)
        ax.legend(fontsize=40)
        
        # Set tick locators
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_color('#666666')
            spine.set_linewidth(5)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.28)
    
    return fig

