"""
This script benchmarks the performance of the Arnold Cat Map (ACM) encryption
on the GPU. It compares the efficient and inefficient implementations of the
Arnold Cat Map.
"""

import torch
import numpy as np
import time

# --- Original inefficient GPU Arnold Cat Map (for comparison) ---
def arnold_gpu_inefficient(matrix_in, key):
    N_iter, a, b, c, d = key
    
    if not isinstance(matrix_in, torch.Tensor):
        matrix = torch.from_numpy(matrix_in).cuda()
    elif not matrix_in.is_cuda:
        matrix = matrix_in.cuda()
    else:
        matrix = matrix_in # Assume it's already a CUDA tensor

    h, w = matrix.shape[:2]

    if h != w:
        raise ValueError("Matrix must be square")
    # Simple check, assumes w > 1. A more robust check:
    # if w <= 1 or pow(a * d - b * c, -1, w) is not well-defined (i.e., gcd(ad-bc, w) != 1)
    # For the specific key (1,1,1,2) -> ad-bc = 1, so it's fine for w > 1.
    # if (a * d - b * c) % w != 1 and w > 1 : # Allow w=1 for tiny test cases if ad-bc could be non-1
    #     raise ValueError(f"Invalid Arnold key: determinant {(a * d - b * c)} must be 1 (mod {w})")

    # Ensure matrix is float for consistent processing if it might be int
    matrix = matrix.float() 

    y_coords, x_coords = torch.meshgrid(torch.arange(h, device='cuda', dtype=torch.long), 
                                        torch.arange(w, device='cuda', dtype=torch.long),
                                        indexing='ij')
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()
    
    matrix_flat = matrix.reshape(h * w, -1) # Shape (H*W, num_channels) or (H*W, 1)
    # new_flat buffer needs to be distinct if matrix_flat is used as source and dest in loop
    new_flat_buffer = torch.zeros_like(matrix_flat)


    # These are the coordinates that will be transformed and used for scatter *destinations*
    current_x_dest = x_coords_flat.clone()
    current_y_dest = y_coords_flat.clone()

    # This loop applies the transformation N times, with a scatter at each step
    for _ in range(N_iter):
        # The coordinates to transform are updated at the end of the loop
        next_x_dest = (a * current_x_dest + b * current_y_dest) % w
        next_y_dest = (c * current_x_dest + d * current_y_dest) % h
        
        dest_indices_flat = (next_y_dest * w + next_x_dest).long()

        # Scatter: new_flat_buffer[dest_indices_flat[k],:] = matrix_flat[k,:] (conceptually)
        # matrix_flat is the source from the previous step (or original)
        # dest_indices_flat are the target locations for elements from matrix_flat's current order
        new_flat_buffer.index_copy_(0, dest_indices_flat, matrix_flat)
        
        matrix_flat = new_flat_buffer.clone() # Result of this scatter becomes input for next

        # Update coordinates for the *next* transformation step
        current_x_dest = next_x_dest 
        current_y_dest = next_y_dest
        # This is what makes it N-scatter. If we were transforming original coords N times,
        # we wouldn't update current_x_dest/current_y_dest with the intermediate results.

    return matrix_flat.reshape(matrix.shape)

def iarnold_gpu_inefficient(matrix_in, key):
    N_iter, a, b, c, d = key
    h, w = matrix_in.shape[:2]
    
    inv_a = d % w
    inv_b = (-b % w + w) % w 
    inv_c = (-c % w + w) % w 
    inv_d = a % w
    
    inv_key = (N_iter, inv_a, inv_b, inv_c, inv_d)
    return arnold_gpu_inefficient(matrix_in, inv_key)

# --- Efficient GPU Arnold Cat Map (1-Scatter) ---
def arnold_gpu_efficient(matrix_in, key):
    """
    Applies Arnold Cat Map efficiently on GPU.
    Calculates final coordinates after N iterations, then scatters data once.
    """
    N_iter, a, b, c, d = key
    
    if not isinstance(matrix_in, torch.Tensor):
        # If numpy, convert to tensor and send to GPU
        matrix = torch.from_numpy(matrix_in).cuda()
    elif not matrix_in.is_cuda:
        matrix = matrix_in.cuda()
    else:
        matrix = matrix_in # Assume it's already a CUDA tensor
    
    # Ensure matrix is float for consistent processing
    matrix = matrix.float()

    h, w = matrix.shape[:2]
    if h != w:
        raise ValueError("Matrix must be square for this ACM implementation.")

    # Original coordinates (long type for direct use as indices later)
    y_orig_grid, x_orig_grid = torch.meshgrid(
        torch.arange(h, device='cuda', dtype=torch.long),
        torch.arange(w, device='cuda', dtype=torch.long),
        indexing='ij'
    )
    # These are the initial coordinates of pixels that we will transform
    x_coords_to_transform_flat = x_orig_grid.flatten() # Shape (H*W)
    y_coords_to_transform_flat = y_orig_grid.flatten() # Shape (H*W)

    # Transform these initial coordinates N times to find their final destinations
    final_dest_x = x_coords_to_transform_flat.clone()
    final_dest_y = y_coords_to_transform_flat.clone()

    for _ in range(N_iter):
        next_dest_x = (a * final_dest_x + b * final_dest_y) % w
        next_dest_y = (c * final_dest_x + d * final_dest_y) % h 
        final_dest_x = next_dest_x
        final_dest_y = next_dest_y
    
    # final_dest_x[k] and final_dest_y[k] are the final (x,y) for the pixel
    # that was originally at x_coords_to_transform_flat[k], y_coords_to_transform_flat[k] (i.e., original flat index k).
    # So, we need to map original_matrix_flat[k] to output_flat[final_dest_flat_indices[k]]
    final_dest_flat_indices = (final_dest_y * w + final_dest_x).long() # Shape (H*W)

    original_matrix_flat = matrix.reshape(h * w, -1) # Shape (H*W, num_channels) or (H*W, 1)
    output_flat = torch.zeros_like(original_matrix_flat)

    # Scatter: output_flat[final_dest_flat_indices[k], :] = original_matrix_flat[k, :]
    # This means the element from original flat position 'k' goes to 'final_dest_flat_indices[k]'
    output_flat.index_copy_(0, final_dest_flat_indices, original_matrix_flat)
    
    return output_flat.reshape(matrix.shape)

def iarnold_gpu_efficient(matrix_in, key):
    """Inverse ACM for the efficient GPU version."""
    N_iter, a, b, c, d = key
    h, w = matrix_in.shape[:2]
    inv_a = d % w
    inv_b = (-b % w + w) % w
    inv_c = (-c % w + w) % w
    inv_d = a % w
    inv_key = (N_iter, inv_a, inv_b, inv_c, inv_d)
    return arnold_gpu_efficient(matrix_in, inv_key)


# --- Efficient CPU Arnold Cat Map (1-Scatter) ---
def arnold_cpu_efficient(matrix_in, key):
    """
    Applies Arnold Cat Map efficiently on CPU.
    Calculates final coordinates after N iterations, then scatters data once.
    """
    N_iter, a, b, c, d = key

    if not isinstance(matrix_in, torch.Tensor):
        matrix = torch.from_numpy(matrix_in)
    else:
        matrix = matrix_in

    # Ensure matrix is float for consistent processing
    matrix = matrix.float()

    h, w = matrix.shape[:2]
    if h != w:
        raise ValueError("Matrix must be square for this ACM implementation.")

    device = matrix.device

    # Original coordinates (long type for direct use as indices later)
    y_orig_grid, x_orig_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.long),
        torch.arange(w, device=device, dtype=torch.long),
        indexing='ij'
    )
    # These are the initial coordinates of pixels that we will transform
    x_coords_to_transform_flat = x_orig_grid.flatten()  # Shape (H*W)
    y_coords_to_transform_flat = y_orig_grid.flatten()  # Shape (H*W)

    # Transform these initial coordinates N times to find their final destinations
    final_dest_x = x_coords_to_transform_flat.clone()
    final_dest_y = y_coords_to_transform_flat.clone()

    for _ in range(N_iter):
        next_dest_x = (a * final_dest_x + b * final_dest_y) % w
        next_dest_y = (c * final_dest_x + d * final_dest_y) % h
        final_dest_x = next_dest_x
        final_dest_y = next_dest_y

    # final_dest_x[k] and final_dest_y[k] are the final (x,y) for the pixel
    # that was originally at x_coords_to_transform_flat[k], y_coords_to_transform_flat[k] (i.e., original flat index k).
    # So, we need to map original_matrix_flat[k] to output_flat[final_dest_flat_indices[k]]
    final_dest_flat_indices = (final_dest_y * w + final_dest_x).long()  # Shape (H*W)

    original_matrix_flat = matrix.reshape(h * w, -1)  # Shape (H*W, num_channels) or (H*W, 1)
    output_flat = torch.zeros_like(original_matrix_flat)

    # Scatter: output_flat[final_dest_flat_indices[k], :] = original_matrix_flat[k, :]
    # This means the element from original flat position 'k' goes to 'final_dest_flat_indices[k]'
    output_flat.index_copy_(0, final_dest_flat_indices, original_matrix_flat)

    return output_flat.reshape(matrix.shape)


def iarnold_cpu_efficient(matrix_in, key):
    """Inverse ACM for the efficient CPU version."""
    N_iter, a, b, c, d = key
    h, w = matrix_in.shape[:2]
    inv_a = d % w
    inv_b = (-b % w + w) % w
    inv_c = (-c % w + w) % w
    inv_d = a % w
    inv_key = (N_iter, inv_a, inv_b, inv_c, inv_d)
    return arnold_cpu_efficient(matrix_in, inv_key)


# --- Benchmarking Function (supports both CPU and GPU) ---
def benchmark_acm(acm_func, iacm_func, matrix_size, N_iterations, num_runs=10, num_channels=0, data_type=torch.float32, device='cpu'):
    """
    Benchmark ACM implementations on specified device (CPU or GPU).

    Args:
        acm_func: Arnold encryption function
        iacm_func: Arnold decryption function
        matrix_size: Size of square matrix
        N_iterations: Number of Arnold iterations
        num_runs: Number of runs to average
        num_channels: Number of channels (0 for 2D matrix)
        data_type: Data type for the matrix
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        tuple: (avg_enc_time, avg_dec_time) or (None, None) if failed
    """
    if num_channels > 0:
        shape = (matrix_size, matrix_size, num_channels)
    else:
        shape = (matrix_size, matrix_size)

    # Create data on specified device
    if device == 'cuda' and torch.cuda.is_available():
        data = torch.randn(shape, dtype=data_type, device='cuda')
    else:
        data = torch.randn(shape, dtype=data_type, device='cpu')

    key_params = (1, 1, 1, 2)
    key = (N_iterations, *key_params)

    enc_times = []
    dec_times = []

    # Warm-up run
    try:
        if device == 'cuda':
            torch.cuda.synchronize()
        encrypted_warmup = acm_func(data, key)
        if device == 'cuda':
            torch.cuda.synchronize()
        _ = iacm_func(encrypted_warmup, key)
        if device == 'cuda':
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Error during {device} warm-up for {acm_func.__name__} with size {matrix_size}, N={N_iterations}: {e}")
        return None, None

    for i_run in range(num_runs):
        try:
            # Encryption
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            encrypted_data = acm_func(data, key)
            if device == 'cuda':
                torch.cuda.synchronize()
            enc_times.append(time.perf_counter() - start_time)

            # Decryption
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            decrypted_data = iacm_func(encrypted_data, key)
            if device == 'cuda':
                torch.cuda.synchronize()
            dec_times.append(time.perf_counter() - start_time)

            if not torch.allclose(data, decrypted_data, atol=1e-4):
                print(f"{device.upper()} Verification FAILED for {acm_func.__name__} with size {matrix_size}, N={N_iterations}, run {i_run+1}")
                # return None, None # Stop if verification fails
        except Exception as e:
            print(f"Error during {device} benchmark run {i_run+1} for {acm_func.__name__} with size {matrix_size}, N={N_iterations}: {e}")
            # return None, None

    if not enc_times or not dec_times:
        return None, None

    avg_enc_time = sum(enc_times) / len(enc_times)
    avg_dec_time = sum(dec_times) / len(dec_times)

    return avg_enc_time, avg_dec_time

# --- Main Benchmarking Loop (CPU and GPU) ---
def run_all_acm_benchmarks():
    results = []
    matrix_sizes = [16, 32, 64, 128, 256, 512, 768, 1024]
    N_iterations_list = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    num_runs_avg = 10 # Number of times to run each specific benchmark for averaging
    channels = 0 # 0 for 2D matrix (e.g. HxW), or 3 for (HxWx3)

    # Define all implementations to benchmark
    implementations = []

    # CPU implementations
    implementations.extend([
        ("CPU Efficient (1-Scatter)", arnold_cpu_efficient, iarnold_cpu_efficient, 'cpu'),
    ])

    # GPU implementations (only if CUDA is available)
    if torch.cuda.is_available():
        implementations.extend([
            ("GPU Efficient (1-Scatter)", arnold_gpu_efficient, iarnold_gpu_efficient, 'cuda'),
            ("GPU Inefficient (N-Scatter)", arnold_gpu_inefficient, iarnold_gpu_inefficient, 'cuda'),
        ])
    else:
        print("CUDA is not available. Skipping GPU benchmarks.")

    print(f"Starting ACM Benchmarks (Averaging over {num_runs_avg} runs each)...")

    total_benchmarks = len(matrix_sizes) * len(N_iterations_list) * len(implementations)
    current_benchmark = 0

    for size in matrix_sizes:
        for N_iter in N_iterations_list:
            for impl_name, acm_f, iacm_f, device in implementations:
                current_benchmark += 1
                print(f"Running Benchmark {current_benchmark}/{total_benchmarks}: {impl_name}, Matrix: {size}x{size}, N: {N_iter}, Channels: {channels if channels > 0 else 'N/A'}")

                avg_enc, avg_dec = benchmark_acm(acm_f, iacm_f, size, N_iter, num_runs=num_runs_avg, num_channels=channels, device=device)

                if avg_enc is not None and avg_dec is not None:
                    results.append({
                        "implementation": impl_name,
                        "device": device,
                        "matrix_size": size,
                        "N_iterations": N_iter,
                        "channels": channels,
                        "avg_enc_time_s": avg_enc,
                        "avg_dec_time_s": avg_dec
                    })
                    print(f"  Avg Enc: {avg_enc:.6f}s, Avg Dec: {avg_dec:.6f}s")
                else:
                    print(f"  Skipped/Failed.")
                print("-" * 40)

    print("\nAll ACM benchmarks completed.")
    return results


# Keep the old function name for backward compatibility
def run_all_gpu_acm_benchmarks():
    """Legacy function name - now calls the new unified benchmarking function."""
    return run_all_acm_benchmarks()

if __name__ == '__main__':
    gpu_acm_results = run_all_gpu_acm_benchmarks()

    if gpu_acm_results:
        try:
            import pandas as pd
            df_results = pd.DataFrame(gpu_acm_results)
            print("\nACM Benchmark Results:")
            # Set pandas to display all rows
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
                print(df_results)

            # Example plotting (requires matplotlib)
            # import matplotlib.pyplot as plt
            # for impl_name, group in df_results.groupby('implementation'):
            #     plt.figure(figsize=(10,6))
            #     for n_iter, subgroup in group.groupby('N_iterations'):
            #         plt.plot(subgroup['matrix_size'], subgroup['avg_enc_time_s'], marker='o', label=f'N={n_iter} Enc')
            #     plt.xlabel('Matrix Size (side length)')
            #     plt.ylabel('Avg Encryption Time (s)')
            #     plt.title(f'GPU ACM Encryption Performance ({impl_name})')
            #     plt.legend()
            #     plt.xscale('log', base=2)
            #     plt.yscale('log')
            #     plt.grid(True, which="both", ls="-.", alpha=0.7)
            #     plt.show()

        except ImportError:
            print("\nPandas not installed. Raw results list:")
            for res in gpu_acm_results:
                print(res)

        # Create visualizations
        try:
            import matplotlib.pyplot as plt
            import os
            os.makedirs("results/analysis", exist_ok=True)

            # Set up the plotting style
            plt.style.use('default')
            colors = {'CPU Efficient (1-Scatter)': '#1f77b4',
                     'GPU Efficient (1-Scatter)': '#2ca02c',
                     'GPU Inefficient (N-Scatter)': '#d62728'}

            # Plot 1: Performance vs Matrix Size for different N_iterations
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            fig.suptitle('ACM Implementation Performance Comparison', fontsize=16, fontweight='bold')

            # Get the current values from the benchmark
            n_iterations_list = sorted(df_results['N_iterations'].unique())
            matrix_sizes = sorted(df_results['matrix_size'].unique())

            # Show first 12 N values (or all if less than 12)
            for idx, n_iter in enumerate(n_iterations_list[:12]):
                if idx >= 12:  # Only show up to 12 subplots
                    break
                row = idx // 4
                col = idx % 4

                ax = axes[row, col]
                subset = df_results[df_results['N_iterations'] == n_iter]

                for impl in df_results['implementation'].unique():
                    impl_data = subset[subset['implementation'] == impl]
                    ax.plot(impl_data['matrix_size'], impl_data['avg_enc_time_s'] * 1000,
                           marker='o', label=f'{impl} (Enc)', color=colors.get(impl, 'black'), linestyle='-')
                    ax.plot(impl_data['matrix_size'], impl_data['avg_dec_time_s'] * 1000,
                           marker='s', label=f'{impl} (Dec)', color=colors.get(impl, 'black'), linestyle='--')

                ax.set_title(f'N = {n_iter} iterations')
                ax.set_xlabel('Matrix Size')
                ax.set_ylabel('Time (ms)')
                ax.legend(fontsize='small', loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log', base=2)

            plt.tight_layout()
            plt.savefig('results/analysis/performance_by_n_iterations.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: results/analysis/performance_by_n_iterations.png")

            # Plot 2: Implementation comparison for largest matrix size
            plt.figure(figsize=(16, 10))
            largest_size = df_results['matrix_size'].max()
            largest_data = df_results[df_results['matrix_size'] == largest_size]

            bar_width = 0.25
            x_pos = range(len(n_iterations_list))

            for i, impl in enumerate(df_results['implementation'].unique()):
                impl_data = largest_data[largest_data['implementation'] == impl]
                enc_times = [impl_data[impl_data['N_iterations'] == n]['avg_enc_time_s'].iloc[0] * 1000
                           for n in n_iterations_list]
                dec_times = [impl_data[impl_data['N_iterations'] == n]['avg_dec_time_s'].iloc[0] * 1000
                           for n in n_iterations_list]

                plt.bar([x + i * bar_width for x in x_pos], enc_times, width=bar_width,
                       label=f'{impl} (Enc)', alpha=0.8, color=colors.get(impl, 'black'))
                plt.bar([x + i * bar_width for x in x_pos], dec_times, width=bar_width,
                       label=f'{impl} (Dec)', alpha=0.6, color=colors.get(impl, 'black'),
                       hatch='//')

            plt.xlabel('Number of Iterations')
            plt.ylabel('Time (ms)')
            plt.title(f'Performance Comparison for {largest_size}x{largest_size} Matrices')
            plt.xticks([x + bar_width for x in x_pos], n_iterations_list)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/analysis/largest_matrix_comparison.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: results/analysis/largest_matrix_comparison.png")

            # Plot 3: Efficiency comparison (Efficient vs Inefficient)
            plt.figure(figsize=(16, 10))

            # Focus on GPU implementations for efficiency comparison
            gpu_data = df_results[df_results['device'] == 'cuda']

            # Use a selection of N_iterations for comparison
            selected_n_iters = sorted(gpu_data['N_iterations'].unique())[::2][:8]  # Every 2nd value, max 8
            n_cols = min(4, len(selected_n_iters))
            n_rows = (len(selected_n_iters) + n_cols - 1) // n_cols

            for i, n_iter in enumerate(selected_n_iters):
                plt.subplot(n_rows, n_cols, i+1)

                subset = gpu_data[gpu_data['N_iterations'] == n_iter]
                efficient = subset[subset['implementation'] == 'GPU Efficient (1-Scatter)']
                inefficient = subset[subset['implementation'] == 'GPU Inefficient (N-Scatter)']

                plt.plot(efficient['matrix_size'], efficient['avg_enc_time_s'] * 1000,
                        'o-', label='Efficient (1-Scatter)', color='#2ca02c', linewidth=2)
                plt.plot(inefficient['matrix_size'], inefficient['avg_enc_time_s'] * 1000,
                        's-', label='Inefficient (N-Scatter)', color='#d62728', linewidth=2)

                plt.title(f'N = {n_iter} Iterations')
                plt.xlabel('Matrix Size')
                plt.ylabel('Encryption Time (ms)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xscale('log', base=2)

            plt.suptitle('GPU Implementation Efficiency Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('results/analysis/gpu_efficiency_comparison.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: results/analysis/gpu_efficiency_comparison.png")

            # Plot 4: CPU vs GPU scaling
            plt.figure(figsize=(12, 6))

            for impl in ['CPU Efficient (1-Scatter)', 'GPU Efficient (1-Scatter)']:
                impl_data = df_results[df_results['implementation'] == impl]
                # Group by matrix size and calculate mean time across all N_iterations
                mean_times = impl_data.groupby('matrix_size')[['avg_enc_time_s', 'avg_dec_time_s']].mean()

                plt.plot(mean_times.index, mean_times['avg_enc_time_s'] * 1000,
                        marker='o', label=f'{impl} (Enc)', linewidth=2)
                plt.plot(mean_times.index, mean_times['avg_dec_time_s'] * 1000,
                        marker='s', label=f'{impl} (Dec)', linewidth=2, linestyle='--')

            plt.xlabel('Matrix Size')
            plt.ylabel('Average Time (ms)')
            plt.title('CPU vs GPU Scaling Performance (Averaged Across All N_Iterations)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xscale('log', base=2)
            plt.yscale('log')
            plt.savefig('results/analysis/cpu_vs_gpu_scaling.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: results/analysis/cpu_vs_gpu_scaling.png")

            print(f"\n📊 Generated 4 visualization plots in results/analysis/")
            print("   - performance_by_n_iterations.png: Performance across different N values")
            print("   - largest_matrix_comparison.png: Comparison for largest matrix size")
            print("   - gpu_efficiency_comparison.png: GPU efficiency comparison")
            print("   - cpu_vs_gpu_scaling.png: CPU vs GPU scaling analysis")

        except ImportError as e:
            print(f"\nWarning: Could not generate plots. Missing dependencies: {e}")
            print("Install matplotlib with: pip install matplotlib")

    elif torch.cuda.is_available():
         print("No GPU results generated, check for errors during benchmark.") 
