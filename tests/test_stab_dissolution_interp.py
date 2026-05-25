import numpy as np
import sys, os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
# Add project root to Python path
project_root = Path(__file__).resolve().parents[1] / "src"  # repo src/ for the editable package
sys.path.append(str(project_root))

from toddlers.constants import MYR_TO_SEC

def create_slice_plot(x_arr, y_arr, x_label, y_label, dissolution_times_myr, title, fixed_params, 
                     x_scale='log', y_scale='linear', output_path=None):
    """Create a single slice plot through parameter space."""
    plt.figure(figsize=(10, 8))
    
    if x_scale == 'log':
        x_data = 10**x_arr
    else:
        x_data = x_arr
        
    if y_scale == 'log':
        y_data = 10**y_arr
    else:
        y_data = y_arr
        
    x_mesh, y_mesh = np.meshgrid(x_data, y_data)
    
    plt.pcolormesh(x_mesh, y_mesh, dissolution_times_myr, 
                   shading='auto', cmap='viridis')
    cb = plt.colorbar(label='Dissolution Time (Myr)')
    cb.ax.tick_params(labelsize=10)
    
    if x_scale == 'log':
        plt.xscale('log')
    if y_scale == 'log':
        plt.yscale('log')
        
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # Add fixed parameter info to title
    fixed_info = ", ".join([f"{k}={v:.2f}" for k, v in fixed_params.items()])
    plt.title(f'{title}\n({fixed_info})', fontsize=14, pad=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def test_dissolution_interpolant(interpolant_path):
    """
    Test dissolution time interpolant by visualizing predictions across parameter space.
    
    Args:
        interpolant_path (str|Path): Path to dissolution_time_interp.pkl file
    """
    # Load interpolant
    with open(interpolant_path, 'rb') as f:
        interpolator = pickle.load(f)

    # Get parameter ranges from interpolator points
    log_Z_arr = interpolator.grid[0]  # log10 Z values
    eta_arr = interpolator.grid[1]    # linear eta values  
    log_n_arr = interpolator.grid[2]  # log10 n values
    logM_arr = interpolator.grid[3]   # log10 M values

    # Print actual ranges and values
    print("\nParameter Ranges and Values:")
    print("\nlog10(Z) values:", log_Z_arr)
    print("Z values:", 10**log_Z_arr)
    print(f"Z range: {10**log_Z_arr[0]:.3e} to {10**log_Z_arr[-1]:.3e}")
    
    print("\neta_sf values:", eta_arr)
    print(f"eta range: {eta_arr[0]:.3f} to {eta_arr[-1]:.3f}")
    
    print("\nlog10(n) values:", log_n_arr)
    print("n values:", 10**log_n_arr)
    print(f"n range: {10**log_n_arr[0]:.1f} to {10**log_n_arr[-1]:.1f}")
    
    print("\nlog10(M) values:", logM_arr)
    print(f"logM range: {logM_arr[0]:.2f} to {logM_arr[-1]:.2f}")

    # Create visualization grid
    num_points = 50
    
    # Create output directory for plots
    output_dir = Path(interpolant_path).parent / 'dissolution_plots'
    output_dir.mkdir(exist_ok=True)
    
    # 1. Z-eta plots at different n, M combinations
    log_Z_grid = np.linspace(log_Z_arr[0], log_Z_arr[-1], num_points)
    eta_grid = np.linspace(eta_arr[0], eta_arr[-1], num_points)
    log_Z_mesh, eta_mesh = np.meshgrid(log_Z_grid, eta_grid)
    
    for n_val in [log_n_arr[0], np.median(log_n_arr), log_n_arr[-1]]:
        for M_val in [logM_arr[0], np.median(logM_arr), logM_arr[-1]]:
            points = np.column_stack((
                log_Z_mesh.flatten(),
                eta_mesh.flatten(),
                np.full_like(log_Z_mesh.flatten(), n_val),
                np.full_like(log_Z_mesh.flatten(), M_val)
            ))
            times = interpolator(points).reshape(log_Z_mesh.shape) / MYR_TO_SEC
            
            create_slice_plot(
                log_Z_grid, eta_grid,
                'Metallicity (Z)', 'Star Formation Efficiency',
                times,
                'Dissolution Times (Z-eta plane)',
                {'log10(n)': n_val, 'log10(M)': M_val},
                x_scale='log',
                output_path=output_dir / f'Z_eta_n{n_val:.1f}_M{M_val:.1f}.png'
            )
            
    # 2. Z-n plots at different eta, M combinations
    log_n_grid = np.linspace(log_n_arr[0], log_n_arr[-1], num_points)
    log_Z_mesh, log_n_mesh = np.meshgrid(log_Z_grid, log_n_grid)
    
    for eta in [eta_arr[0], np.median(eta_arr), eta_arr[-1]]:
        for M_val in [logM_arr[0], np.median(logM_arr), logM_arr[-1]]:
            points = np.column_stack((
                log_Z_mesh.flatten(),
                np.full_like(log_Z_mesh.flatten(), eta),
                log_n_mesh.flatten(),
                np.full_like(log_Z_mesh.flatten(), M_val)
            ))
            times = interpolator(points).reshape(log_Z_mesh.shape) / MYR_TO_SEC
            
            create_slice_plot(
                log_Z_grid, log_n_grid,
                'Metallicity (Z)', 'Number Density (cm⁻³)',
                times,
                'Dissolution Times (Z-n plane)',
                {'eta_sf': eta, 'log10(M)': M_val},
                x_scale='log', y_scale='log',
                output_path=output_dir / f'Z_n_eta{eta:.2f}_M{M_val:.1f}.png'
            )
            
    # 3. Z-M plots at different eta, n combinations
    logM_grid = np.linspace(logM_arr[0], logM_arr[-1], num_points)
    log_Z_mesh, logM_mesh = np.meshgrid(log_Z_grid, logM_grid)
    
    for eta in [eta_arr[0], np.median(eta_arr), eta_arr[-1]]:
        for n_val in [log_n_arr[0], np.median(log_n_arr), log_n_arr[-1]]:
            points = np.column_stack((
                log_Z_mesh.flatten(),
                np.full_like(log_Z_mesh.flatten(), eta),
                np.full_like(log_Z_mesh.flatten(), n_val),
                logM_mesh.flatten()
            ))
            times = interpolator(points).reshape(log_Z_mesh.shape) / MYR_TO_SEC
            
            create_slice_plot(
                log_Z_grid, logM_grid,
                'Metallicity (Z)', 'log10(Mass/M_sun)',
                times,
                'Dissolution Times (Z-M plane)',
                {'eta_sf': eta, 'log10(n)': n_val},
                x_scale='log',
                output_path=output_dir / f'Z_M_eta{eta:.2f}_n{n_val:.1f}.png'
            )
    
    print(f"\nCreated parameter space visualizations in: {output_dir}")
    
    # Test specific parameter combinations
    print("\nTesting predictions at grid endpoints and medians:")
    test_points = []
    
    # Add corner points of parameter space
    for log_Z in [log_Z_arr[0], log_Z_arr[-1]]:
        for eta in [eta_arr[0], eta_arr[-1]]:
            for log_n in [log_n_arr[0], log_n_arr[-1]]:
                for logM in [logM_arr[0], logM_arr[-1]]:
                    test_points.append([log_Z, eta, log_n, logM])
    
    # Add median point
    test_points.append([
        np.median(log_Z_arr),
        np.median(eta_arr),
        np.median(log_n_arr),
        np.median(logM_arr)
    ])
    
    print("\nPredictions at test points:")
    for point in test_points:
        try:
            pred_time = interpolator(point).item()
            pred_time_myr = pred_time / MYR_TO_SEC
            print(f"\nParameters:")
            print(f"  Z = {10**point[0]:.3e}")
            print(f"  eta_sf = {point[1]:.3f}")
            print(f"  n = {10**point[2]:.1f} cm^-3")
            print(f"  log10(M) = {point[3]:.2f}")
            print(f"Predicted dissolution time: {pred_time_myr:.2f} Myr")
        except Exception as e:
            print(f"Error predicting for point {point}: {str(e)}")

if __name__ == "__main__":
    # Replace with path to your interpolant file
    interpolant_path = "dissolution_time_interp.pkl"
    test_dissolution_interpolant(interpolant_path)