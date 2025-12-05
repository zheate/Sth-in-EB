
from copy import deepcopy
import numpy as np
from application.models.laser_diode_calculation import LaserDiodeCalculation
from application.models.parallel_utils import ParallelConfig
from application.models.parameters_conversion import parameters_convert

def optimization_objective(params, config):
    """
    Optimization objective function.
    Must be top-level for multiprocessing pickling.
    """
    # Update config with new params
    temp_config = deepcopy(config)
    # SAC remains unchanged from config
    temp_config['coupling_lens_effective_focal_length_f']['value'] = params[0]
    temp_config['coupling_lens_effective_focal_length_s']['value'] = params[1]
    
    try:
        # Convert parameters
        calc_data = parameters_convert(temp_config)
        
        # Disable internal parallelism since differential_evolution is already parallel
        # This prevents spawning too many processes (nested parallelism)
        calculation = LaserDiodeCalculation(calc_data, parallel_config=ParallelConfig.disabled())
        
        far_near = calculation.na_and_coupling_calculate()
        
        # Extract metrics
        na_arr = far_near[6]
        na_ratio_arr = far_near[7]
        coupling_efficiency = far_near[15]
        cladding_ratio = far_near[16]
        
        # Calculate NA for 95% energy
        na_95 = 0.22 # Default fallback
        for n, r in zip(na_arr, na_ratio_arr):
            if r >= 0.95:
                na_95 = n
                break
        
        # Get energy ratio at fiber NA
        fiber_na = temp_config['fiber_na']['value']
        na_ratio_fiber = 0.0
        if len(na_arr) > 0:
            idx = (np.abs(na_arr - fiber_na)).argmin()
            na_ratio_fiber = na_ratio_arr[idx]

        # Objective function (minimize)
        # Priority:
        # 1. Coupling Efficiency (Max) -> -1000 * efficiency
        # 2. NA (Min) -> +10 * na_95
        # 3. NA Energy Ratio (Max) -> -1 * na_ratio_fiber
        # 4. Cladding Light (Min) -> +1 * cladding_ratio
        
        score = -1000 * coupling_efficiency + 10 * na_95 - 1 * na_ratio_fiber + 1 * cladding_ratio
        return score
        
    except Exception:
        return 1e6 # Penalty for failure
