"""
Implementación Optimizada del Modelo de Ising con Metropolis-Hastings y Propp-Wilson
Para la tarea de Cadenas de Markov 3

Optimizaciones aplicadas:
- Uso de NumPy vectorizado para operaciones eficientes
- Reemplazo de for loops con while loops donde es apropiado
- Pre-cálculo de valores constantes
- Uso de numba JIT para operaciones críticas
- Tamaños de lattice optimizados: 10x10, 15x15, 20x20
- Cache de exponenciales para Metropolis-Hastings
- Algoritmos más eficientes
"""

import numpy as np
from numba import jit, prange
import random
from typing import Tuple, List, Dict, Optional
import pickle
import time
from functools import lru_cache

# Pre-calcular exponenciales comunes para eficiencia
@lru_cache(maxsize=1000)
def cached_exp(x: float) -> float:
    """Cache de función exponencial para valores comunes"""
    return np.exp(x)

@jit(nopython=True, fastmath=True)
def fast_local_energy_change(lattice: np.ndarray, i: int, j: int, size: int, J: float) -> float:
    """
    Calcula rápidamente el cambio de energía al voltear un spin
    Optimizado con numba JIT compilation
    """
    spin = lattice[i, j]
    # Suma de vecinos con condiciones periódicas (vectorizado)
    neighbors_sum = (lattice[(i-1) % size, j] + 
                    lattice[(i+1) % size, j] + 
                    lattice[i, (j-1) % size] + 
                    lattice[i, (j+1) % size])
    
    # Delta E = 2 * J * spin * sum_neighbors (factor 2 porque cambiamos el signo)
    return 2.0 * J * spin * neighbors_sum

@jit(nopython=True, parallel=True, fastmath=True)
def fast_total_energy(lattice: np.ndarray, size: int, J: float) -> float:
    """Calcula la energía total usando paralelización numba"""
    energy = 0.0
    
    # Paralelizar sobre filas
    for i in prange(size):
        row_energy = 0.0
        for j in range(size):
            spin = lattice[i, j]
            # Solo contar enlaces hacia derecha y abajo para evitar doble conteo
            if j < size - 1:
                row_energy -= J * spin * lattice[i, j + 1]
            else:  # Condición periódica
                row_energy -= J * spin * lattice[i, 0]
                
            if i < size - 1:
                row_energy -= J * spin * lattice[i + 1, j]
            else:  # Condición periódica
                row_energy -= J * spin * lattice[0, j]
        
        energy += row_energy
    
    return energy

class OptimizedIsingModel:
    """Modelo de Ising optimizado en 2D con condiciones de frontera periódicas"""
    
    __slots__ = ['size', 'J', 'B', 'lattice', '_energy_cache_valid', '_cached_energy']
    
    def __init__(self, size: int, J: float = 1.0, B: float = 0.0):
        self.size = size
        self.J = J
        self.B = B
        # Inicializar con distribución aleatoria eficiente
        self.lattice = np.random.choice(np.array([-1, 1], dtype=np.int8), 
                                       size=(size, size))
        self._energy_cache_valid = False
        self._cached_energy = 0.0
    
    def local_energy_change(self, i: int, j: int) -> float:
        """Calcula el cambio de energía al voltear un spin (optimizado)"""
        return fast_local_energy_change(self.lattice, i, j, self.size, self.J)
    
    def total_energy(self) -> float:
        """Calcula la energía total del sistema (con cache)"""
        if not self._energy_cache_valid:
            self._cached_energy = fast_total_energy(self.lattice, self.size, self.J)
            if self.B != 0:
                self._cached_energy -= self.B * np.sum(self.lattice)
            self._energy_cache_valid = True
        return self._cached_energy
    
    def magnetization(self) -> float:
        """Calcula la magnetización total (vectorizado)"""
        return float(np.sum(self.lattice))
    
    def flip_spin(self, i: int, j: int) -> None:
        """Voltea un spin e invalida cache de energía"""
        self.lattice[i, j] *= -1
        self._energy_cache_valid = False
    
    def copy(self):
        """Crea una copia eficiente del modelo"""
        new_model = OptimizedIsingModel(self.size, self.J, self.B)
        new_model.lattice = self.lattice.copy()
        new_model._energy_cache_valid = self._energy_cache_valid
        new_model._cached_energy = self._cached_energy
        return new_model

class OptimizedMetropolisHastings:
    """Implementación optimizada del algoritmo Metropolis-Hastings"""
    
    __slots__ = ['model', '_beta_cache', '_max_delta_e']
    
    def __init__(self, model: OptimizedIsingModel):
        self.model = model
        self._beta_cache = {}
        # Pre-calcular el máximo cambio de energía posible
        self._max_delta_e = 8 * abs(model.J)  # 4 vecinos * 2 * J
    
    def _get_acceptance_prob(self, delta_e: float, beta: float) -> float:
        """Calcula probabilidad de aceptación con cache"""
        if delta_e <= 0:
            return 1.0
        
        # Cache para combinaciones comunes de beta y delta_e
        cache_key = (round(beta, 3), round(delta_e, 3))
        if cache_key not in self._beta_cache:
            self._beta_cache[cache_key] = np.exp(-beta * delta_e)
        
        return self._beta_cache[cache_key]
    
    def step(self, beta: float) -> bool:
        """Un paso optimizado del algoritmo Metropolis-Hastings"""
        # Selección aleatoria eficiente
        i = random.randrange(self.model.size)
        j = random.randrange(self.model.size)
        
        # Calcular cambio de energía directamente
        delta_e = self.model.local_energy_change(i, j)
        
        # Criterio de aceptación optimizado
        acceptance_prob = self._get_acceptance_prob(delta_e, beta)
        
        if random.random() < acceptance_prob:
            self.model.flip_spin(i, j)
            return True
        return False
    
    def run(self, beta: float, steps: int, burn_in: int = 0) -> OptimizedIsingModel:
        """Ejecuta el algoritmo con optimizaciones"""
        # Burn-in con while loop
        burn_count = 0
        while burn_count < burn_in:
            self.step(beta)
            burn_count += 1
        
        # Pasos principales con while loop optimizado
        accepted = 0
        step_count = 0
        while step_count < steps:
            if self.step(beta):
                accepted += 1
            step_count += 1
        
        return self.model.copy()

class OptimizedProppWilson:
    """Implementación optimizada del algoritmo Propp-Wilson"""
    
    __slots__ = ['size', 'J', 'B', '_rng_state']
    
    def __init__(self, size: int, J: float = 1.0, B: float = 0.0):
        self.size = size
        self.J = J
        self.B = B
        self._rng_state = np.random.RandomState()
    
    def create_extremal_states(self) -> Tuple[OptimizedIsingModel, OptimizedIsingModel]:
        """Crea estados extremos optimizados"""
        model_up = OptimizedIsingModel(self.size, self.J, self.B)
        model_down = OptimizedIsingModel(self.size, self.J, self.B)
        
        # Llenar eficientemente
        model_up.lattice.fill(1)
        model_down.lattice.fill(-1)
        
        return model_up, model_down
    
    def states_coalesced(self, model1: OptimizedIsingModel, model2: OptimizedIsingModel) -> bool:
        """Verifica coalescencia usando operaciones vectorizadas"""
        return np.array_equal(model1.lattice, model2.lattice)
    
    def apply_transition_batch(self, model_up: OptimizedIsingModel, 
                              model_down: OptimizedIsingModel, 
                              beta: float, sites: List[Tuple[int, int]], 
                              random_vals: np.ndarray) -> None:
        """Aplica transiciones en lote para eficiencia"""
        site_idx = 0
        while site_idx < len(sites):
            i, j = sites[site_idx]
            random_val = random_vals[site_idx]
            
            # Aplicar a modelo up
            delta_e_up = model_up.local_energy_change(i, j)
            prob_accept_up = 1.0 if delta_e_up <= 0 else np.exp(-beta * delta_e_up)
            
            if random_val < prob_accept_up:
                model_up.flip_spin(i, j)
            
            # Aplicar a modelo down  
            delta_e_down = model_down.local_energy_change(i, j)
            prob_accept_down = 1.0 if delta_e_down <= 0 else np.exp(-beta * delta_e_down)
            
            if random_val < prob_accept_down:
                model_down.flip_spin(i, j)
            
            site_idx += 1
    
    def sample(self, beta: float, max_time: int = 1000) -> OptimizedIsingModel:
        """Genera muestra perfecta optimizada"""
        T = 1
        
        while T <= max_time:
            model_up, model_down = self.create_extremal_states()
            
            # Pre-generar todos los números aleatorios
            total_sites = self.size * self.size * T
            self._rng_state.seed(42)  # Reproducibilidad
            
            t = 0
            while t < T:
                # Generar sitios y valores aleatorios en lotes
                sites = [(i, j) for i in range(self.size) for j in range(self.size)]
                self._rng_state.shuffle(sites)
                random_vals = self._rng_state.random(len(sites))
                
                # Aplicar transiciones en lote
                self.apply_transition_batch(model_up, model_down, beta, sites, random_vals)
                
                t += 1
            
            # Verificar coalescencia
            if self.states_coalesced(model_up, model_down):
                return model_up.copy()
            
            # Doblar tiempo
            T <<= 1  # Bitshift más eficiente que T *= 2
        
        # Fallback optimizado
        print(f"Advertencia: No coalescencia en tiempo {max_time} para beta={beta}")
        fallback_model = OptimizedIsingModel(self.size, self.J, self.B)
        mh = OptimizedMetropolisHastings(fallback_model)
        return mh.run(beta, 10000, 1000)

def run_optimized_experiments():
    """Ejecuta experimentos optimizados con mejores prácticas"""
    
    # Parámetros optimizados
    lattice_sizes = [10, 15, 20]  # Reducido a tamaños más eficientes
    beta_values = np.arange(0, 1.1, 0.1)  # Más eficiente que list comprehension
    n_samples = 100
    mh_steps = 100000
    
    print("=== Experimentos Optimizados del Modelo de Ising ===")
    print(f"Tamaños de lattice: {lattice_sizes}")
    print(f"Valores de β: {beta_values}")
    print(f"Número de muestras: {n_samples}")
    print(f"Pasos MH por muestra: {mh_steps}")
    print()
    
    print("Distribución de Boltzmann con temperatura inversa β:")
    print("π(σ) = exp(-β * H(σ)) / Z(β)")
    print("donde H(σ) = -J∑σᵢσⱼ con J=1, B=0")
    print()
    
    # Pre-alocar estructura de resultados
    results = {
        'metropolis_hastings': {size: {} for size in lattice_sizes},
        'propp_wilson': {size: {} for size in lattice_sizes},
        'parameters': {
            'lattice_sizes': lattice_sizes,
            'beta_values': beta_values.tolist(),
            'n_samples': n_samples,
            'mh_steps': mh_steps,
            'J': 1.0,
            'B': 0.0
        }
    }
    
    size_idx = 0
    while size_idx < len(lattice_sizes):
        size = lattice_sizes[size_idx]
        print(f"\n--- Lattice {size}x{size} ---")
        
        beta_idx = 0
        while beta_idx < len(beta_values):
            beta = beta_values[beta_idx]
            print(f"β = {beta:.1f}")
            
            # Metropolis-Hastings optimizado
            mh_samples = []
            mh_start_time = time.perf_counter()  # Más preciso que time()
            
            sample_idx = 0
            while sample_idx < n_samples:
                if (sample_idx + 1) % 25 == 0:
                    elapsed = time.perf_counter() - mh_start_time
                    eta = elapsed * n_samples / (sample_idx + 1) - elapsed
                    print(f"  MH muestra {sample_idx + 1}/{n_samples} - ETA: {eta:.1f}s")
                
                model = OptimizedIsingModel(size, J=1.0, B=0.0)
                mh = OptimizedMetropolisHastings(model)
                final_config = mh.run(beta, mh_steps, burn_in=10000)
                
                mh_samples.append({
                    'lattice': final_config.lattice.copy(),
                    'magnetization': final_config.magnetization(),
                    'energy': final_config.total_energy()
                })
                
                sample_idx += 1
            
            mh_time = time.perf_counter() - mh_start_time
            
            # Propp-Wilson optimizado
            pw_samples = []
            pw_start_time = time.perf_counter()
            
            sample_idx = 0
            while sample_idx < n_samples:
                if (sample_idx + 1) % 25 == 0:
                    elapsed = time.perf_counter() - pw_start_time
                    eta = elapsed * n_samples / (sample_idx + 1) - elapsed
                    print(f"  PW muestra {sample_idx + 1}/{n_samples} - ETA: {eta:.1f}s")
                
                pw = OptimizedProppWilson(size, J=1.0, B=0.0)
                final_config = pw.sample(beta, max_time=1000)
                
                pw_samples.append({
                    'lattice': final_config.lattice.copy(),
                    'magnetization': final_config.magnetization(),
                    'energy': final_config.total_energy()
                })
                
                sample_idx += 1
            
            pw_time = time.perf_counter() - pw_start_time
            
            # Almacenar resultados
            results['metropolis_hastings'][size][beta] = {
                'samples': mh_samples,
                'computation_time': mh_time
            }
            
            results['propp_wilson'][size][beta] = {
                'samples': pw_samples,
                'computation_time': pw_time
            }
            
            print(f"  Tiempos: MH={mh_time:.1f}s, PW={pw_time:.1f}s")
            
            # Guardar progreso parcial cada 5 experimentos
            if (beta_idx + 1) % 5 == 0:
                with open('ising_results_partial.pkl', 'wb') as f:
                    pickle.dump(results, f)
                print(f"  >> Progreso guardado")
            
            beta_idx += 1
        
        size_idx += 1
    
    # Guardar resultados finales
    with open('ising_results_optimized.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n=== Experimentos Optimizados Completados ===")
    print("Resultados guardados en 'ising_results_optimized.pkl'")
    
    return results

def analyze_optimized_results(results: Dict):
    """Análisis optimizado de resultados con estadísticas mejoradas"""
    print("\n=== Análisis Optimizado de Resultados ===")
    
    sizes = results['parameters']['lattice_sizes'] 
    betas = results['parameters']['beta_values']
    
    # Pre-calcular estadísticas para eficiencia
    stats_cache = {}
    
    size_idx = 0
    while size_idx < len(sizes):
        size = sizes[size_idx]
        print(f"\nLattice {size}x{size}:")
        
        beta_idx = 0
        while beta_idx < len(betas):
            beta = betas[beta_idx]
            
            # Cache key para evitar recálculos
            cache_key = (size, beta)
            
            if cache_key not in stats_cache:
                mh_data = results['metropolis_hastings'][size][beta]
                pw_data = results['propp_wilson'][size][beta]
                
                # Vectorizar cálculos de estadísticas
                mh_mags = np.array([s['magnetization'] for s in mh_data['samples']])
                pw_mags = np.array([s['magnetization'] for s in pw_data['samples']])
                mh_energies = np.array([s['energy'] for s in mh_data['samples']])
                pw_energies = np.array([s['energy'] for s in pw_data['samples']])
                
                stats_cache[cache_key] = {
                    'mh_mag_mean': np.mean(mh_mags),
                    'pw_mag_mean': np.mean(pw_mags),
                    'mh_energy_mean': np.mean(mh_energies),
                    'pw_energy_mean': np.mean(pw_energies),
                    'mh_mag_std': np.std(mh_mags),
                    'pw_mag_std': np.std(pw_mags)
                }
            
            stats = stats_cache[cache_key]
            print(f"  β={beta:.1f}: MH_mag={stats['mh_mag_mean']:.2f}±{stats['mh_mag_std']:.2f}, "
                  f"PW_mag={stats['pw_mag_mean']:.2f}±{stats['pw_mag_std']:.2f}, "
                  f"MH_E={stats['mh_energy_mean']:.2f}, PW_E={stats['pw_energy_mean']:.2f}")
            
            beta_idx += 1
        
        size_idx += 1

if __name__ == "__main__":
    print("Iniciando experimentos optimizados...")
    start_time = time.perf_counter()
    
    # Ejecutar experimentos optimizados
    results = run_optimized_experiments()
    
    # Análisis optimizado
    analyze_optimized_results(results)
    
    total_time = time.perf_counter() - start_time
    print(f"\nTiempo total de ejecución: {total_time:.2f} segundos")