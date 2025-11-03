"""
Implementación del Modelo de Ising con Metropolis-Hastings y Propp-Wilson
Para la tarea de Cadenas de Markov 3

Parámetros del experimento:
- Lattice sizes: 10x10, 12x12, 15x15, 17x17, 20x20
- Beta (temperatura inversa): 0, 0.1, 0.2, ..., 0.9, 1.0
- J = 1 (constante de acoplamiento)
- B = 0 (sin campo magnético externo)
- 100 muestras de cada método
- Metropolis-Hastings: 10^5 iteraciones por muestra
"""

import numpy as np
import random
from typing import Tuple, List, Dict
import pickle
import time

class IsingModel:
    """Modelo de Ising en 2D con condiciones de frontera periódicas"""
    
    def __init__(self, size: int, J: float = 1.0, B: float = 0.0):
        self.size = size
        self.J = J  # Constante de acoplamiento
        self.B = B  # Campo magnético externo
        self.lattice = np.random.choice([-1, 1], size=(size, size))
    
    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Obtiene los vecinos con condiciones de frontera periódicas"""
        return [
            ((i-1) % self.size, j),
            ((i+1) % self.size, j),
            (i, (j-1) % self.size),
            (i, (j+1) % self.size)
        ]
    
    def local_energy(self, i: int, j: int) -> float:
        """Calcula la energía local de un spin en posición (i,j)"""
        spin = self.lattice[i, j]
        neighbors_sum = sum(self.lattice[ni, nj] for ni, nj in self.get_neighbors(i, j))
        return -self.J * spin * neighbors_sum - self.B * spin
    
    def total_energy(self) -> float:
        """Calcula la energía total del sistema"""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.lattice[i, j]
                # Solo contar cada par una vez
                if i < self.size - 1:
                    energy -= self.J * spin * self.lattice[i+1, j]
                if j < self.size - 1:
                    energy -= self.J * spin * self.lattice[i, j+1]
                # Condiciones periódicas
                if i == self.size - 1:
                    energy -= self.J * spin * self.lattice[0, j]
                if j == self.size - 1:
                    energy -= self.J * spin * self.lattice[i, 0]
                # Campo magnético
                energy -= self.B * spin
        return energy
    
    def magnetization(self) -> float:
        """Calcula la magnetización total"""
        return np.sum(self.lattice)
    
    def copy(self):
        """Crea una copia del modelo"""
        new_model = IsingModel(self.size, self.J, self.B)
        new_model.lattice = self.lattice.copy()
        return new_model

class MetropolisHastings:
    """Implementación del algoritmo Metropolis-Hastings para el modelo de Ising"""
    
    def __init__(self, model: IsingModel):
        self.model = model
    
    def step(self, beta: float) -> bool:
        """
        Un paso del algoritmo Metropolis-Hastings
        
        Args:
            beta: temperatura inversa (1/kT, con k=1)
        
        Returns:
            bool: True si se aceptó el movimiento, False si se rechazó
        """
        # Seleccionar posición aleatoria
        i = random.randint(0, self.model.size - 1)
        j = random.randint(0, self.model.size - 1)
        
        # Calcular energía antes del flip
        energy_before = self.model.local_energy(i, j)
        
        # Flip del spin
        self.model.lattice[i, j] *= -1
        
        # Calcular energía después del flip
        energy_after = self.model.local_energy(i, j)
        
        # Cambio de energía
        delta_E = energy_after - energy_before
        
        # Criterio de aceptación de Metropolis
        if delta_E <= 0:
            return True  # Aceptar siempre si la energía disminuye
        else:
            # Aceptar con probabilidad exp(-β * ΔE)
            if random.random() < np.exp(-beta * delta_E):
                return True
            else:
                # Rechazar: deshacer el flip
                self.model.lattice[i, j] *= -1
                return False
    
    def run(self, beta: float, steps: int, burn_in: int = 0) -> IsingModel:
        """
        Ejecuta el algoritmo Metropolis-Hastings
        
        Args:
            beta: temperatura inversa
            steps: número de pasos de Monte Carlo
            burn_in: pasos de burn-in (no contados)
        
        Returns:
            IsingModel: configuración final
        """
        # Burn-in
        for _ in range(burn_in):
            self.step(beta)
        
        # Pasos principales
        accepted = 0
        for _ in range(steps):
            if self.step(beta):
                accepted += 1
        
        return self.model.copy()

class ProppWilson:
    """Implementación del algoritmo Propp-Wilson para muestreo perfecto"""
    
    def __init__(self, size: int, J: float = 1.0, B: float = 0.0):
        self.size = size
        self.J = J
        self.B = B
    
    def create_extremal_states(self) -> Tuple[IsingModel, IsingModel]:
        """Crea los estados extremos: todos +1 y todos -1"""
        model_up = IsingModel(self.size, self.J, self.B)
        model_down = IsingModel(self.size, self.J, self.B)
        
        model_up.lattice.fill(1)
        model_down.lattice.fill(-1)
        
        return model_up, model_down
    
    def states_coalesced(self, model1: IsingModel, model2: IsingModel) -> bool:
        """Verifica si dos estados han coalescido"""
        return np.array_equal(model1.lattice, model2.lattice)
    
    def apply_transition(self, model: IsingModel, i: int, j: int, beta: float, 
                        random_val: float) -> None:
        """
        Aplica una transición determinística basada en el valor aleatorio dado
        
        Args:
            model: modelo a modificar
            i, j: posición del spin
            beta: temperatura inversa  
            random_val: valor aleatorio [0,1)
        """
        # Calcular energía antes del flip
        energy_before = model.local_energy(i, j)
        
        # Flip temporal para calcular nueva energía
        model.lattice[i, j] *= -1
        energy_after = model.local_energy(i, j)
        delta_E = energy_after - energy_before
        
        # Probabilidad de aceptación
        if delta_E <= 0:
            prob_accept = 1.0
        else:
            prob_accept = np.exp(-beta * delta_E)
        
        # Decisión determinística basada en random_val
        if random_val < prob_accept:
            # Aceptar (ya hicimos el flip)
            pass
        else:
            # Rechazar: deshacer el flip
            model.lattice[i, j] *= -1
    
    def sample(self, beta: float, max_time: int = 1000) -> IsingModel:
        """
        Genera una muestra perfecta usando Propp-Wilson
        
        Args:
            beta: temperatura inversa
            max_time: tiempo máximo a considerar
        
        Returns:
            IsingModel: muestra perfecta
        """
        T = 1
        
        while T <= max_time:
            # Crear estados extremos
            model_up, model_down = self.create_extremal_states()
            
            # Generar secuencia de números aleatorios para el período [-T, 0]
            random.seed(42)  # Para reproducibilidad en esta ventana de tiempo
            
            # Simular desde -T hasta 0
            for t in range(T):
                # Para cada sitio del lattice en orden aleatorio
                sites = [(i, j) for i in range(self.size) for j in range(self.size)]
                random.shuffle(sites)
                
                for i, j in sites:
                    random_val = random.random()
                    # Aplicar la misma transición a ambos estados
                    self.apply_transition(model_up, i, j, beta, random_val)
                    self.apply_transition(model_down, i, j, beta, random_val)
            
            # Verificar coalescencia
            if self.states_coalesced(model_up, model_down):
                return model_up.copy()
            
            # Doblar el tiempo si no hay coalescencia
            T *= 2
        
        # Si no se alcanza coalescencia, usar Metropolis-Hastings como fallback
        print(f"Advertencia: No se alcanzó coalescencia en tiempo {max_time} para beta={beta}")
        fallback_model = IsingModel(self.size, self.J, self.B)
        mh = MetropolisHastings(fallback_model)
        return mh.run(beta, 10000, 1000)

def run_experiments():
    """Ejecuta todos los experimentos especificados"""
    
    # Parámetros del experimento
    lattice_sizes = [10, 12, 15, 17, 20]
    beta_values = [i * 0.1 for i in range(11)]  # 0, 0.1, 0.2, ..., 1.0
    n_samples = 100
    mh_steps = 100000  # 10^5 iteraciones
    
    print("=== Experimentos del Modelo de Ising ===")
    print(f"Tamaños de lattice: {lattice_sizes}")
    print(f"Valores de β: {beta_values}")
    print(f"Número de muestras: {n_samples}")
    print(f"Pasos MH por muestra: {mh_steps}")
    print()
    
    print("Distribución de Boltzmann con temperatura inversa β:")
    print("π(σ) = exp(-β * H(σ)) / Z(β)")
    print("donde H(σ) = -J∑σᵢσⱼ - B∑σᵢ con J=1, B=0")
    print()
    
    results = {
        'metropolis_hastings': {},
        'propp_wilson': {},
        'parameters': {
            'lattice_sizes': lattice_sizes,
            'beta_values': beta_values,
            'n_samples': n_samples,
            'mh_steps': mh_steps,
            'J': 1.0,
            'B': 0.0
        }
    }
    
    for size in lattice_sizes:
        print(f"\n--- Lattice {size}x{size} ---")
        
        results['metropolis_hastings'][size] = {}
        results['propp_wilson'][size] = {}
        
        for beta in beta_values:
            print(f"β = {beta:.1f}")
            
            # Metropolis-Hastings
            mh_samples = []
            mh_start_time = time.time()
            
            for sample_idx in range(n_samples):
                if (sample_idx + 1) % 20 == 0:
                    print(f"  MH muestra {sample_idx + 1}/{n_samples}")
                
                model = IsingModel(size, J=1.0, B=0.0)
                mh = MetropolisHastings(model)
                final_config = mh.run(beta, mh_steps, burn_in=10000)
                
                mh_samples.append({
                    'lattice': final_config.lattice.copy(),
                    'magnetization': final_config.magnetization(),
                    'energy': final_config.total_energy()
                })
            
            mh_time = time.time() - mh_start_time
            
            # Propp-Wilson
            pw_samples = []
            pw_start_time = time.time()
            
            for sample_idx in range(n_samples):
                if (sample_idx + 1) % 20 == 0:
                    print(f"  PW muestra {sample_idx + 1}/{n_samples}")
                
                pw = ProppWilson(size, J=1.0, B=0.0)
                final_config = pw.sample(beta)
                
                pw_samples.append({
                    'lattice': final_config.lattice.copy(),
                    'magnetization': final_config.magnetization(),
                    'energy': final_config.total_energy()
                })
            
            pw_time = time.time() - pw_start_time
            
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
    
    # Guardar resultados
    with open('ising_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n=== Experimentos completados ===")
    print("Resultados guardados en 'ising_results.pkl'")
    
    return results

def analyze_results(results: Dict):
    """Análisis básico de los resultados"""
    print("\n=== Análisis de Resultados ===")
    
    for size in results['parameters']['lattice_sizes']:
        print(f"\nLattice {size}x{size}:")
        
        for beta in results['parameters']['beta_values']:
            mh_data = results['metropolis_hastings'][size][beta]
            pw_data = results['propp_wilson'][size][beta]
            
            # Magnetización promedio
            mh_mag = np.mean([s['magnetization'] for s in mh_data['samples']])
            pw_mag = np.mean([s['magnetization'] for s in pw_data['samples']])
            
            # Energía promedio  
            mh_energy = np.mean([s['energy'] for s in mh_data['samples']])
            pw_energy = np.mean([s['energy'] for s in pw_data['samples']])
            
            print(f"  β={beta:.1f}: MH_mag={mh_mag:.2f}, PW_mag={pw_mag:.2f}, "
                  f"MH_E={mh_energy:.2f}, PW_E={pw_energy:.2f}")

if __name__ == "__main__":
    # Ejecutar experimentos
    results = run_experiments()
    
    # Análisis básico
    analyze_results(results)