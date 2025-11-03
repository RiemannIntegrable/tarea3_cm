"""
Script de prueba para verificar la implementación del modelo de Ising
"""

import numpy as np
from ising_sampling import IsingModel, MetropolisHastings, ProppWilson

def test_ising_model():
    """Prueba básica del modelo de Ising"""
    print("=== Prueba del Modelo de Ising ===")
    
    # Crear modelo pequeño
    model = IsingModel(size=5, J=1.0, B=0.0)
    print(f"Lattice inicial 5x5:")
    print(model.lattice)
    print(f"Magnetización: {model.magnetization()}")
    print(f"Energía total: {model.total_energy()}")
    print()

def test_metropolis_hastings():
    """Prueba del algoritmo Metropolis-Hastings"""
    print("=== Prueba Metropolis-Hastings ===")
    
    model = IsingModel(size=5, J=1.0, B=0.0)
    mh = MetropolisHastings(model)
    
    print("Configuración inicial:")
    print(model.lattice)
    print(f"Energía inicial: {model.total_energy()}")
    
    # Ejecutar con diferentes temperaturas
    for beta in [0.1, 0.5, 1.0]:
        print(f"\nβ = {beta}")
        model_copy = model.copy()
        mh_copy = MetropolisHastings(model_copy)
        final_config = mh_copy.run(beta, steps=1000, burn_in=100)
        
        print(f"Magnetización final: {final_config.magnetization()}")
        print(f"Energía final: {final_config.total_energy()}")

def test_propp_wilson():
    """Prueba del algoritmo Propp-Wilson"""
    print("\n=== Prueba Propp-Wilson ===")
    
    size = 4  # Tamaño pequeño para prueba rápida
    pw = ProppWilson(size, J=1.0, B=0.0)
    
    for beta in [0.2, 0.8]:
        print(f"\nβ = {beta}, lattice {size}x{size}")
        sample = pw.sample(beta, max_time=100)
        print(f"Magnetización: {sample.magnetization()}")
        print(f"Energía: {sample.total_energy()}")
        print("Configuración:")
        print(sample.lattice)

def test_boltzmann_distribution():
    """Verificar la formulación con temperatura inversa"""
    print("\n=== Distribución de Boltzmann con β ===")
    print("π(σ) = exp(-β * H(σ)) / Z(β)")
    print("donde β = 1/T es la temperatura inversa")
    print("H(σ) = -J∑σᵢσⱼ - B∑σᵢ")
    print("Con J=1, B=0: H(σ) = -∑σᵢσⱼ (suma sobre vecinos)")
    print()
    
    # Ejemplo con configuración específica
    model = IsingModel(size=3, J=1.0, B=0.0)
    model.lattice = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, 1]])
    
    print("Configuración de ejemplo 3x3:")
    print(model.lattice)
    print(f"Energía H(σ) = {model.total_energy()}")
    
    print("\nProbabilidades relativas para diferentes β:")
    for beta in [0.1, 0.5, 1.0]:
        prob = np.exp(-beta * model.total_energy())
        print(f"β = {beta}: exp(-β*H) = {prob:.6f}")

if __name__ == "__main__":
    test_ising_model()
    test_metropolis_hastings() 
    test_propp_wilson()
    test_boltzmann_distribution()
    print("\n=== Pruebas completadas ===")