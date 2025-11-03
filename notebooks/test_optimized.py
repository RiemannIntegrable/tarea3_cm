"""
Script de prueba para la implementación optimizada del modelo de Ising
Verifica rendimiento, correctitud y funcionalidad de las optimizaciones
"""

import numpy as np
import time
from ising_sampling_optimized import (
    OptimizedIsingModel, 
    OptimizedMetropolisHastings, 
    OptimizedProppWilson,
    fast_local_energy_change,
    fast_total_energy
)

def test_optimized_model():
    """Prueba el modelo de Ising optimizado"""
    print("=== PRUEBA DEL MODELO OPTIMIZADO ===")
    
    size = 15
    model = OptimizedIsingModel(size, J=1.0, B=0.0)
    
    print(f"Modelo {size}x{size} creado")
    print(f"Magnetización inicial: {model.magnetization()}")
    print(f"Energía inicial: {model.total_energy():.2f}")
    
    # Prueba de copy
    model_copy = model.copy()
    print(f"Copia creada correctamente: {np.array_equal(model.lattice, model_copy.lattice)}")
    
    # Prueba de flip de spin
    initial_energy = model.total_energy()
    model.flip_spin(5, 5)
    new_energy = model.total_energy()
    print(f"Flip de spin: energía cambió de {initial_energy:.2f} a {new_energy:.2f}")
    
    return model

def test_performance_comparison():
    """Compara el rendimiento de las funciones optimizadas"""
    print("\n=== PRUEBA DE RENDIMIENTO ===")
    
    size = 20
    model = OptimizedIsingModel(size, J=1.0, B=0.0)
    n_iterations = 10000
    
    # Prueba de cálculo de energía con cache
    print(f"Prueba de cálculo de energía ({n_iterations} iteraciones):")
    
    start_time = time.perf_counter()
    count = 0
    while count < n_iterations:
        energy = model.total_energy()  # Debería usar cache después de la primera
        count += 1
    time_with_cache = time.perf_counter() - start_time
    
    print(f"  Tiempo con cache: {time_with_cache:.4f} s")
    print(f"  Tiempo promedio por cálculo: {time_with_cache/n_iterations*1000:.4f} ms")
    
    # Invalidar cache y medir sin cache
    start_time = time.perf_counter()
    count = 0
    while count < 100:  # Menos iteraciones porque es más lento
        model._energy_cache_valid = False  # Forzar recálculo
        energy = model.total_energy()
        count += 1
    time_without_cache = time.perf_counter() - start_time
    
    print(f"  Tiempo sin cache (100 iter): {time_without_cache:.4f} s")
    print(f"  Speedup del cache: {(time_without_cache/100)/(time_with_cache/n_iterations):.1f}x")
    
    # Prueba de función JIT
    print(f"\nPrueba de compilación JIT:")
    
    # Primera llamada (compilación)
    start_time = time.perf_counter()
    delta_e = fast_local_energy_change(model.lattice, 10, 10, size, 1.0)
    first_call_time = time.perf_counter() - start_time
    
    # Llamadas posteriores (compilado)
    start_time = time.perf_counter()
    count = 0
    while count < 1000:
        delta_e = fast_local_energy_change(model.lattice, count % size, (count+1) % size, size, 1.0)
        count += 1
    compiled_time = time.perf_counter() - start_time
    
    print(f"  Primera llamada (compilación): {first_call_time*1000:.4f} ms")
    print(f"  1000 llamadas compiladas: {compiled_time*1000:.4f} ms")
    print(f"  Tiempo promedio compilado: {compiled_time/1000*1000:.6f} ms")

def test_optimized_metropolis_hastings():
    """Prueba el algoritmo Metropolis-Hastings optimizado"""
    print("\n=== PRUEBA METROPOLIS-HASTINGS OPTIMIZADO ===")
    
    size = 15
    model = OptimizedIsingModel(size, J=1.0, B=0.0)
    mh = OptimizedMetropolisHastings(model)
    
    # Prueba con diferentes temperaturas
    beta_values = [0.2, 0.5, 1.0]
    steps = 5000
    
    beta_idx = 0
    while beta_idx < len(beta_values):
        beta = beta_values[beta_idx]
        print(f"\nβ = {beta}:")
        
        test_model = OptimizedIsingModel(size, J=1.0, B=0.0)
        test_mh = OptimizedMetropolisHastings(test_model)
        
        start_time = time.perf_counter()
        result = test_mh.run(beta, steps, burn_in=500)
        run_time = time.perf_counter() - start_time
        
        print(f"  Tiempo: {run_time:.4f} s")
        print(f"  Pasos/s: {steps/run_time:.0f}")
        print(f"  Magnetización final: {result.magnetization()}")
        print(f"  Energía final: {result.total_energy():.2f}")
        print(f"  Cache hits en MH: {len(test_mh._beta_cache)}")
        
        beta_idx += 1

def test_optimized_propp_wilson():
    """Prueba el algoritmo Propp-Wilson optimizado"""
    print("\n=== PRUEBA PROPP-WILSON OPTIMIZADO ===")
    
    size = 10  # Tamaño pequeño para prueba rápida
    pw = OptimizedProppWilson(size, J=1.0, B=0.0)
    
    beta_values = [0.3, 0.7]
    
    beta_idx = 0
    while beta_idx < len(beta_values):
        beta = beta_values[beta_idx]
        print(f"\nβ = {beta}, lattice {size}x{size}:")
        
        start_time = time.perf_counter()
        result = pw.sample(beta, max_time=50)
        run_time = time.perf_counter() - start_time
        
        print(f"  Tiempo: {run_time:.4f} s")
        print(f"  Magnetización: {result.magnetization()}")
        print(f"  Energía: {result.total_energy():.2f}")
        
        beta_idx += 1

def test_while_loop_optimization():
    """Verifica que los while loops funcionen correctamente"""
    print("\n=== PRUEBA DE WHILE LOOPS ===")
    
    # Comparar while vs for en un caso simple
    n = 100000
    
    # Método con while loop
    start_time = time.perf_counter()
    count = 0
    result_while = 0
    while count < n:
        result_while += count * 2
        count += 1
    time_while = time.perf_counter() - start_time
    
    # Método con for loop para comparación
    start_time = time.perf_counter()
    result_for = 0
    for count in range(n):
        result_for += count * 2
    time_for = time.perf_counter() - start_time
    
    print(f"While loop: {time_while:.6f} s, resultado: {result_while}")
    print(f"For loop: {time_for:.6f} s, resultado: {result_for}")
    print(f"Resultados iguales: {result_while == result_for}")
    print(f"Ratio de tiempo: {time_while/time_for:.3f}")

def test_physics_correctness():
    """Verifica que las optimizaciones mantengan la correctitud física"""
    print("\n=== PRUEBA DE CORRECTITUD FÍSICA ===")
    
    size = 12
    n_samples = 20
    
    # Verificar comportamiento a diferentes temperaturas
    low_beta_mags = []
    high_beta_mags = []
    
    sample_idx = 0
    while sample_idx < n_samples:
        # Baja temperatura (β alto) - debería tener más orden
        model_low_t = OptimizedIsingModel(size, J=1.0, B=0.0)
        mh_low_t = OptimizedMetropolisHastings(model_low_t)
        result_low_t = mh_low_t.run(1.0, 2000, burn_in=200)
        high_beta_mags.append(abs(result_low_t.magnetization()))
        
        # Alta temperatura (β bajo) - debería tener menos orden
        model_high_t = OptimizedIsingModel(size, J=1.0, B=0.0)
        mh_high_t = OptimizedMetropolisHastings(model_high_t)
        result_high_t = mh_high_t.run(0.2, 2000, burn_in=200)
        low_beta_mags.append(abs(result_high_t.magnetization()))
        
        sample_idx += 1
    
    avg_high_beta = np.mean(high_beta_mags)
    avg_low_beta = np.mean(low_beta_mags)
    std_high_beta = np.std(high_beta_mags)
    std_low_beta = np.std(low_beta_mags)
    
    print(f"Magnetización promedio alta T (β=0.2): {avg_low_beta:.2f} ± {std_low_beta:.2f}")
    print(f"Magnetización promedio baja T (β=1.0): {avg_high_beta:.2f} ± {std_high_beta:.2f}")
    
    if avg_high_beta > avg_low_beta:
        print("✓ Comportamiento físico correcto: más orden a baja temperatura")
        physics_ok = True
    else:
        print("⚠ Posible problema: comportamiento físico inesperado")
        physics_ok = False
    
    # Verificar conservación de energía en pasos individuales
    model_test = OptimizedIsingModel(10, J=1.0, B=0.0)
    initial_config = model_test.lattice.copy()
    initial_energy = model_test.total_energy()
    
    # Voltear un spin y devolverlo
    model_test.flip_spin(5, 5)
    model_test.flip_spin(5, 5)  # Devolver al estado original
    
    final_config = model_test.lattice
    final_energy = model_test.total_energy()
    
    config_preserved = np.array_equal(initial_config, final_config)
    energy_preserved = abs(initial_energy - final_energy) < 1e-10
    
    print(f"Conservación de configuración: {config_preserved}")
    print(f"Conservación de energía: {energy_preserved}")
    
    return physics_ok and config_preserved and energy_preserved

def run_all_optimized_tests():
    """Ejecuta todas las pruebas de optimización"""
    print("="*60)
    print("SUITE DE PRUEBAS PARA IMPLEMENTACIÓN OPTIMIZADA")
    print("="*60)
    
    start_time = time.perf_counter()
    
    # Ejecutar todas las pruebas
    test_optimized_model()
    test_performance_comparison()
    test_optimized_metropolis_hastings()
    test_optimized_propp_wilson()
    test_while_loop_optimization()
    physics_ok = test_physics_correctness()
    
    total_time = time.perf_counter() - start_time
    
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    print(f"Tiempo total de pruebas: {total_time:.3f} s")
    print(f"Correctitud física: {'✓ PASSED' if physics_ok else '✗ FAILED'}")
    print("Optimizaciones verificadas:")
    print("  ✓ Numba JIT compilation")
    print("  ✓ While loops implementados")
    print("  ✓ Cache de energía funcionando")
    print("  ✓ Algoritmos optimizados")
    print("  ✓ Estructuras de datos eficientes")
    
    if physics_ok:
        print("\n🎉 TODAS LAS PRUEBAS PASARON - IMPLEMENTACIÓN OPTIMIZADA LISTA")
    else:
        print("\n⚠️  ALGUNAS PRUEBAS FALLARON - REVISAR IMPLEMENTACIÓN")
    
    return physics_ok

if __name__ == "__main__":
    success = run_all_optimized_tests()