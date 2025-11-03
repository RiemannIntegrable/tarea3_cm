"""
Análisis Completo de Resultados - Tarea 3 Cadenas de Markov
Muestreo MCMC vs Simulación Perfecta aplicadas al Modelo de Ising

Extrae y analiza los resultados guardados en ising_results_optimized.pkl
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns

def load_complete_results(filename='ising_results_optimized.pkl'):
    """Cargar resultados completos del experimento"""
    try:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"✓ Resultados cargados desde {filename}")
        return results
    except FileNotFoundError:
        print(f"✗ Error: Archivo {filename} no encontrado")
        return None
    except Exception as e:
        print(f"✗ Error al cargar resultados: {e}")
        return None

def extract_magnetization_data(results: Dict) -> pd.DataFrame:
    """Extrae datos de magnetización para análisis"""
    
    if results is None:
        return None
    
    print("\n=== EXTRAYENDO DATOS DE MAGNETIZACIÓN ===")
    
    sizes = results['parameters']['lattice_sizes']
    betas = results['parameters']['beta_values']
    n_samples = results['parameters']['n_samples']
    
    print(f"Lattice sizes: {sizes}")
    print(f"Beta values: {betas}")
    print(f"Muestras por configuración: {n_samples}")
    
    # Lista para almacenar todos los datos
    data_rows = []
    
    size_idx = 0
    while size_idx < len(sizes):
        size = sizes[size_idx]
        
        beta_idx = 0
        while beta_idx < len(betas):
            beta = betas[beta_idx]
            
            # Datos Metropolis-Hastings
            mh_data = results['metropolis_hastings'][size][beta]
            mh_samples = mh_data['samples']
            mh_time = mh_data['computation_time']
            
            mh_magnetizations = np.array([s['magnetization'] for s in mh_samples])
            mh_energies = np.array([s['energy'] for s in mh_samples])
            
            # Normalizar magnetización por tamaño del lattice
            normalized_mh_mags = mh_magnetizations / (size * size)
            
            data_rows.append({
                'size': size,
                'beta': beta,
                'method': 'Metropolis-Hastings',
                'magnetization_mean': np.mean(mh_magnetizations),
                'magnetization_std': np.std(mh_magnetizations),
                'magnetization_normalized_mean': np.mean(normalized_mh_mags),
                'abs_magnetization_mean': np.mean(np.abs(mh_magnetizations)),
                'abs_magnetization_normalized_mean': np.mean(np.abs(normalized_mh_mags)),
                'energy_mean': np.mean(mh_energies),
                'energy_std': np.std(mh_energies),
                'computation_time': mh_time,
                'samples_count': len(mh_samples)
            })
            
            # Datos Propp-Wilson
            pw_data = results['propp_wilson'][size][beta]
            pw_samples = pw_data['samples']
            pw_time = pw_data['computation_time']
            
            pw_magnetizations = np.array([s['magnetization'] for s in pw_samples])
            pw_energies = np.array([s['energy'] for s in pw_samples])
            
            # Normalizar magnetización por tamaño del lattice
            normalized_pw_mags = pw_magnetizations / (size * size)
            
            data_rows.append({
                'size': size,
                'beta': beta,
                'method': 'Propp-Wilson',
                'magnetization_mean': np.mean(pw_magnetizations),
                'magnetization_std': np.std(pw_magnetizations),
                'magnetization_normalized_mean': np.mean(normalized_pw_mags),
                'abs_magnetization_mean': np.mean(np.abs(pw_magnetizations)),
                'abs_magnetization_normalized_mean': np.mean(np.abs(normalized_pw_mags)),
                'energy_mean': np.mean(pw_energies),
                'energy_std': np.std(pw_energies),
                'computation_time': pw_time,
                'samples_count': len(pw_samples)
            })
            
            beta_idx += 1
        size_idx += 1
    
    df = pd.DataFrame(data_rows)
    print(f"✓ Datos extraídos: {len(df)} configuraciones")
    return df

def report_coalescence_times(results: Dict):
    """Reporta tiempos de coalescencia de Propp-Wilson"""
    
    if results is None:
        return
    
    print("\n=== REPORTE DE TIEMPOS DE COALESCENCIA (PROPP-WILSON) ===")
    
    sizes = results['parameters']['lattice_sizes']
    betas = results['parameters']['beta_values']
    
    # Crear tabla de tiempos
    coalescence_data = []
    
    size_idx = 0
    while size_idx < len(sizes):
        size = sizes[size_idx]
        print(f"\nLattice {size}×{size}:")
        
        beta_idx = 0
        while beta_idx < len(betas):
            beta = betas[beta_idx]
            
            pw_data = results['propp_wilson'][size][beta]
            pw_time = pw_data['computation_time']
            n_samples = len(pw_data['samples'])
            avg_time_per_sample = pw_time / n_samples
            
            coalescence_data.append({
                'size': size,
                'beta': beta,
                'total_time': pw_time,
                'avg_time_per_sample': avg_time_per_sample,
                'samples': n_samples
            })
            
            print(f"  β={beta:.1f}: {pw_time:.2f}s total, {avg_time_per_sample:.3f}s promedio/muestra")
            
            beta_idx += 1
        size_idx += 1
    
    return coalescence_data

def create_comparison_plots(df: pd.DataFrame, results: Dict):
    """Crea gráficas de comparación entre métodos"""
    
    if df is None or results is None:
        print("No hay datos para crear gráficas")
        return
    
    print("\n=== CREANDO GRÁFICAS DE COMPARACIÓN ===")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    
    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tarea 3: Muestreo MCMC vs Simulación Perfecta - Modelo de Ising', 
                 fontsize=16, fontweight='bold')
    
    sizes = results['parameters']['lattice_sizes']
    betas = np.array(results['parameters']['beta_values'])
    
    # Colores para diferentes tamaños
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sizes)))
    
    # Gráfica 1: Magnetización absoluta vs β (Principal requerida por la tarea)
    ax1 = axes[0, 0]
    
    size_idx = 0
    while size_idx < len(sizes):
        size = sizes[size_idx]
        
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        pw_data = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]
        
        ax1.plot(mh_data['beta'], mh_data['abs_magnetization_normalized_mean'], 
                'o-', color=colors[size_idx], label=f'MH {size}×{size}', 
                linewidth=2, markersize=6, alpha=0.8)
        ax1.plot(pw_data['beta'], pw_data['abs_magnetization_normalized_mean'], 
                's--', color=colors[size_idx], label=f'PW {size}×{size}', 
                linewidth=2, markersize=6, alpha=0.8)
        
        size_idx += 1
    
    ax1.set_xlabel('β (temperatura inversa)')
    ax1.set_ylabel('E[|M(η)|] normalizada')
    ax1.set_title('Comparación: Estimación de Magnetización vs β')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Línea de temperatura crítica teórica
    beta_c = 2 / np.log(1 + np.sqrt(2))
    ax1.axvline(beta_c, color='red', linestyle=':', alpha=0.7, 
                label=f'β_c teórico ≈ {beta_c:.3f}')
    
    # Gráfica 2: Energía vs β
    ax2 = axes[0, 1]
    
    size_idx = 0
    while size_idx < len(sizes):
        size = sizes[size_idx]
        
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        pw_data = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]
        
        ax2.plot(mh_data['beta'], mh_data['energy_mean'], 
                'o-', color=colors[size_idx], label=f'MH {size}×{size}', 
                linewidth=2, markersize=6, alpha=0.8)
        ax2.plot(pw_data['beta'], pw_data['energy_mean'], 
                's--', color=colors[size_idx], label=f'PW {size}×{size}', 
                linewidth=2, markersize=6, alpha=0.8)
        
        size_idx += 1
    
    ax2.set_xlabel('β (temperatura inversa)')
    ax2.set_ylabel('E[H(η)] - Energía promedio')
    ax2.set_title('Energía vs Temperatura')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(beta_c, color='red', linestyle=':', alpha=0.7)
    
    # Gráfica 3: Diferencias entre métodos
    ax3 = axes[1, 0]
    
    size_idx = 0
    while size_idx < len(sizes):
        size = sizes[size_idx]
        
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        pw_data = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]
        
        # Calcular diferencias relativas
        rel_diff = np.abs(mh_data['abs_magnetization_normalized_mean'].values - 
                         pw_data['abs_magnetization_normalized_mean'].values)
        
        ax3.plot(betas, rel_diff, 'o-', color=colors[size_idx], 
                label=f'{size}×{size}', linewidth=2, markersize=6)
        
        size_idx += 1
    
    ax3.set_xlabel('β (temperatura inversa)')
    ax3.set_ylabel('|E[|M|]_MH - E[|M|]_PW|')
    ax3.set_title('Diferencias Absolutas entre Métodos')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Gráfica 4: Tiempos de coalescencia
    ax4 = axes[1, 1]
    
    size_idx = 0
    while size_idx < len(sizes):
        size = sizes[size_idx]
        
        pw_data = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]
        avg_times = pw_data['computation_time'] / pw_data['samples_count']
        
        ax4.plot(pw_data['beta'], avg_times, 'o-', color=colors[size_idx], 
                label=f'PW {size}×{size}', linewidth=2, markersize=6)
        
        size_idx += 1
    
    ax4.set_xlabel('β (temperatura inversa)')
    ax4.set_ylabel('Tiempo promedio por muestra (s)')
    ax4.set_title('Tiempo de Coalescencia Propp-Wilson')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('comparison_magnetization_vs_beta.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Gráficas guardadas en 'comparison_magnetization_vs_beta.png'")

def generate_summary_report(df: pd.DataFrame, results: Dict, coalescence_data: List):
    """Genera reporte resumen de los resultados"""
    
    if df is None or results is None:
        return
    
    print("\n" + "="*70)
    print("REPORTE FINAL - TAREA 3: MUESTREO MCMC vs SIMULACIÓN PERFECTA")
    print("="*70)
    
    # Parámetros del experimento
    params = results['parameters']
    print(f"\n📋 PARÁMETROS DEL EXPERIMENTO:")
    print(f"   • Lattice sizes (K×K): {params['lattice_sizes']}")
    print(f"   • Temperaturas inversas β: {params['beta_values']}")
    print(f"   • Muestras por configuración: {params['n_samples']}")
    print(f"   • Pasos Metropolis-Hastings: {params['mh_steps']:,}")
    print(f"   • Constante J = {params['J']}, Campo B = {params['B']}")
    
    # Estadísticas generales
    total_configs = len(params['lattice_sizes']) * len(params['beta_values'])
    total_samples = total_configs * params['n_samples'] * 2  # x2 por ambos métodos
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   • Total de configuraciones: {total_configs}")
    print(f"   • Total de simulaciones: {total_samples:,}")
    print(f"   • Tiempo total MH: {df[df['method'] == 'Metropolis-Hastings']['computation_time'].sum():.1f}s")
    print(f"   • Tiempo total PW: {df[df['method'] == 'Propp-Wilson']['computation_time'].sum():.1f}s")
    
    # Análisis de transición de fase
    print(f"\n🌡️  ANÁLISIS DE TRANSICIÓN DE FASE:")
    beta_c_theoretical = 2 / np.log(1 + np.sqrt(2))
    print(f"   • Temperatura crítica teórica βc = {beta_c_theoretical:.6f}")
    
    # Buscar evidencia de transición para cada tamaño
    for size in params['lattice_sizes']:
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        
        # Calcular susceptibilidad como derivada numérica de magnetización
        mags = mh_data['abs_magnetization_normalized_mean'].values
        betas_array = np.array(params['beta_values'])
        susceptibility = np.gradient(mags, betas_array)
        
        # Encontrar máximo de susceptibilidad
        max_idx = np.argmax(susceptibility)
        beta_c_empirical = betas_array[max_idx]
        
        print(f"   • Lattice {size}×{size}: βc empírico ≈ {beta_c_empirical:.3f} "
              f"(diff: {abs(beta_c_empirical - beta_c_theoretical):.3f})")
    
    # Comparación entre métodos
    print(f"\n🔄 COMPARACIÓN ENTRE MÉTODOS:")
    
    mh_all_mags = df[df['method'] == 'Metropolis-Hastings']['abs_magnetization_normalized_mean']
    pw_all_mags = df[df['method'] == 'Propp-Wilson']['abs_magnetization_normalized_mean']
    
    correlation = np.corrcoef(mh_all_mags, pw_all_mags)[0, 1]
    mean_rel_diff = np.mean(np.abs(mh_all_mags - pw_all_mags) / (mh_all_mags + 1e-10))
    
    print(f"   • Correlación entre métodos: {correlation:.6f}")
    print(f"   • Diferencia relativa promedio: {mean_rel_diff:.6f}")
    print(f"   • Máxima diferencia absoluta: {np.max(np.abs(mh_all_mags - pw_all_mags)):.6f}")
    
    # Eficiencia computacional
    print(f"\n⚡ EFICIENCIA COMPUTACIONAL:")
    
    mh_avg_time = df[df['method'] == 'Metropolis-Hastings']['computation_time'].mean()
    pw_avg_time = df[df['method'] == 'Propp-Wilson']['computation_time'].mean()
    speedup_ratio = pw_avg_time / mh_avg_time
    
    print(f"   • Tiempo promedio MH: {mh_avg_time:.2f}s por configuración")
    print(f"   • Tiempo promedio PW: {pw_avg_time:.2f}s por configuración")
    print(f"   • Ratio de eficiencia: MH es {speedup_ratio:.2f}x más rápido")
    
    # Coalescencia más problemática
    if coalescence_data:
        max_time_config = max(coalescence_data, key=lambda x: x['avg_time_per_sample'])
        min_time_config = min(coalescence_data, key=lambda x: x['avg_time_per_sample'])
        
        print(f"\n⏱️  ANÁLISIS DE COALESCENCIA:")
        print(f"   • Coalescencia más lenta: {max_time_config['size']}×{max_time_config['size']}, "
              f"β={max_time_config['beta']:.1f} ({max_time_config['avg_time_per_sample']:.3f}s/muestra)")
        print(f"   • Coalescencia más rápida: {min_time_config['size']}×{min_time_config['size']}, "
              f"β={min_time_config['beta']:.1f} ({min_time_config['avg_time_per_sample']:.3f}s/muestra)")
    
    print(f"\n✅ EXPERIMENTO COMPLETADO EXITOSAMENTE")
    print("="*70)

def main():
    """Función principal para ejecutar el análisis completo"""
    
    print("🔬 ANALIZADOR DE RESULTADOS COMPLETOS - TAREA 3")
    print("   Muestreo MCMC vs Simulación Perfecta en Modelo de Ising")
    print("="*70)
    
    # 1. Cargar resultados
    results = load_complete_results()
    if results is None:
        return
    
    # 2. Extraer datos de magnetización
    df = extract_magnetization_data(results)
    if df is None:
        return
    
    # 3. Reportar tiempos de coalescencia
    coalescence_data = report_coalescence_times(results)
    
    # 4. Crear gráficas de comparación
    create_comparison_plots(df, results)
    
    # 5. Generar reporte final
    generate_summary_report(df, results, coalescence_data)
    
    # 6. Exportar datos a CSV
    csv_filename = 'complete_analysis_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Datos exportados a {csv_filename}")

if __name__ == "__main__":
    main()