"""
Generador de Imágenes para Reporte - Tarea 3
Genera todas las gráficas requeridas y las guarda en report/images/
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import os
from pathlib import Path

def ensure_images_directory():
    """Crear directorio de imágenes si no existe"""
    images_dir = Path("../report/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir

def load_results():
    """Cargar resultados del experimento"""
    try:
        with open('ising_results_optimized.pkl', 'rb') as f:
            results = pickle.load(f)
        print("✓ Resultados cargados exitosamente")
        return results
    except FileNotFoundError:
        print("✗ Error: Archivo de resultados no encontrado")
        return None

def extract_data(results):
    """Extraer datos para análisis"""
    if results is None:
        return None
    
    sizes = results['parameters']['lattice_sizes']
    betas = results['parameters']['beta_values']
    
    data_rows = []
    
    for size in sizes:
        for beta in betas:
            # Metropolis-Hastings
            mh_samples = results['metropolis_hastings'][size][beta]['samples']
            mh_time = results['metropolis_hastings'][size][beta]['computation_time']
            mh_mags = np.array([s['magnetization'] for s in mh_samples])
            mh_energies = np.array([s['energy'] for s in mh_samples])
            mh_normalized = mh_mags / (size * size)
            
            data_rows.append({
                'size': size, 'beta': beta, 'method': 'Metropolis-Hastings',
                'E_abs_M_normalized': np.mean(np.abs(mh_normalized)),
                'E_energy': np.mean(mh_energies),
                'std_M': np.std(mh_mags),
                'computation_time': mh_time,
                'avg_time_per_sample': mh_time / len(mh_samples)
            })
            
            # Propp-Wilson
            pw_samples = results['propp_wilson'][size][beta]['samples']
            pw_time = results['propp_wilson'][size][beta]['computation_time']
            pw_mags = np.array([s['magnetization'] for s in pw_samples])
            pw_energies = np.array([s['energy'] for s in pw_samples])
            pw_normalized = pw_mags / (size * size)
            
            data_rows.append({
                'size': size, 'beta': beta, 'method': 'Propp-Wilson',
                'E_abs_M_normalized': np.mean(np.abs(pw_normalized)),
                'E_energy': np.mean(pw_energies),
                'std_M': np.std(pw_mags),
                'computation_time': pw_time,
                'avg_time_per_sample': pw_time / len(pw_samples)
            })
    
    return pd.DataFrame(data_rows)

def generate_main_comparison_plot(df, results, images_dir):
    """Genera la gráfica principal de comparación (REQUERIDA)"""
    
    print("🎨 Generando gráfica principal de comparación...")
    
    sizes = results['parameters']['lattice_sizes']
    betas = np.array(results['parameters']['beta_values'])
    
    # Configurar figura
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sizes)))
    
    # Plotear datos para cada tamaño
    for i, size in enumerate(sizes):
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        pw_data = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]
        
        plt.plot(mh_data['beta'], mh_data['E_abs_M_normalized'], 
                'o-', color=colors[i], label=f'MH {size}×{size}', 
                linewidth=3, markersize=8, alpha=0.9)
        
        plt.plot(pw_data['beta'], pw_data['E_abs_M_normalized'], 
                's--', color=colors[i], label=f'PW {size}×{size}', 
                linewidth=3, markersize=8, alpha=0.9)
    
    # Temperatura crítica teórica
    beta_c = 2 / np.log(1 + np.sqrt(2))
    plt.axvline(beta_c, color='red', linestyle=':', linewidth=2.5, alpha=0.8,
                label=f'βc teórico ≈ {beta_c:.3f}')
    
    # Configuración de la gráfica
    plt.xlabel('β (temperatura inversa)', fontsize=14, fontweight='bold')
    plt.ylabel('E[|M(η)|] normalizada', fontsize=14, fontweight='bold')
    plt.title('Tarea 3: Comparación de Estimaciones de Magnetización\n' +
              'Muestreo MCMC vs Simulación Perfecta - Modelo de Ising',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Añadir texto explicativo
    plt.text(0.02, 0.98, 
             'Líneas sólidas: Metropolis-Hastings (MCMC)\n' +
             'Líneas punteadas: Propp-Wilson (Perfect Sampling)',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Guardar imagen
    plt.tight_layout()
    output_path = images_dir / "magnetization_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Guardada: {output_path}")
    return output_path

def generate_energy_analysis_plot(df, results, images_dir):
    """Genera gráfica de análisis de energía"""
    
    print("🎨 Generando gráfica de análisis de energía...")
    
    sizes = results['parameters']['lattice_sizes']
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sizes)))
    
    plt.figure(figsize=(12, 8))
    
    for i, size in enumerate(sizes):
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        pw_data = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]
        
        plt.plot(mh_data['beta'], mh_data['E_energy'], 
                'o-', color=colors[i], label=f'MH {size}×{size}', 
                linewidth=3, markersize=7, alpha=0.9)
        
        plt.plot(pw_data['beta'], pw_data['E_energy'], 
                's--', color=colors[i], label=f'PW {size}×{size}', 
                linewidth=3, markersize=7, alpha=0.9)
    
    # Temperatura crítica
    beta_c = 2 / np.log(1 + np.sqrt(2))
    plt.axvline(beta_c, color='red', linestyle=':', linewidth=2, alpha=0.8)
    
    plt.xlabel('β (temperatura inversa)', fontsize=14, fontweight='bold')
    plt.ylabel('E[H(η)] - Energía promedio', fontsize=14, fontweight='bold')
    plt.title('Análisis de Energía vs Temperatura - Modelo de Ising',
              fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    output_path = images_dir / "energy_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Guardada: {output_path}")
    return output_path

def generate_coalescence_times_plot(df, results, images_dir):
    """Genera gráfica de tiempos de coalescencia"""
    
    print("🎨 Generando gráfica de tiempos de coalescencia...")
    
    sizes = results['parameters']['lattice_sizes']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sizes)))
    
    plt.figure(figsize=(12, 8))
    
    for i, size in enumerate(sizes):
        pw_data = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]
        
        plt.plot(pw_data['beta'], pw_data['avg_time_per_sample'], 
                'o-', color=colors[i], label=f'PW {size}×{size}', 
                linewidth=3, markersize=8, alpha=0.9)
    
    plt.xlabel('β (temperatura inversa)', fontsize=14, fontweight='bold')
    plt.ylabel('Tiempo promedio por muestra (s)', fontsize=14, fontweight='bold')
    plt.title('Tiempos de Coalescencia - Propp-Wilson Perfect Sampling',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.yscale('log')
    
    plt.tight_layout()
    output_path = images_dir / "coalescence_times.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Guardada: {output_path}")
    return output_path

def generate_method_differences_plot(df, results, images_dir):
    """Genera gráfica de diferencias entre métodos"""
    
    print("🎨 Generando gráfica de diferencias entre métodos...")
    
    sizes = results['parameters']['lattice_sizes']
    betas = np.array(results['parameters']['beta_values'])
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(sizes)))
    
    plt.figure(figsize=(12, 8))
    
    for i, size in enumerate(sizes):
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        pw_data = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]
        
        # Calcular diferencias absolutas
        diff = np.abs(mh_data['E_abs_M_normalized'].values - pw_data['E_abs_M_normalized'].values)
        
        plt.plot(betas, diff, 'o-', color=colors[i], 
                label=f'{size}×{size}', linewidth=3, markersize=8)
    
    plt.xlabel('β (temperatura inversa)', fontsize=14, fontweight='bold')
    plt.ylabel('|E[|M|]_MH - E[|M|]_PW|', fontsize=14, fontweight='bold')
    plt.title('Diferencias Absolutas entre Métodos\nMetropolis-Hastings vs Propp-Wilson',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.yscale('log')
    
    plt.tight_layout()
    output_path = images_dir / "method_differences.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Guardada: {output_path}")
    return output_path

def generate_phase_transition_plot(df, results, images_dir):
    """Genera gráfica destacando la transición de fase"""
    
    print("🎨 Generando gráfica de transición de fase...")
    
    sizes = results['parameters']['lattice_sizes']
    betas = np.array(results['parameters']['beta_values'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sizes)))
    
    # Panel 1: Magnetización con zoom en transición
    for i, size in enumerate(sizes):
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        
        ax1.plot(mh_data['beta'], mh_data['E_abs_M_normalized'], 
                'o-', color=colors[i], label=f'{size}×{size}', 
                linewidth=3, markersize=8, alpha=0.9)
    
    # Temperatura crítica
    beta_c = 2 / np.log(1 + np.sqrt(2))
    ax1.axvline(beta_c, color='red', linestyle='--', linewidth=3, alpha=0.8,
                label=f'βc = {beta_c:.3f}')
    ax1.axvspan(beta_c-0.1, beta_c+0.1, alpha=0.2, color='red', label='Zona crítica')
    
    ax1.set_xlabel('β (temperatura inversa)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('E[|M(η)|] normalizada', fontsize=12, fontweight='bold')
    ax1.set_title('Transición de Fase - Magnetización', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    
    # Panel 2: Susceptibilidad magnética (derivada)
    for i, size in enumerate(sizes):
        mh_data = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]
        mags = mh_data['E_abs_M_normalized'].values
        
        # Calcular susceptibilidad como derivada numérica
        susceptibility = np.gradient(mags, betas)
        
        ax2.plot(betas, susceptibility, 'o-', color=colors[i], 
                label=f'{size}×{size}', linewidth=3, markersize=6)
    
    ax2.axvline(beta_c, color='red', linestyle='--', linewidth=3, alpha=0.8)
    ax2.set_xlabel('β (temperatura inversa)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('χ ~ dM/dβ (susceptibilidad)', fontsize=12, fontweight='bold')
    ax2.set_title('Susceptibilidad Magnética', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    
    fig.suptitle('Análisis de Transición de Fase - Modelo de Ising 2D', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = images_dir / "phase_transition.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Guardada: {output_path}")
    return output_path

def generate_computational_efficiency_plot(df, results, images_dir):
    """Genera gráfica de eficiencia computacional"""
    
    print("🎨 Generando gráfica de eficiencia computacional...")
    
    sizes = results['parameters']['lattice_sizes']
    
    # Calcular estadísticas por tamaño
    efficiency_data = []
    for size in sizes:
        mh_times = df[(df['size'] == size) & (df['method'] == 'Metropolis-Hastings')]['avg_time_per_sample']
        pw_times = df[(df['size'] == size) & (df['method'] == 'Propp-Wilson')]['avg_time_per_sample']
        
        efficiency_data.append({
            'size': size,
            'area': size*size,
            'mh_avg': mh_times.mean(),
            'pw_avg': pw_times.mean(),
            'speedup': pw_times.mean() / mh_times.mean()
        })
    
    eff_df = pd.DataFrame(efficiency_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel 1: Tiempos absolutos
    x_pos = np.arange(len(sizes))
    width = 0.35
    
    ax1.bar(x_pos - width/2, eff_df['mh_avg'], width, 
            label='Metropolis-Hastings', color='steelblue', alpha=0.8)
    ax1.bar(x_pos + width/2, eff_df['pw_avg'], width, 
            label='Propp-Wilson', color='orange', alpha=0.8)
    
    ax1.set_xlabel('Tamaño de Lattice', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Tiempo promedio por muestra (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Tiempos de Ejecución por Método', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{s}×{s}' for s in sizes])
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.4)
    
    # Panel 2: Escalabilidad
    ax2.loglog(eff_df['area'], eff_df['mh_avg'], 'o-', 
              color='steelblue', linewidth=3, markersize=8, label='Metropolis-Hastings')
    ax2.loglog(eff_df['area'], eff_df['pw_avg'], 's-', 
              color='orange', linewidth=3, markersize=8, label='Propp-Wilson')
    
    ax2.set_xlabel('Área del lattice (N²)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tiempo promedio por muestra (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Escalabilidad Computacional', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    
    fig.suptitle('Análisis de Eficiencia Computacional', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = images_dir / "computational_efficiency.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Guardada: {output_path}")
    return output_path

def main():
    """Función principal para generar todas las imágenes"""
    
    print("🖼️  GENERADOR DE IMÁGENES PARA REPORTE - TAREA 3")
    print("="*60)
    
    # 1. Preparar directorios
    images_dir = ensure_images_directory()
    print(f"📁 Directorio de imágenes: {images_dir}")
    
    # 2. Cargar datos
    results = load_results()
    if results is None:
        return False
    
    # 3. Extraer datos para análisis
    df = extract_data(results)
    if df is None:
        return False
    
    print(f"📊 Datos extraídos: {len(df)} configuraciones")
    
    # 4. Generar todas las gráficas
    generated_images = []
    
    # Gráfica principal (REQUERIDA por la tarea)
    img1 = generate_main_comparison_plot(df, results, images_dir)
    generated_images.append(img1)
    
    # Gráficas de análisis adicional
    img2 = generate_energy_analysis_plot(df, results, images_dir)
    generated_images.append(img2)
    
    img3 = generate_coalescence_times_plot(df, results, images_dir)
    generated_images.append(img3)
    
    img4 = generate_method_differences_plot(df, results, images_dir)
    generated_images.append(img4)
    
    img5 = generate_phase_transition_plot(df, results, images_dir)
    generated_images.append(img5)
    
    img6 = generate_computational_efficiency_plot(df, results, images_dir)
    generated_images.append(img6)
    
    # 5. Resumen final
    print("\n" + "="*60)
    print("✅ GENERACIÓN DE IMÁGENES COMPLETADA")
    print("="*60)
    print(f"\n📁 Ubicación: {images_dir}/")
    print("\n🖼️  Imágenes generadas:")
    for i, img_path in enumerate(generated_images, 1):
        print(f"   {i}. {img_path.name}")
    
    print(f"\n🎯 IMAGEN PRINCIPAL PARA LA TAREA:")
    print(f"   • magnetization_comparison.png - Gráfica requerida de comparación")
    
    print(f"\n📊 Total de imágenes generadas: {len(generated_images)}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ¡Todas las imágenes están listas para el reporte!")
    else:
        print("\n❌ Error durante la generación de imágenes")