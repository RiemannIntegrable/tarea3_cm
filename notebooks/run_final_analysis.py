#!/usr/bin/env python3
"""
Script Ejecutable - Análisis Final Tarea 3
Genera automáticamente todos los resultados requeridos por la tarea

USAGE: python run_final_analysis.py
"""

import sys
import os

def main():
    """Ejecuta el análisis completo de los resultados"""
    
    print("🔬 ANÁLISIS FINAL - TAREA 3")
    print("   Muestreo MCMC vs Simulación Perfecta - Modelo de Ising")
    print("="*60)
    
    # Verificar que existe el archivo de resultados
    if not os.path.exists('ising_results_optimized.pkl'):
        print("❌ ERROR: No se encontró 'ising_results_optimized.pkl'")
        print("   Ejecute primero: python ising_sampling_optimized.py")
        return False
    
    try:
        # Importar y ejecutar el analizador completo
        from analyze_complete_results import main as analyze_main
        
        print("📊 Ejecutando análisis completo...")
        analyze_main()
        
        print("\n" + "="*60)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\n📁 ARCHIVOS GENERADOS:")
        print("   • comparison_magnetization_vs_beta.png - Gráfica principal")
        print("   • complete_analysis_results.csv - Datos completos")
        print("   • tarea3_final_analysis.ipynb - Notebook completo")
        print("\n🎯 ENTREGABLES LISTOS:")
        print("   1. ✅ 100 muestras MCMC (Metropolis-Hastings)")
        print("   2. ✅ 100 muestras Perfect Sampling (Propp-Wilson)")
        print("   3. ✅ Estimación E[M(η)] para ambos métodos")
        print("   4. ✅ Tiempos de coalescencia reportados")
        print("   5. ✅ Gráfica de comparación vs β")
        print("   6. ✅ Notebook con análisis completo")
        
        return True
        
    except ImportError as e:
        print(f"❌ ERROR de importación: {e}")
        print("   Asegúrese de tener todas las dependencias instaladas:")
        print("   pip install numpy matplotlib pandas seaborn scipy")
        return False
        
    except Exception as e:
        print(f"❌ ERROR durante análisis: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)