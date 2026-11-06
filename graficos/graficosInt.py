import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

# =========================================================================
# DATOS EMPÍRICOS COMPLETOS (N = 4, 8, 10, 14, 16)
# =========================================================================
data_text = """Case,N_Qubits,Direction,Spheres,Fidelity,Time_Decoding,Time_Encoding,Extra_Memory_MB
Separable State,4,1,4,99.99999952,0.000221,0.000022,0.0054
Partial Entanglement With Bell Pairs,4,1,4,100.00000000,0.000191,0.000023,0.0052
Total Entanglement With GHZ State,4,1,4,100.00000000,0.000370,0.000045,0.0049
Robust Entanglement With W-State,4,1,10,99.99999983,0.000300,0.000044,0.0065
Dense Chaotic Entanglement,4,1,15,99.99999803,0.000259,0.000089,0.0093
Separable State,8,1,8,99.99999965,0.0235,0.001,0.0453
Partial Entanglement With Bell Pairs,8,1,8,100.00000000,0.0211,0.001,0.0448
Total Entanglement With GHZ State,8,1,8,100.00000000,0.0181,0.001,0.0455
Robust Entanglement With W-State,8,1,36,99.99999957,0.0184,0.001,0.0534
Dense Chaotic Entanglement,8,1,255,99.99999950,0.0230,0.002,0.1559
Separable State,10,1,10,99.99999967,0.3207,0.002,0.1876
Partial Entanglement With Bell Pairs,10,1,10,100.00000000,0.2921,0.002,0.1876
Total Entanglement With GHZ State,10,1,10,100.00000000,0.2732,0.002,0.1875
Robust Entanglement With W-State,10,1,55,99.99999948,0.2768,0.002,0.1876
Dense Chaotic Entanglement,10,1,1023,99.99999929,0.3157,0.010,0.7329
Separable State,14,1,14,99.99999893,182.283290,0.040940,3.1440
Partial Entanglement With Bell Pairs,14,1,14,100.00000000,124.839399,0.024078,3.1440
Total Entanglement With GHZ State,14,1,14,100.00000000,184.985133,0.037712,3.1440
Robust Entanglement With W-State,14,1,105,99.99999941,183.023477,0.058685,3.1440
Dense Chaotic Entanglement,14,1,16383,99.99993874,183.307579,6.410159,18.0977
Separable State,16,1,16,99.99999841,2902.679629,0.155838,12.6822
Partial Entanglement With Bell Pairs,16,1,16,100.00000000,2847.910828,0.160042,12.6821
Total Entanglement With GHZ State,16,1,16,100.00000000,2853.158731,0.171211,12.6821
Robust Entanglement With W-State,16,1,136,99.99999930,2835.757277,0.268186,12.6821
Dense Chaotic Entanglement,16,1,65535,99.99986111,2919.756586,113.301859,76.6629
"""

df = pd.read_csv(io.StringIO(data_text))
df = df[df['Direction'] == 1].copy()

rename_map = {
    'Separable State': 'Estado Separable (Límite Lineal O(N))',
    'Partial Entanglement With Bell Pairs': 'Pares de Bell (O(N))',
    'Total Entanglement With GHZ State': 'Estado GHZ (Entrelazamiento Total O(N))',
    'Robust Entanglement With W-State': 'Estado W (Entrelazamiento Robusto O(N²))',
    'Dense Chaotic Entanglement': 'Caos Denso (Límite Clásico 2^N - 1)'
}
df['Case_ES'] = df['Case'].replace(rename_map)

plt.style.use('seaborn-v0_8-whitegrid')
n_qubits = sorted(df['N_Qubits'].unique())
cases = list(rename_map.values())
markers = ['o', 's', '^', 'D', 'v']
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#1f77b4']

# =========================================================================
# PANEL 1: LAS 4 GRÁFICAS COMPLETAS PARA EL TFG (2x2)
# =========================================================================
fig = plt.figure(figsize=(18, 12))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

# 1. Rendimiento Temporal
for i, case in enumerate(cases):
    subset = df[df['Case_ES'] == case].sort_values('N_Qubits')
    ax1.plot(subset['N_Qubits'], subset['Time_Decoding'], marker=markers[i],
             linestyle='-' if 'Caos' in case else '--', color=colors[i],
             linewidth=2.5 if 'Caos' in case else 2, markersize=7, label=case)

ax1.set_yscale('log')
ax1.set_title('1. Rendimiento Temporal del Decodificador ($O(2^{2N})$)', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Número de Cúbits ($N$)', fontsize=11)
ax1.set_ylabel('Tiempo de Ejecución (segundos) [Log]', fontsize=11)
ax1.set_xticks(n_qubits)
ax1.legend(fontsize=9, loc='upper left', frameon=True)

# 2. Compresión Topológica (Esferas)
for i, case in enumerate(cases):
    subset = df[df['Case_ES'] == case].sort_values('N_Qubits')
    ax2.plot(subset['N_Qubits'], subset['Spheres'], marker=markers[i],
             linestyle='-' if 'Caos' in case else '--', color=colors[i],
             linewidth=2.5 if 'Caos' in case else 2, markersize=7, label=case)

ref_esferas = [2**n - 1 for n in n_qubits]
ax2.plot(n_qubits, ref_esferas, color='black', linestyle=':', linewidth=2, label='Teórico Clásico ($2^N - 1$)')

ax2.set_yscale('log')
ax2.set_title('2. Compresión Topológica: Esferas Instanciadas', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Número de Cúbits ($N$)', fontsize=11)
ax2.set_ylabel('Número de Esferas Alojadas [Log]', fontsize=11)
ax2.set_xticks(n_qubits)
ax2.legend(fontsize=9, loc='upper left', frameon=True)

# 3. Consumo de Memoria (MiB)
mem_vector = [16 * (1 << n) / (1024**2) for n in n_qubits]
ax3.plot(n_qubits, mem_vector, color='black', linestyle='-.', linewidth=2.5,
         label='Vector Denso Clásico complex128 (16 B/amp)')

for i, case in enumerate(cases):
    subset = df[df['Case_ES'] == case].sort_values('N_Qubits')
    mem_spheres = subset['Spheres'] * 100 / (1024**2)
    ax3.plot(subset['N_Qubits'], mem_spheres, marker=markers[i],
             linestyle='-', color=colors[i], markersize=7, label=case)

ax3.set_yscale('log')
ax3.set_title('3. Consumo de Memoria: Esferas Python vs Vector Clásico (MiB)', fontsize=13, fontweight='bold', pad=10)
ax3.set_xlabel('Número de Cúbits ($N$)', fontsize=11)
ax3.set_ylabel('Memoria Estimada (MiB) [Log]', fontsize=11)
ax3.set_xticks(n_qubits)
ax3.legend(fontsize=8.5, loc='upper left', frameon=True)

# 4. Fidelidad de Cuantización
for i, case in enumerate(cases):
    subset = df[df['Case_ES'] == case].sort_values('N_Qubits')
    ax4.plot(subset['N_Qubits'], subset['Fidelity'], marker=markers[i],
             linestyle='-', color=colors[i], linewidth=2, markersize=7, label=case)

ax4.set_title('4. Precisión: Fidelidad de Cuantización (Punto Fijo 16-bits)', fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel('Número de Cúbits ($N$)', fontsize=11)
ax4.set_ylabel('Fidelidad (%)', fontsize=11)
ax4.set_xticks(n_qubits)
ax4.set_ylim(99.99980, 100.00002)
ax4.ticklabel_format(useOffset=False, style='plain', axis='y')
ax4.legend(fontsize=9, loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig('panel_4_graficas_tfg.png', dpi=300, bbox_inches='tight')
plt.show()