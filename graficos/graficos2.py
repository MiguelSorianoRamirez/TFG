import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# DATOS EMPÍRICOS EXTRAÍDOS DE LA VALIDACIÓN (PRECISIÓN 64-BITS)
# ==========================================
# Solo incluimos hasta N=14 ya que son los datos proporcionados
n_qubits = [8, 10, 12, 14]

# Tiempo de Proceso Matemático (segundos)
tiempo_separable = [0.3365, 4.6970, 114.1452, 1049.4841]
tiempo_parcial   = [0.2100, 3.6404, 104.8234, 956.0783]
tiempo_ghz       = [0.1981, 3.6396, 104.8015, 953.2373]
tiempo_w         = [0.2069, 3.5431, 102.3461, 956.6327]
tiempo_denso     = [0.3434, 4.7934, 116.9395, 1068.0814]

# Memoria: Número de esferas extraídas (Pruning perfecto a 64-bits)
esferas_separable = [8, 10, 12, 14]      # O(N) perfecto sin ruido
esferas_parcial   = [8, 10, 12, 14]      # Aislados en pares, poda a N
esferas_ghz       = [8, 10, 12, 14]      # Simetría total, escala linealmente O(N)
esferas_w         = [36, 55, 78, 105]    # Caso Medio (Polinómico)
esferas_denso     = [255, 1023, 4095, 16383] # Límite clásico caótico O(2^N)

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ==========================================
# GRÁFICA 1: RENDIMIENTO TEMPORAL
# ==========================================
ax1.plot(n_qubits, tiempo_denso, marker='X', linestyle='-', color='#d62728', linewidth=2.5, label='Nivel 4: Denso Caótico (Muro de RAM)')
ax1.plot(n_qubits, tiempo_separable, marker='o', linestyle=':', color='#7f7f7f', linewidth=2, label='Nivel 1: Separable (Sin ruido)')
ax1.plot(n_qubits, tiempo_w, marker='D', linestyle='--', color='#9467bd', linewidth=2, label='Nivel 3.5: Estado W (Caso Medio)')
ax1.plot(n_qubits, tiempo_parcial, marker='s', linestyle='-.', color='#ff7f0e', linewidth=2, label='Nivel 2: Bell (Entrelaz. Parcial)')
ax1.plot(n_qubits, tiempo_ghz, marker='^', linestyle='-', color='#1f77b4', linewidth=2.5, label='Nivel 3: GHZ (Entrelaz. Total)')

ax1.set_yscale('log')
ax1.set_title('Rendimiento Temporal del Decodificador (Python)', fontsize=14, pad=15)
ax1.set_xlabel('Número de Cúbits ($N$)', fontsize=12)
ax1.set_ylabel('Tiempo de Proceso Matemático (segundos) [Escala Log]', fontsize=12)
ax1.set_xticks(n_qubits)
ax1.legend(fontsize=10, loc='upper left')

# ==========================================
# GRÁFICA 2: COMPRESIÓN DE MEMORIA (ESFERAS)
# ==========================================
ax2.plot(n_qubits, esferas_denso, marker='X', linestyle='-', color='#d62728', linewidth=2.5, label='Denso (Límite Clásico $2^N-1$)')
ax2.plot(n_qubits, esferas_w, marker='D', linestyle='--', color='#9467bd', linewidth=2, label='Estado W (Entrelazamiento Robusto)')
# Separable, Bell y GHZ coinciden exactamente en la recta N. Dibujamos GHZ por encima.
ax2.plot(n_qubits, esferas_separable, marker='o', linestyle='', color='#7f7f7f', markersize=8, label='Separable (Escalamiento Exacto $N$)')
ax2.plot(n_qubits, esferas_parcial, marker='s', linestyle='', color='#ff7f0e', markersize=6, label='Pares Bell (Escalamiento Exacto $N$)')
ax2.plot(n_qubits, esferas_ghz, marker='^', linestyle='-', color='#2ca02c', linewidth=3, label='GHZ (Escalamiento Exacto $N$)')

ax2.set_yscale('log')
ax2.set_title('Topología de Memoria: Esferas de Bloch vs Entropía', fontsize=14, pad=15)
ax2.set_xlabel('Número de Cúbits ($N$)', fontsize=12)
ax2.set_ylabel('Número de Esferas Alojadas [Escala Log]', fontsize=12)
ax2.set_xticks(n_qubits)
ax2.legend(fontsize=10, loc='upper left')

# Ajustes finales y guardado
plt.tight_layout()
plt.savefig('benchmark_esferas_64bits.png', dpi=300, bbox_inches='tight')
plt.show()