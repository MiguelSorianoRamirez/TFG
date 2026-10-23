import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# DATOS EMPÍRICOS EXTRAÍDOS DE LA VALIDACIÓN
# ==========================================
n_qubits = [8, 10, 12, 14, 16, 18]

# Tiempo de CPU (segundos)
tiempo_separable = [0.0332, 0.4319, 5.4146, 117.5586, 2020.2077, 47212.6811]
tiempo_parcial = [0.0390, 0.3359, 4.3514, 68.5117, 1710.7145, 48509.6158]
tiempo_ghz = [0.0233, 0.3274, 4.3245, 68.0749, 1078.5832, 31077.5839]

# Memoria: Número de esferas instanciadas
esferas_separable = [255, 1023, 4095, 16383, 65534, 262142]  # Límite clásico O(2^N)
esferas_parcial = [30, 62, 126, 254, 510, 1022]
esferas_ghz = [8, 10, 12, 14, 16, 18]                        # Límite optimizado O(N)

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# GRÁFICA 1: RENDIMIENTO TEMPORAL
# ==========================================
ax1.plot(n_qubits, tiempo_separable, marker='o', linestyle='-', color='#d62728', linewidth=2, label='Separable (Ruido)')
ax1.plot(n_qubits, tiempo_parcial, marker='s', linestyle='--', color='#ff7f0e', linewidth=2, label='Bell (Entrelaz. Parcial)')
ax1.plot(n_qubits, tiempo_ghz, marker='^', linestyle='-.', color='#1f77b4', linewidth=2, label='GHZ (Entrelaz. Total)')

ax1.set_yscale('log')
ax1.set_title('Rendimiento Temporal del Decodificador $O(2^{2N})$', fontsize=14, pad=15)
ax1.set_xlabel('Número de Cúbits ($N$)', fontsize=12)
ax1.set_ylabel('Tiempo de Ejecución (segundos) [Escala Log]', fontsize=12)
ax1.set_xticks(n_qubits)
ax1.legend(fontsize=11)

# ==========================================
# GRÁFICA 2: COMPRESIÓN DE MEMORIA
# ==========================================
ax2.plot(n_qubits, esferas_separable, marker='o', linestyle='-', color='#d62728', linewidth=2, label='Separable (Límite Clásico $2^N-1$)')
ax2.plot(n_qubits, esferas_parcial, marker='s', linestyle='--', color='#ff7f0e', linewidth=2, label='Bell (Compresión Parcial)')
ax2.plot(n_qubits, esferas_ghz, marker='^', linestyle='-', color='#2ca02c', linewidth=3, label='GHZ (Escalamiento Lineal $N$)')

ax2.set_yscale('log')
ax2.set_title('Compresión de Memoria: Esferas vs Complejidad Clásica', fontsize=14, pad=15)
ax2.set_xlabel('Número de Cúbits ($N$)', fontsize=12)
ax2.set_ylabel('Número de Esferas Alojadas [Escala Log]', fontsize=12)
ax2.set_xticks(n_qubits)
ax2.legend(fontsize=11)

# Ajustes finales y guardado
plt.tight_layout()
plt.savefig('benchmark_esferas_bloch.png', dpi=300, bbox_inches='tight')
plt.show()