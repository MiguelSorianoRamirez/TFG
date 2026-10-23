import cmath
import itertools
import math
import random
import sys
import time
import numpy as np
import tracemalloc

class BlochSphere:
    """
    We represent the Bloch Sphere with its angular coordinates theta and phi.
    In this original version, angles are truncated to 16-bit precision using np.float16
    to simulate hardware limitations, though the underlying Python float remains 64-bit.
    """
    def __init__(self, theta: float = 0.0, phi: float = 0.0):
        self.theta = float(theta)
        self.phi = float(phi)
        self.aplicar_valores_canonicos()

    def aplicar_valores_canonicos(self):
        """
        Forces every angle to the natural [0, 2*pi) period and removes
        numerical noise if the angle is indistinguishably close to 0 or 2*pi.
        """
        # Clean the noise
        if abs(self.theta) < 1e-11: self.theta = 0.0
        if abs(self.phi) < 1e-11: self.phi = 0.0

        # Normalize the angles
        self.theta = self.theta % (2.0 * math.pi)
        self.phi = self.phi % (2.0 * math.pi)
        
        # Clean noise again to transform values slightly under 2pi to exactly 0.0
        if abs(self.theta - 2.0 * math.pi) < 1e-11: self.theta = 0.0
        if abs(self.phi - 2.0 * math.pi) < 1e-11: self.phi = 0.0

        # Truncate to simulate 16-bit hardware precision (Introduces quantization noise)
        self.theta = float(np.float16(self.theta))
        self.phi = float(np.float16(self.phi))

     
    def es_redundante(self, tiene_controles: bool) -> bool:
        """
        Checks if a sphere can be safely pruned to optimize RAM.
        If it's an isolated (unitary) sphere, we prune it if theta is 0 (since it applies no amplitude).
        If it's an entanglement sphere (has controls), we must ensure both theta and phi are 0 
        to guarantee it applies no phase shifts.
        """
        margen = 1e-4
        if not tiene_controles:
            return math.isclose(self.theta, 0.0, abs_tol=margen)
        else:
            return math.isclose(self.theta, 0.0, abs_tol=margen) and math.isclose(self.phi, 0.0, abs_tol=margen)

    # Not strictly necessary, checks if two spheres are mathematically equivalent
    def __eq__(self, other):
        if not isinstance(other, BlochSphere):
            return False
        margen = 1e-4
        return math.isclose(self.theta, other.theta, abs_tol=margen) and math.isclose(self.phi, other.phi, abs_tol=margen)

    # Not strictly necessary, structured representation
    def __repr__(self):
        return f"(theta = {self.theta:.5f} rad, phi = {self.phi:.5f} rad)"


class NQubitSystem:
    def __init__(self, n_qubits: int, spheres: dict, gamma: float = 0.0):
        self.n = n_qubits
        self.spheres = spheres 
        self.gamma = float(gamma)

    def to_state_vector(self) -> list[complex]:
        """
        Reconstructs the full N-qubit state vector from the Bloch sphere dictionary.
        Builds the 2^N amplitudes iteratively, processing qubit by qubit and applying
        the superposition rotations dictated by the sphere parameters.
        """
        # Initialization of the state vector (base state |00...0>)
        amplitudes = {0: 1.0 + 0j}

        for i in range(1, self.n + 1):
            new_amplitudes = {}
            # Calculate bit position for the current qubit (assuming the 1st qubit is the MSB)
            bit_pos = self.n - i 
            
            # Filter and isolate the entanglement spheres that target the current qubit 'i'
            target_spheres = [
                (controls, sphere)
                for (target, controls), sphere in self.spheres.items()
                if target == i and controls
            ]

            # Process every active branch generated so far
            for prefix, amp in amplitudes.items():
                # Identify which preceding qubits (1 to i-1) are in the |1> state using a bitmask
                active_bits = [k for k in range(1, i) if (prefix & (1 << (self.n - k))) != 0]
                active_bits_set = set(active_bits)

                # Get the individual uncontrolled (unitary) sphere for qubit 'i'
                individual_sphere = self.spheres.get((i, frozenset()), BlochSphere(0.0, 0.0))
                
                # Start accumulating the effective angle.
                # theta is divided by 2 following the standard R_y(theta/2) rotation
                theta_eff = individual_sphere.theta / 2.0
                phi_eff = individual_sphere.phi

                # Sum the rotations provoked by entanglement spheres if their control bits are active
                for controls, sphere_J in target_spheres:
                    if controls.issubset(active_bits_set):
                        theta_eff += sphere_J.theta
                        phi_eff += sphere_J.phi

                # Bifurcate the probability by updating the |0> branch (prefix remains the same)
                new_amplitudes[prefix] = amp * math.cos(theta_eff)
                
                # Bifurcate the probability by updating the |1> branch (activate the bit using bitwise OR)
                # Apply the sine function and the accumulated phase phi_eff as a complex exponential
                new_amplitudes[prefix | (1 << bit_pos)] = amp * math.sin(theta_eff) * cmath.exp(1j * phi_eff)

            amplitudes = new_amplitudes

        # Once every branch is expanded, apply the global phase to all the amplitudes
        phase_global = cmath.exp(1j * self.gamma)
        return [amplitudes[x] * phase_global for x in range(1 << self.n)]
    
    @classmethod
    def from_state_vector(cls, x_input: list[complex]):
        """
        Compresses a dense 2^N state vector into a sparse dictionary of Bloch spheres,
        following the Top-Down algorithm described in Theorem 3.
        """
        EPS = 1e-10 
        num_amps = len(x_input)
        n = int(math.log2(num_amps))

        # 1. Normalization
        norma = math.sqrt(math.fsum([abs(c) ** 2 for c in x_input]))
        x_norm = [c / norma for c in x_input]

        # 2. Extract the Global Phase gamma
        # We ensure the first non-zero amplitude is real by extracting its phase
        gamma = 0.0
        for comp in x_norm:
            if abs(comp) > EPS:
                gamma = cmath.phase(comp)
                break

        # Rotate the whole vector backwards to factor out the global phase
        phase_fix = cmath.exp(-1j * gamma)
        x = [c * phase_fix for c in x_norm]

        # 3. Separate moduli for theta and relative phases for phi
        spheres = {}
        r = [abs(c) for c in x]
        phases = [cmath.phase(c) for c in x]

        # 4. Hierarchical Top-Down decoding Qubit by Qubit
        for i in range(1, n + 1):
            bit_pos_i = n - i
            # The current qubit can only be controlled by preceding ones
            control_indices = list(range(1, i))

            # Iterate over all possible combinations of control qubits
            for control_len in range(0, len(control_indices) + 1):
                for J_tuple in itertools.combinations(control_indices, control_len):
                    J = frozenset(J_tuple)

                    # 5. Base Case: Unitary Sphere (no controls, J is empty)
                    if not J:
                        # Isolate branches where qubit is 0 and 1 requiring all the preceding qubits are 0 
                        # (without prior entanglement influence)
                        mask_qi_0 = [idx for idx in range(num_amps) if (idx & (1 << bit_pos_i)) == 0 and all((idx & (1 << (n - k))) == 0 for k in range(1, i))]
                        mask_qi_1 = [idx for idx in range(num_amps) if (idx & (1 << bit_pos_i)) != 0 and all((idx & (1 << (n - k))) == 0 for k in range(1, i))]

                        # Add the probabilities in those branches
                        norm_0 = math.sqrt(math.fsum([r[idx] ** 2 for idx in mask_qi_0]))
                        norm_1 = math.sqrt(math.fsum([r[idx] ** 2 for idx in mask_qi_1]))

                        # Calculate the base angle. Multiplied by 2 because theta defines the half-angle amplitude
                        theta = 2.0 * math.atan2(norm_1, norm_0)
                        phi = phases[1 << bit_pos_i]

                        # Pruning: only add the sphere if its not redundant
                        esfera = BlochSphere(theta, phi)
                        if not esfera.es_redundante(tiene_controles=False):
                            spheres[(i, J)] = esfera

                    # 6. Entanglement Spheres (Conditional Spheres)
                    else:
                        # Isolate branches where all control qubits in J are 1 and the other preceding qubits not in J are 0
                        mask_J_0 = [
                            idx for idx in range(num_amps)
                            if (idx & (1 << bit_pos_i)) == 0
                            and all((idx & (1 << (n - k))) != 0 for k in J_tuple)
                            and all((idx & (1 << (n - k))) == 0 for k in control_indices if k not in J_tuple)
                        ]
                        mask_J_1 = [
                            idx for idx in range(num_amps)
                            if (idx & (1 << bit_pos_i)) != 0
                            and all((idx & (1 << (n - k))) != 0 for k in J_tuple)
                            and all((idx & (1 << (n - k))) == 0 for k in control_indices if k not in J_tuple)
                        ]

                        norm_0 = math.sqrt(math.fsum([r[idx] ** 2 for idx in mask_J_0]))
                        norm_1 = math.sqrt(math.fsum([r[idx] ** 2 for idx in mask_J_1]))

                        # If both norms are 0, this branch has no probability. 
                        # To preserve bijection, we assign the canonical value 0.0
                        if norm_0 < EPS and norm_1 < EPS:
                            theta = 0.0
                        else:
                            # 7. Reverse Engineering the Effective Angle
                            # Calculate the effective theta for the specific entanglement branch
                            theta_eff = math.atan2(norm_1, norm_0)
                            
                            # Subtract the unitary angle of the qubit
                            ind_theta = spheres.get((i, frozenset()), BlochSphere(0,0)).theta / 2.0
                            
                            # Subtract all lower order entanglement angles that fired before this one
                            sub_theta_sum = sum(
                                spheres.get((i, frozenset(K_tuple)), BlochSphere(0,0)).theta
                                for K_len in range(1, len(J_tuple))
                                for K_tuple in itertools.combinations(J_tuple, K_len)
                            )
                            # The remaining value is the pure parametric contribution of this J
                            theta = theta_eff - ind_theta - sub_theta_sum

                        # 8. Extract the conditional phase phi
                        idx_1_pure = (1 << bit_pos_i) + sum(1 << (n - k) for k in J_tuple)
                        
                        # Phase is only meaningful if there is probability amplitude
                        if r[idx_1_pure] > EPS:
                            # Get the phase by subtracting lower order phase contributions
                            target_prior_phase = spheres.get((i, frozenset()), BlochSphere(0,0)).phi + sum(
                                spheres.get((i, frozenset(K_tuple)), BlochSphere(0,0)).phi
                                for K_len in range(1, len(J_tuple))
                                for K_tuple in itertools.combinations(J_tuple, K_len)
                            )
                            prior_qubits_phase = 0.0
                            for k in J_tuple:
                                prior_qubits_phase += spheres.get((k, frozenset()), BlochSphere(0,0)).phi
                                prev_controls = [c for c in J_tuple if c < k]
                                for r_c in range(1, len(prev_controls) + 1):
                                    for H_tuple in itertools.combinations(prev_controls, r_c):
                                        prior_qubits_phase += spheres.get((k, frozenset(H_tuple)), BlochSphere(0,0)).phi

                            pred_phi_prior = target_prior_phase + prior_qubits_phase
                            phi = phases[idx_1_pure] - pred_phi_prior
                        else:
                            phi = 0.0 # Canonical value if probability is 0

                        # Pruning: Add the sphere if its not mathematically redundant
                        esfera = BlochSphere(theta, phi)
                        if not esfera.es_redundante(tiene_controles=True):
                            spheres[(i, J)] = esfera

        return cls(n, spheres, gamma)


# =====================================================================
# RECONSTRUCTION TESTS AND DEGREES OF ENTANGLEMENT
# =====================================================================

def calcular_fidelidad(v_orig, v_rec):
    """
    Measures fidelity between two quantum states.
    Fidelity is the squared absolute value of the inner product of both vectors.
    A fidelity of 1.0 means the vectors are identical.
    """
    producto_interno = sum(c_orig.conjugate() * c_rec for c_orig, c_rec in zip(v_orig, v_rec))
    return abs(producto_interno) ** 2

def vector_aleatorio_separable(n_qubits):
    """
    Generates a pure separable state.
    Simulates a state where each qubit is in a random independent superposition,
    meaning the system is a tensor product of individual qubits.
    The model should compress this to exactly N spheres.
    """
    v = [1.0 + 0j]
    for _ in range(n_qubits):
        alpha = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        beta = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        # Normalize individual qubit
        norm = math.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha, beta = alpha/norm, beta/norm
        
        # Tensor product with the current state vector
        v = [amp * coef for amp in v for coef in (alpha, beta)]
    return v

def vector_entrelazamiento_parcial(n_qubits):
    """
    Generates a state composed of isolated Bell pairs for Partial Entanglement.
    Pairs adjacent qubits into maximum entangled states,
    but keeps the pairs independent from each other.
    """
    v = [1.0 + 0j]
    for i in range(0, n_qubits, 2):
        if i + 1 < n_qubits:
            # Create a Bell pair |Phi+> = (|00> + |11>) / sqrt(2)
            bell = [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)]
            v = [amp * b for amp in v for b in bell]
        else:
            # If there is an odd qubit left, leave it in a simple superposition
            v = [amp * coef for amp in v for coef in (1/math.sqrt(2), 1/math.sqrt(2))]
    return v

def vector_entrelazamiento_total(n_qubits):
    """
    Generates a GHZ state (Total Entanglement, Perfect Symmetry).
    The state is (|00...0> + |11...1>) / sqrt(2).
    Massive entanglement, its symmetry allows the hierarchical model to compress it to O(N) spheres.
    """
    dim = 1 << n_qubits
    v = [0j] * dim
    v[0] = 1.0 / math.sqrt(2)
    v[-1] = 1.0 / math.sqrt(2)
    return v

def test_ejecucion_hardware(n_qubits=8):
    print(f"="*20)
    print(f" EVALUACIÓN FÍSICA (QPU) - RESOLUCIÓN 16-BITS PARA {n_qubits} CÚBITS")
    print(f"="*20)
    
    escenarios = [
        ("Nivel 1: Estado Separable (0 Entrelazamiento)", vector_aleatorio_separable),
        ("Nivel 2: Entrelazamiento Parcial (Pares Bell)", vector_entrelazamiento_parcial),
        ("Nivel 3: Entrelazamiento Total (Estado GHZ)", vector_entrelazamiento_total)
    ]
    
    for nombre, generador in escenarios:
        print(f"\n{nombre}")
        v_orig = generador(n_qubits)
        
        t_start = time.time()
        # 1. Extraction (Angles assume noise and are truncated to 16-bits via np.float16)
        sistema = NQubitSystem.from_state_vector(v_orig)
        # 2. Reconstruction with discretized hardware
        v_rec = sistema.to_state_vector()
        t_exec = time.time() - t_start
        
        fidelidad = calcular_fidelidad(v_orig, v_rec)
        esferas_usadas = len(sistema.spheres)
        
        print(f"  -> Esferas extraídas: {esferas_usadas}")
        print(f"  -> Fidelidad Física del Hardware (F): {fidelidad * 100:.4f}%")
        print(f"  -> Tiempo de proceso matemático: {t_exec:.4f} s")
        
        # In real QPUs, a fidelity above 95% at this bit level is an absolute success
        assert fidelidad > 0.90, f"Pérdida inaceptable de fidelidad por cuantización: {fidelidad}"


def test_rendimiento_masivo():
    print("===================================================================")
    print(" BENCHMARK EXTREMO: ESCALABILIDAD DE TIEMPO Y MEMORIA RAM")
    print("===================================================================")
    
    # We go up to N=10, 12, 14. 
    # N=14 will generate 16.384 amplitudes and a massive double loop in Python.
    niveles_qubits = [10, 12, 14] 
    
    for n in niveles_qubits:
        dim = 1 << n
        print(f"\n[Evaluando Sistema N = {n} cúbits | Amplitudes densas: {dim}]")
        
        # We generate a structured entangled state (mix of separable and Bell)
        # to test memory pruning vs full vector.
        v_orig = [1.0 + 0j]
        for i in range(0, n, 2):
            if i + 1 < n:
                bell = [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)]
                v_orig = [amp * b for amp in v_orig for b in bell]
            else:
                v_orig = [amp * coef for amp in v_orig for coef in (1/math.sqrt(2), 1/math.sqrt(2))]
                
        # Strict RAM (Profiler) and Time measurement
        tracemalloc.start()
        t_start = time.time()
        
        # Mathematical Work: Extraction and Pruning
        sistema = NQubitSystem.from_state_vector(v_orig)
        
        t_dec = time.time() - t_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Mathematical Work: Reconstruction to evaluate fidelity
        v_rec = sistema.to_state_vector()
        fidelidad = calcular_fidelidad(v_orig, v_rec)
        
        # Metrics calculation
        pico_mb = peak_mem / (1024 * 1024)
        esferas_usadas = len(sistema.spheres)
        esferas_teoricas = dim - 1
        compresion = (1 - (esferas_usadas / esferas_teoricas)) * 100
        
        print(f"  -> Tiempo de CPU (Cod/Decod): {t_dec:.4f} segundos")
        print(f"  -> Pico de Memoria RAM consumida: {pico_mb:.4f} MB")
        print(f"  -> Esferas alojadas: {esferas_usadas} de {esferas_teoricas} teóricas (Ahorro del {compresion:.2f}%)")
        print(f"  -> Fidelidad Física post-cuantización (16-bits): {fidelidad * 100:.6f}%")


if __name__ == "__main__":
    #test_ejecucion_hardware(n_qubits=8)
    #test_ejecucion_hardware(n_qubits=10)
    #test_ejecucion_hardware(n_qubits=12)
    #test_ejecucion_hardware(n_qubits=14)
    #test_ejecucion_hardware(n_qubits=16)
    test_ejecucion_hardware(n_qubits=18)
    #test_rendimiento_masivo()