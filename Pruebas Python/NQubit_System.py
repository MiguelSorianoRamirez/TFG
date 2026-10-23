import cmath
import itertools
import math
import random
import time
import tracemalloc

class BlochSphere:
    """
    We represent the Bloch Sphere with its angular coordinates theta and phi with the native precission of phyton float (64 bits)
    """
    def __init__(self, theta: float = 0.0, phi: float = 0.0):
        self.theta = float(theta)
        self.phi = float(phi)
        self.apply_cannonical_values()

    def apply_cannonical_values(self):
        """
        We apply the function for get every angle in the range [0, 2*pi) and remove the noise if the angle is close to 0 or 2*pi
        """
        # Clean the noise
        if abs(self.theta) < 1e-11: self.theta = 0.0
        if abs(self.phi) < 1e-11: self.phi = 0.0

        # Normalize the angles
        self.theta = self.theta % (2.0 * math.pi)
        self.phi = self.phi % (2.0 * math.pi)

        # Clean again the noise and transform theta and phi that are 2pi to 0.0
        if abs(self.theta - 2.0 * math.pi) < 1e-11: self.theta = 0.0
        if abs(self.phi - 2.0 * math.pi) < 1e-11: self.phi = 0.0

    def is_redundant(self, has_controles: bool) -> bool:
        """
        Checks if its an unitary sphere, if theta is 0 in this case, we return true to remove the sphere.
        In case its not an unitary sphere, we also need to check that phi is 0 to remove the sphere.
        """
        margen = 1e-10
        if not has_controles:
            return math.isclose(self.theta, 0.0, abs_tol=margen)
        else:
            return math.isclose(self.theta, 0.0, abs_tol=margen) and math.isclose(self.phi, 0.0, abs_tol=margen)

    def __eq__(self, other):
        if not isinstance(other, BlochSphere):
            return False
        margen = 1e-10
        return math.isclose(self.theta, other.theta, abs_tol=margen) and math.isclose(self.phi, other.phi, abs_tol=margen)

    def __repr__(self):
        return f"(theta = {self.theta:.5f} rad, phi = {self.phi:.5f} rad)"


class NQubitSystem:
    def __init__(self, n_qubits: int, spheres: dict, gamma: float = 0.0):
        self.n = n_qubits
        self.spheres = spheres 
        self.gamma = float(gamma)

    def to_state_vector(self) -> list[complex]:
        """
        It reconstructs the full N-qubit state vector from the Bloch sphere dictionary.
        Builds the 2^N amplitudes iteratively, processing qubit by qubit and applying
        the rotations of the sphere parameters.
        """
        # Initialization of the vector
        amplitudes = {0: 1.0 + 0j}

        for i in range(1, self.n + 1):
            new_amplitudes = {}
            bit_pos = self.n - i # Calculate bit position for the current qubit, the 1st qubit is the most significant bit

            # Filter and isolate the entaglement spheres that target the qubit i
            target_spheres = [
                (controls, sphere)
                for (target, controls), sphere in self.spheres.items()
                if target == i and controls
            ]

            # Process every active branch generated 
            for prefix, amp in amplitudes.items():
                # We identify which qubit from 1 to i - 1 are in the state |1>
                active_bits = [k for k in range(1, i) if (prefix & (1 << (self.n - k))) != 0]
                active_bits_set = set(active_bits)
                # get the individual uncontrollled (unitary) sphere for the qubit i
                individual_sphere = self.spheres.get((i, frozenset()), BlochSphere(0.0, 0.0))
                # start acummulating the effective angle
                # theta is divided by 2 following R_y(theta/2)
                theta_eff = individual_sphere.theta / 2.0
                phi_eff = individual_sphere.phi
                # Sum the rotations provoked by entanglement if the control bits are active on the branch
                for controls, sphere_J in target_spheres:
                    if controls.issubset(active_bits_set):
                        theta_eff += sphere_J.theta
                        phi_eff += sphere_J.phi
                # Bifurcate the probability by updating the |1> branch (activate the bit using the OR bitwise
                # Apply the sine function and the accumulated phase phi_eff as complex exponential
                new_amplitudes[prefix] = amp * math.cos(theta_eff)
                new_amplitudes[prefix | (1 << bit_pos)] = amp * math.sin(theta_eff) * cmath.exp(1j * phi_eff)

            amplitudes = new_amplitudes
        # Once every branch is expanded, apply the global phase to all the amplitudes.
        phase_global = cmath.exp(1j * self.gamma)
        return [amplitudes[x] * phase_global for x in range(1 << self.n)]
    
    @classmethod
    def from_state_vector(cls, x_input: list[complex]):
        """
        Compresses a 2^N state vector into a dictionary of Bloch spheres,
        following the algorithm described in Theorem 3.
        """
        EPS = 1e-10 
        num_amps = len(x_input)
        n = int(math.log2(num_amps))

        # Normalization
        norma = math.sqrt(math.fsum([abs(c) ** 2 for c in x_input]))
        x_norm = [c / norma for c in x_input]

        # Extract the Global Phase gamma
        # We ensure the first non-zero amplitude is real by extracting its phase
        gamma = 0.0
        for comp in x_norm:
            if abs(comp) > EPS:
                gamma = cmath.phase(comp)
                break

        # Rotate the whole vector backwards to factor out the global phase
        phase_fix = cmath.exp(-1j * gamma)
        x = [c * phase_fix for c in x_norm]

        # Separate moduli for theta and relative phases for phi
        spheres = {}
        r = [abs(c) for c in x]
        phases = [cmath.phase(c) for c in x]

        # Hierarchical Top-Down decoding Qubit by Qubit
        for i in range(1, n + 1):
            bit_pos_i = n - i
            # The current qubit can only be controlled by preceding ones
            control_indices = list(range(1, i))

            # Iterate over all possible combinations of control qubits
            for control_len in range(0, len(control_indices) + 1):
                for J_tuple in itertools.combinations(control_indices, control_len):
                    J = frozenset(J_tuple)

                    #Base Case, Unitary Sphere (no controls, J is empty)
                    if not J:
                        # Isolate branches where qubit is 0 and 1 requiring all the preceding qubits are 0 (without entanglement influence)
                        mask_qi_0 = [idx for idx in range(num_amps) if (idx & (1 << bit_pos_i)) == 0 and all((idx & (1 << (n - k))) == 0 for k in range(1, i))]
                        mask_qi_1 = [idx for idx in range(num_amps) if (idx & (1 << bit_pos_i)) != 0 and all((idx & (1 << (n - k))) == 0 for k in range(1, i))]
                        # Add the probabilities in those branches
                        norm_0 = math.sqrt(math.fsum([r[idx] ** 2 for idx in mask_qi_0]))
                        norm_1 = math.sqrt(math.fsum([r[idx] ** 2 for idx in mask_qi_1]))

                        # Calculate the base angle multiplied by 2, theta defines the half-angle amplitude
                        theta = 2.0 * math.atan2(norm_1, norm_0)
                        phi = phases[1 << bit_pos_i]

                        # only add the sphere if its not redundant
                        esfera = BlochSphere(theta, phi)
                        if not esfera.is_redundant(has_controles=False):
                            spheres[(i, J)] = esfera

                    # Entanglement Spheres
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

                        # If both norms are 0, this branch has no probability, and, to preserve bijection, we assign the canonical value 0.0
                        if norm_0 < EPS and norm_1 < EPS:
                            theta = 0.0
                        else:
                            # Calculate the effective angle:
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
                            # The remaining value is the contribution of this J
                            theta = theta_eff - ind_theta - sub_theta_sum

                        # Extract the conditional phase phi
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

                        # Add the sphere if its not redundant
                        esfera = BlochSphere(theta, phi)
                        if not esfera.is_redundant(has_controles=True):
                            spheres[(i, J)] = esfera

        return cls(n, spheres, gamma)



"""
TESTS DE RECONSTRUCCIÓN 
"""



def calculate_fidelity(original_vector, reconstructed_vector) -> float:
    """
    Measures fidelity between two quantum states.
    Fidelity is the squared absolute value of the inner product of both vectors
    A fidelity of 1.0 means the vectors are identical
    """
    inner_product = sum(c_orig.conjugate() * c_rec for c_orig, c_rec in zip(original_vector, reconstructed_vector))
    return abs(inner_product) ** 2

def generate_separable_vector(n_qubits: int) -> list[complex]:
    """
    Generates a pure separable state
    Simulates a state where each qubit is in a random independent superposition,
    meaning the system is a tensor product of individual qubits
    The model should compress this to exactly N spheres.
    """
    state_vector = [1.0 + 0j]
    for _ in range(n_qubits):
        alpha = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        beta = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        # Normalize individual qubit
        norm = math.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha, beta = alpha / norm, beta / norm
        
        # Tensor product with the current state vector
        state_vector = [amp * coef for amp in state_vector for coef in (alpha, beta)]
    return state_vector

def generate_partial_entanglement_vector(n_qubits: int) -> list[complex]:
    """
    Generates a state composed of isolated Bell pairs for Partial Entanglement.
    Pairs adjacent qubits into maximum entangled states,
    but keeps the pairs independent from each other.
    """
    state_vector = [1.0 + 0j]
    for i in range(0, n_qubits, 2):
        if i + 1 < n_qubits:
            # Create a Bell pair |Phi+> = (|00> + |11>) / sqrt(2)
            bell_pair_amplitudes = [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)]
            state_vector = [amp * b for amp in state_vector for b in bell_pair_amplitudes]
        else:
            # If there is an odd qubit left, leave it in a simple superposition
            state_vector = [amp * coef for amp in state_vector for coef in (1/math.sqrt(2), 1/math.sqrt(2))]
    return state_vector

def generate_ghz_vector(n_qubits: int) -> list[complex]:
    """
    Generates a GHZ state (Total Entanglement, Perfect Symmetry).
    The state is (|00...0> + |11...1>) / sqrt(2).
    Massive entanglement, its symmetry allows the hierarchical model to compress it to O(N) spheres.
    """
    dim = 1 << n_qubits
    state_vector = [0j] * dim
    state_vector[0] = 1.0 / math.sqrt(2)
    state_vector[-1] = 1.0 / math.sqrt(2)
    return state_vector

def generate_dense_random_vector(n_qubits: int) -> list[complex]:
    """
    Generates a fully entangled quantum state without simmetry.
    Acts as the worst-case scenario.
    Forces the model to the theoretical limit of 2^N - 1 spheres.
    """
    dim = 1 << n_qubits
    state_vector = [complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(dim)]
    
    # Normalize the vector
    norm = math.sqrt(sum(abs(c)**2 for c in state_vector))
    return [c / norm for c in state_vector]

def generate_w_state_vector(n_qubits: int) -> list[complex]:
    """
    Generates a W-state (Global Entanglement, Asymmetric).
    Acts as the Average-case scenario. The superposition is evenly distributed 
    only among states with a single '1' bit (e.g. |100>, |010>, |001>).
    """
    dim = 1 << n_qubits
    state_vector = [0j] * dim
    
    amplitude = 1.0 / math.sqrt(n_qubits)
    for i in range(n_qubits):
        # 1 << i generates binary numbers with exactly one '1' (powers of 2)
        state_vector[1 << i] = amplitude + 0j
        
    return state_vector

def run_performance_tests(n_qubits=8):
    print(f"-"*50)
    print(f" MATHEMATICAL EVALUATION (64-BIT PRECISION) FOR {n_qubits} QUBITS")
    print(f"-"*50)
    
    scenarios = [
        ("Case 1: Separable State ", generate_separable_vector),
        ("Case 2: Partial Entanglement With Bell Pairs", generate_partial_entanglement_vector),
        ("Case 3: Total Entanglement With GHZ State", generate_ghz_vector),
        ("Case 4: Robust Entanglement With W-State", generate_w_state_vector),
        ("Case 5: Dense Chaotic Entanglement", generate_dense_random_vector)
    ]
    
    for scenario_name, generator_function in scenarios:
        print(f"\n{scenario_name}")
        original_vector = generator_function(n_qubits)
        tracemalloc.start()
        start_time = time.time()
        quantum_system = NQubitSystem.from_state_vector(original_vector)
        reconstructed_vector = quantum_system.to_state_vector()
        execution_time = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        fidelity = calculate_fidelity(original_vector, reconstructed_vector)
        spheres_used = len(quantum_system.spheres)
        peak_ram_mb = peak_mem / (1024 * 1024)
        
        print(f"Spheres extracted: {spheres_used}")
        print(f"Quantization Fidelity (F): {fidelity * 100:.8f}%")
        print(f"Process time: {execution_time:.4f} s")
        print(f"MAX RAM Memory utilized: {peak_ram_mb:.4f} MB")

if __name__ == "__main__":
    run_performance_tests(n_qubits=8)
    run_performance_tests(n_qubits=10)
    run_performance_tests(n_qubits=12)
    run_performance_tests(n_qubits=14)
    run_performance_tests(n_qubits=16)