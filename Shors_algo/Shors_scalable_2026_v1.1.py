#!/usr/bin/env python
# coding: utf-8

# How to run (quick)
# 
# 1. In the "Specify variables" cell, set `N` to an **odd composite** (e.g., 15, 21, 33).
# 2. Keep `N` small.
# 3. Change the variables Max_attempts and shots as needed.
# Things to keep in mind
# 
# - **Aer (ideal)**: Practical up to about `N <= 21` on a typical laptop. Larger values grow quickly in depth and qubit count.
# - **Fake backend (emulator)**: Similar limits to Aer, plus topology/noise effects can reduce success rate.
# - **Real hardware**: Expect smaller limits than Aer. For most devices, `N <= 15` is the realistic ceiling.
# - `N` must be odd and composite. Even numbers or primes will fail or return trivial factors.
# 

# import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
# import pandas as pd
from fractions import Fraction

# # In[ ]:


# Backend selection
# RUN_MODE: 'aer' (ideal noiseless), 'fake' (emulated device), 'hardware' (IBM Quantum)
RUN_MODE = 'aer'

# Hardware config (only used when RUN_MODE == 'hardware')
IBM_TOKEN = ''  # set your token or use saved account
IBM_INSTANCE = ''  # optional: hub/group/project
IBM_BACKEND = ''  # e.g., 'ibm_torino' (must be set for hardware)

def get_backend(mode='aer'):
    if mode == 'aer':
        return AerSimulator(method="matrix_product_state")
    if mode == 'fake':
        # Fake backend emulates a real device's topology/properties
        from qiskit.providers.fake_provider import FakeManila
        fake = FakeManila()
        return AerSimulator.from_backend(fake)
    if mode == 'hardware':
        from qiskit_ibm_runtime import QiskitRuntimeService
        if IBM_BACKEND == '':
            raise ValueError('Set IBM_BACKEND when RUN_MODE=\'hardware\'')
        if IBM_TOKEN:
            service = QiskitRuntimeService(channel='ibm_quantum', token=IBM_TOKEN, instance=IBM_INSTANCE or None)
        else:
            service = QiskitRuntimeService(channel='ibm_quantum', instance=IBM_INSTANCE or None)
        return service.backend(IBM_BACKEND)
    raise ValueError(f'Unknown RUN_MODE: {mode}')


def transpile_no_rccx(qc):
    """Transpile circuit while forcing decomposition away from rccx."""
    # Option A: direct basis transpile that typically avoids rccx emission.
    qc_t = transpile(
        qc,
        basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=None,
        optimization_level=0
    )
    ops = set(qc_t.count_ops().keys())
    print("Transpiled ops:", ops)

    # Option B fallback: explicit rccx decomposition before transpile.
    if "rccx" in ops:
        qc_dec = qc.decompose(["rccx"])
        qc_t = transpile(
            qc_dec,
            basis_gates=["u", "cx"],
            coupling_map=None,
            optimization_level=0
        )
        ops = set(qc_t.count_ops().keys())
        print("Transpiled ops (after rccx decompose):", ops)

    return qc_t


def run_qpe_counts(qc, shots=4, backend=None, n_count=None):
    if backend is None:
        backend = get_backend(RUN_MODE)
    if RUN_MODE in ('aer', 'fake'):
        qc_t = transpile_no_rccx(qc)
        job = backend.run(qc_t, shots=shots)
        result = job.result()
        return result.get_counts(qc_t)
    if RUN_MODE == 'hardware':
        from qiskit_ibm_runtime import Sampler
        if n_count is None:
            raise ValueError('n_count is required for hardware mode')
        qc_t = transpile_no_rccx(qc)    
        sampler = Sampler(backend=backend)
        job = sampler.run([qc_t], shots=shots)
        quasi = job.result().quasi_dists[0]
        total = sum(quasi.values())
        return {format(k, f'0{n_count}b'): int((v/total)*shots) for k, v in quasi.items()}
    raise ValueError(f'Unsupported RUN_MODE: {RUN_MODE}')

backend = get_backend(RUN_MODE)


# # In[4]:


def ripple_carry_adder(n):
    """Simple reversible ripple-carry adder.

    Adds register a into b (b := a + b). a is unchanged.
    Uses an (n+1)-qubit carry register.

    Reversible arithmetic is required in quantum algorithms because
    unitary operations must preserve information (no irreversible
    overwrites).
    """
    a = QuantumRegister(n, 'a')
    b = QuantumRegister(n, 'b')
    c = QuantumRegister(n + 1, 'c')
    qc = QuantumCircuit(a, b, c, name='ADD')
    for i in range(n):
        # Full-adder: carry-out in c[i+1], sum in b[i]
        qc.ccx(a[i], b[i], c[i + 1])
        qc.cx(a[i], b[i])
        qc.ccx(c[i], b[i], c[i + 1])
        qc.cx(c[i], b[i])
    return qc.to_gate()

def controlled_ripple_carry_adder(n):
    """
    ctrl, a[n], b[n], c[n+1]
    If ctrl == 1: b := a + b
    If ctrl == 0: do nothing
    """
    ctrl = QuantumRegister(1, 'ctrl')
    a = QuantumRegister(n, 'a')
    b = QuantumRegister(n, 'b')
    c = QuantumRegister(n + 1, 'c')
    qc = QuantumCircuit(ctrl, a, b, c, name='CADD')

    for i in range(n):
        # controlled full-adder
        qc.mcx([ctrl[0], a[i], b[i]], c[i + 1])
        qc.ccx(ctrl[0], a[i], b[i])
        qc.mcx([ctrl[0], c[i], b[i]], c[i + 1])
        qc.ccx(ctrl[0], c[i], b[i])

    return qc.to_gate()


def _apply_controlled_ripple_carry(qc, ctrl, a, b, c):
    """Inline controlled ripple-carry adder into an existing circuit."""
    n = len(a)
    for i in range(n):
        qc.mcx([ctrl, a[i], b[i]], c[i + 1])
        qc.ccx(ctrl, a[i], b[i])
        qc.mcx([ctrl, c[i], b[i]], c[i + 1])
        qc.ccx(ctrl, c[i], b[i])


def _apply_controlled_add_const(qc, ctrl, x, kreg, c, k):
    """Inline controlled add-constant into an existing circuit."""
    n = len(x)

    # Prepare |k> in kreg, controlled by ctrl.
    for i in range(n):
        if (k >> i) & 1:
            qc.cx(ctrl, kreg[i])

    _apply_controlled_ripple_carry(qc, ctrl, kreg, x, c)

    # Unprepare kreg.
    for i in range(n):
        if (k >> i) & 1:
            qc.cx(ctrl, kreg[i])


def controlled_modular_add_const_circuit(k, N, n):
    """Circuit form of controlled modular add-constant (used for inlining)."""
    ctrl = QuantumRegister(1, 'ctrl')
    x = QuantumRegister(n, 'x')
    kreg = QuantumRegister(n, 'k')
    c = QuantumRegister(n + 1, 'c')
    c_back = QuantumRegister(n + 1, 'c_back')
    flag = QuantumRegister(1, 'flag')

    qc = QuantumCircuit(ctrl, x, kreg, c, c_back, flag, name=f'CADD_{k}_MOD_{N}')

    _apply_controlled_add_const(qc, ctrl[0], list(x), list(kreg), list(c), k)

    two_pow_n = 1 << n
    _apply_controlled_add_const(qc, ctrl[0], list(x), list(kreg), list(c), two_pow_n - N)

    # Compute underflow flag:
    # flag = ctrl AND (NOT c[n]) where c[n] is carry-out of x - N.
    qc.cx(ctrl[0], flag[0])
    qc.mcx([ctrl[0], c[n]], flag[0])

    # Add N back only on underflow.
    _apply_controlled_add_const(qc, flag[0], list(x), list(kreg), list(c_back), N)

    # Uncompute flag back to |0>.
    qc.mcx([ctrl[0], c[n]], flag[0])
    qc.cx(ctrl[0], flag[0])

    return qc

def add_const_gate(k, n):
    """In-place add constant k to n-qubit register x using a work register.

    This is reversible but not optimized; internal carries are cleaned
    when the gate is inverted inside higher-level routines.
    """
    x = QuantumRegister(n, 'x')
    kreg = QuantumRegister(n, 'k')
    c = QuantumRegister(n + 1, 'c')
    qc = QuantumCircuit(x, kreg, c, name=f'ADD_{k}')
    # Prepare |k> in kreg (classical constant)
    for i in range(n):
        if (k >> i) & 1:
            qc.x(kreg[i])
    qc.append(ripple_carry_adder(n), list(kreg) + list(x) + list(c))
    # Unprepare kreg
    for i in range(n):
        if (k >> i) & 1:
            qc.x(kreg[i])
    return qc.to_gate()

def controlled_add_const_gate(k, n):
    """
    ctrl, x[n], kreg[n], c[n+1]
    If ctrl == 1: x := x + k
    """
    ctrl = QuantumRegister(1, 'ctrl')
    x = QuantumRegister(n, 'x')
    kreg = QuantumRegister(n, 'k')
    c = QuantumRegister(n + 1, 'c')

    qc = QuantumCircuit(ctrl, x, kreg, c, name=f'CADD_{k}')
    _apply_controlled_add_const(qc, ctrl[0], list(x), list(kreg), list(c), k)

    return qc.to_gate()

def modular_add_const_gate(k, N, n):
    """
    Add constant k modulo N to an n-qubit register.
    Uncontrolled version.
    """
    x = QuantumRegister(n, 'x')
    kreg = QuantumRegister(n, 'k')
    c = QuantumRegister(n + 1, 'c')
    flag = QuantumRegister(1, 'flag')

    qc = QuantumCircuit(x, kreg, c, flag, name=f'ADD_{k}_MOD_{N}')

    # 1) x = x + k
    qc.append(add_const_gate(k, n), list(x) + list(kreg) + list(c))

    # 2) x = x - N
    two_pow_n = 1 << n
    qc.append(add_const_gate(two_pow_n - N, n), list(x) + list(kreg) + list(c))

    # Capture carry
    qc.cx(c[n], flag[0])

    # 3) If underflow, add N back
    qc.x(flag[0])
    qc.append(add_const_gate(N, n), list(x) + list(kreg) + list(c))
    qc.x(flag[0])

    return qc.to_gate()

def controlled_modular_add_const_gate(k, N, n):
    """
    ctrl, x[n], kreg[n], c[n+1], c_back[n+1], flag[1]
    If ctrl == 1: x := x + k (mod N)
    """
    return controlled_modular_add_const_circuit(k, N, n).to_gate()

def modinv(a, N):
    """Classical modular inverse for use in reversible uncomputation.
    Returns a^{-1} mod N if it exists.
    """
    try:
        return pow(a, -1, N)
    except TypeError:
        # Fallback for older Python versions
        t, new_t = 0, 1
        r, new_r = N, a
        while new_r != 0:
            q = r // new_r
            t, new_t = new_t, t - q * new_t
            r, new_r = new_r, r - q * new_r
        if r > 1:
            raise ValueError('a has no modular inverse')
        if t < 0:
            t += N
        return t


def modular_multiply_const(a, N, n):

    x = QuantumRegister(n, 'x')
    acc = QuantumRegister(n, 'acc')
    kreg = QuantumRegister(n, 'k')
    c = QuantumRegister(n + 1, 'c')
    c_back = QuantumRegister(n + 1, 'c_back')
    flag = QuantumRegister(1, 'flag')

    qc = QuantumCircuit(x, acc, kreg, c, c_back, flag, name=f'MUL_{a}_MOD_{N}')

    for i in range(n):
        k = (a * (2 ** i)) % N
        qc.append(
            controlled_modular_add_const_gate(k, N, n),
            [x[i]] + list(acc) + list(kreg) + list(c) + list(c_back) + list(flag)
        )

    for i in range(n):
        qc.swap(x[i], acc[i])

    a_inv = modinv(a, N)
    for i in reversed(range(n)):
        k_inv = (a_inv * (2 ** i)) % N
        qc.append(
            controlled_modular_add_const_gate(k_inv, N, n).inverse(),
            [x[i]] + list(acc) + list(kreg) + list(c) + list(c_back) + list(flag)
        )

    return qc.to_gate()

def controlled_modular_multiply(a, N, n, debug=False):
    """
    ctrl, x[n], acc[n], kreg[n], c[n+1], c_back[n+1], flag[1], scratch[1]
    If ctrl==1: apply modular_multiply_const(a)
    """
    ctrl = QuantumRegister(1, 'ctrl')
    x = QuantumRegister(n, 'x')
    acc = QuantumRegister(n, 'acc')
    kreg = QuantumRegister(n, 'k')
    c = QuantumRegister(n + 1, 'c')
    c_back = QuantumRegister(n + 1, 'c_back')
    flag = QuantumRegister(1, 'flag')
    scratch = QuantumRegister(1, 'scratch')

    qc = QuantumCircuit(ctrl, x, acc, kreg, c, c_back, flag, scratch, name=f'C_MUL_{a}')

    # Use explicit AND logic: scratch = ctrl AND x[i].
    # scratch must be disjoint from carry qubits to avoid duplicate wiring.

    for i in range(n):
        k = (a * (2 ** i)) % N
        # Compute scratch = ctrl AND x[i].
        qc.ccx(ctrl[0], x[i], scratch[0])

        modadd = controlled_modular_add_const_circuit(k, N, n)
        qc.compose(modadd,
                   qubits=[scratch[0]] + list(acc) + list(kreg) + list(c) + list(c_back) + list(flag),
                   inplace=True)

        # Uncompute scratch.
        qc.ccx(ctrl[0], x[i], scratch[0])

    # Swap acc into x (only when ctrl==1)
    for i in range(n):
        qc.cswap(ctrl[0], x[i], acc[i])

    # Uncompute acc with inverse (same ctrl AND x[i] trick)
    a_inv = modinv(a, N)
    for i in reversed(range(n)):
        k_inv = (a_inv * (2 ** i)) % N

        qc.ccx(ctrl[0], x[i], scratch[0])
        modadd_inv = controlled_modular_add_const_circuit(k_inv, N, n).inverse()
        qc.compose(modadd_inv,
                   qubits=[scratch[0]] + list(acc) + list(kreg) + list(c) + list(c_back) + list(flag),
                   inplace=True)

        qc.ccx(ctrl[0], x[i], scratch[0])

    return qc


def a2jmodN(a, j, N):
    """Compute a^(2^j) (mod N) by repeated squaring (classical helper).

    This mirrors the structure used in the quantum circuit: each counting
    qubit controls a multiplication by a^(2^j) mod N.
    """
    for _ in range(j):
        a = (a * a) % N
    return a


def c_amodN(a, power, N, n):
    """Controlled multiplication by a^(2^power) mod N.

    Modular exponentiation is the heart of Shor's algorithm. The
    repeated-squaring structure (powers of two) is what makes the
    circuit scale with log(N) instead of being hardcoded for a
    particular small N.
    """
    a_power = a2jmodN(a, power, N)
    return modular_multiply_const(a_power, N, n)


def choose_coprime_base(N, rng=None):
    """Pick a random a in [2, N-1] that is coprime to N.

    Returns (a, gcd) so callers can short-circuit if gcd != 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    while True:
        a = int(rng.integers(2, N))
        g = gcd(a, N)
        if g == 1:
            return a, g
        # If g is a non-trivial factor, caller can use it directly.
        return a, g


# Specify variables
# Set N in one place for programmatic sweeps
N = 33
n = int(np.ceil(np.log2(N)))  # register size scales with log2(N)
N_COUNT = n + 1  # was 2n but causing issues with N >15
rng = np.random.default_rng(1)
a, g = choose_coprime_base(N, rng=rng)
if g != 1:
    print(f"Non-trivial factor found classically: {g}")


def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ
    """
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFTdagger"
    return qc

def build_qpe_circuit(a, N, N_COUNT=None, debug_modexp=False):

    n = int(np.ceil(np.log2(N)))
    if N_COUNT is None:
        N_COUNT = n + 1

    # Total qubits:
    # counting (N_COUNT)
    # x (n)
    # acc (n)
    # kreg (n)
    # carry (n+1)
    # carry_back (n+1)
    # flag (1)
    # scratch (1)
    total_qubits = N_COUNT + 5 * n + 4

    qc = QuantumCircuit(total_qubits, N_COUNT)

    # ---- Counting register in |+> ----
    for q in range(N_COUNT):
        qc.h(q)

    # ---- Work register initialized to |1> ----
    x_start = N_COUNT
    qc.x(x_start)

    # ---- Register layout ----
    acc_start = x_start + n
    kreg_start = acc_start + n
    carry_start = kreg_start + n
    carry_back_start = carry_start + (n + 1)
    flag = carry_back_start + (n + 1)
    scratch = flag + 1

    # ---- Register slices ----
    x = [x_start + i for i in range(n)]
    acc = [acc_start + i for i in range(n)]
    kreg = [kreg_start + i for i in range(n)]
    carry = [carry_start + i for i in range(n + 1)]
    carry_back = [carry_back_start + i for i in range(n + 1)]
    work_qubits = x + acc + kreg + carry + carry_back + [flag] + [scratch]

    # Ensure all non-counting qubits are disjoint.
    if len(set(work_qubits)) != len(work_qubits):
        raise ValueError(f'Overlapping work qubits detected: {work_qubits}')

    # ---- Modular exponentiation ----
    for q in range(N_COUNT):
        a_power = a2jmodN(a, q, N)
        append_qubits = [q] + work_qubits

        # Prevent duplicate qubit arguments in append.
        if len(set(append_qubits)) != len(append_qubits):
            raise ValueError(f'Duplicate qubit indices in append list: {append_qubits}')

        mul_circ = controlled_modular_multiply(a_power, N, n, debug=debug_modexp)

        # Compose inlined ops so modular exponentiation is visible in qc.size()/qc.depth().
        qc.compose(mul_circ, qubits=append_qubits, inplace=True)

    # ---- Inverse QFT ----
    qc.append(qft_dagger(N_COUNT), range(N_COUNT))

    # ---- Measure counting register ----
    qc.measure(range(N_COUNT), range(N_COUNT))

    return qc, N_COUNT

# # In[8]:


# Build QPE circuit (construction only; execution is separate)
qc, N_COUNT = build_qpe_circuit(a, N, N_COUNT)
# qc.draw(fold=-1)  # -1 means 'do not fold'


# # In[9]:


# Execute and collect counts
counts = run_qpe_counts(qc, shots=4, backend=backend, n_count=N_COUNT)
# plot_histogram(counts)
rows, measured_phases = [], []
for output in counts:
    decimal = int(output, 2)  # Convert (base 2) string to decimal
    phase = decimal/(2**N_COUNT)  # Find corresponding eigenvalue
    measured_phases.append(phase)
    # Add these values to the rows in our table:
    rows.append([f"{output}(bin) = {decimal:>3}(dec)",
                 f"{decimal}/{2**N_COUNT} = {phase:.2f}"])
# Print the rows in a table
headers=["Register Output", "Phase"]
# df = pd.DataFrame(rows, columns=headers)
# print(df)


# # In[10]:


Fraction(0.666)

# Get fraction that most closely resembles 0.666
# with denominator < N
Fraction(0.666).limit_denominator(N)
rows = []
for phase in measured_phases:
    frac = Fraction(phase).limit_denominator(N)
    rows.append([phase,
                 f"{frac.numerator}/{frac.denominator}",
                 frac.denominator])
# Print as a table
headers=["Phase", "Fraction", "Guess for r"]
# df = pd.DataFrame(rows, columns=headers)
# print(df)


# # In[11]:


a2jmodN(7, 2049, 53)


rng = np.random.default_rng(1)  # reproducible base selection
a, g = choose_coprime_base(N, rng=rng)
print(a)

from math import gcd # greatest common divisor
gcd(a, N)
def qpe_amodN(a, N, shots=4, backend=None, rng=None):
    """Performs quantum phase estimation on the operation a*r mod N.
    Args:
        a (int): This is 'a' in a*r mod N
        N (int): Composite to factor
        shots (int): Number of samples for phase estimation
        backend: Qiskit backend (simulator or hardware).
    Returns:
        float: Estimate of the phase
    """
    qc, N_COUNT = build_qpe_circuit(a, N)
    # Execute on provided backend (simulator or hardware)
    # `memory=True` tells the backend to save each measurement in a list
    qc_t = transpile_no_rccx(qc)
    print(f"[CIRCUIT] qubits={qc.num_qubits} raw_depth={qc.depth()} raw_size={qc.size()} "
          f"transpiled_depth={qc_t.depth()} transpiled_size={qc_t.size()}")
    if rng is None:
        rng = np.random.default_rng()
    if backend is None:
        backend = get_backend(RUN_MODE)

    # ---------------- EXECUTION ----------------
    job = backend.run(qc_t, shots=shots)
    result = job.result()
    counts = result.get_counts(qc_t)
   

    outcomes = list(counts.keys())
    weights = np.array(list(counts.values()), dtype=float)
    weights = weights / weights.sum()
    reading = outcomes[int(rng.choice(len(outcomes), p=weights))]
    print('Register Reading: ' + reading)
    phase = int(reading,2)/(2**N_COUNT)
    print(f'Corresponding Phase: {phase}')
    return phase
#phase = qpe_amodN(a, N, shots=16, rng=rng) # Phase = s/r
#Fraction(phase).limit_denominator(N)

#frac = Fraction(phase).limit_denominator(N)
#s, r = frac.numerator, frac.denominator
#print(r)
#guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
#print(guesses)

##demo block for building the QPE circuit (without execution) ^^
# # In[12]:


a, g = choose_coprime_base(N, rng=rng)
MAX_ATTEMPTS = 3
FACTOR_FOUND = False

for ATTEMPT in range(1, MAX_ATTEMPTS + 1):
    print(f"\nATTEMPT {ATTEMPT}:")
    if g != 1:
        print(f"*** Non-trivial factor found classically: {g} ***")
        FACTOR_FOUND = True
        break
    phase = qpe_amodN(a, N, shots=4, rng=rng) # Phase = s/r
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    print(f"Result: r = {r}")
    if phase != 0:
        # Guesses for factors are gcd(x^{r/2} +- 1 , N)
        guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
        print(f"Guessed Factors: {guesses[0]} and {guesses[1]}")
        for guess in guesses:
            if guess not in [1,N] and (N % guess) == 0:
                other = N // guess
                # Guess is a factor!
                print(f"*** Non-trivial factor found: {guess} and {other} ***")
                FACTOR_FOUND = True
                break
    if FACTOR_FOUND:
        break
    a, g = choose_coprime_base(N, rng=rng)


# Loop terminates when a non-trivial factor is found
assert FACTOR_FOUND
