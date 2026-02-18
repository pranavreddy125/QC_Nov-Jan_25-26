#!/usr/bin/env python
# coding: utf-8

# How to run (quick)
# 
# 1. Open this notebook and run all cells top to bottom.
# 2. In the "Specify variables" cell, set `N` to an **odd composite** (e.g., 15, 21, 33).
# 3. Keep `N` small. This notebook is for demonstration, not large-scale factoring.
# 
# Things to keep in mind
# 
# - **Aer (ideal)**: Practical up to about `N <= 21` on a typical laptop. Larger values grow quickly in depth and qubit count.
# - **Fake backend (emulator)**: Similar limits to Aer, plus topology/noise effects can reduce success rate.
# - **Real hardware**: Expect smaller limits than Aer. For most devices, `N <= 15` is the realistic ceiling.
# - `N` must be odd and composite. Even numbers or primes will fail or return trivial factors.
# 

# # In[3]:


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
        return AerSimulator()
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

def run_qpe_counts(qc, shots=128, backend=None, n_count=None):
    if backend is None:
        backend = get_backend(RUN_MODE)
    if RUN_MODE in ('aer', 'fake'):
        qc_t = transpile(qc, backend)
        job = backend.run(qc_t, shots=shots)
        result = job.result()
        return result.get_counts(qc_t)
    if RUN_MODE == 'hardware':
        from qiskit_ibm_runtime import Sampler
        if n_count is None:
            raise ValueError('n_count is required for hardware mode')
        sampler = Sampler(backend=backend)
        job = sampler.run([qc], shots=shots)
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


def modular_add_const_gate(k, N, n):
    """Add constant k modulo N to an n-qubit register (reversible, not optimized).

    Modular addition is a building block for modular multiplication,
    which in turn dominates the depth of Shor's algorithm.
    """
    x = QuantumRegister(n, 'x')
    kreg = QuantumRegister(n, 'k')
    c = QuantumRegister(n + 1, 'c')
    flag = QuantumRegister(1, 'flag')
    qc = QuantumCircuit(x, kreg, c, flag, name=f'ADD_{k}_MOD_{N}')

    # 1) x = x + k
    qc.append(add_const_gate(k, n), list(x) + list(kreg) + list(c))

    # 2) x = x - N (by adding 2^n - N), capture carry-out in flag
    two_pow_n = 1 << n
    qc.append(add_const_gate(two_pow_n - N, n), list(x) + list(kreg) + list(c))
    qc.cx(c[n], flag[0])

    # 3) If we underflowed (flag == 0), add N back
    qc.x(flag[0])
    qc.append(add_const_gate(N, n).control(), [flag[0]] + list(x) + list(kreg) + list(c))
    qc.x(flag[0])

    return qc.to_gate()


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
    """In-place modular multiplication by a (correct but not optimized).

    This uses reversible modular addition and a clean uncompute step
    based on the modular inverse of a. The structure is scalable
    (polynomial in n = log2(N)) but not hardware-efficient yet.
    """
    x = QuantumRegister(n, 'x')
    acc = QuantumRegister(n, 'acc')
    kreg = QuantumRegister(n, 'k')
    c = QuantumRegister(n + 1, 'c')
    flag = QuantumRegister(1, 'flag')
    qc = QuantumCircuit(x, acc, kreg, c, flag, name=f'MUL_{a}_MOD_{N}')

    # Compute acc = a * x (mod N) using repeated squaring additions
    for i in range(n):
        k = (a * (2 ** i)) % N
        qc.append(modular_add_const_gate(k, N, n).control(),
                  [x[i]] + list(acc) + list(kreg) + list(c) + list(flag))

    # Swap acc into x to make the operation in-place on x
    for i in range(n):
        qc.swap(x[i], acc[i])

    # Uncompute acc using a^{-1} so all ancillas return to |0>
    a_inv = modinv(a, N)
    for i in reversed(range(n)):
        k_inv = (a_inv * (2 ** i)) % N
        qc.append(modular_add_const_gate(k_inv, N, n).inverse().control(),
                  [x[i]] + list(acc) + list(kreg) + list(c) + list(flag))

    return qc.to_gate()


def controlled_modular_multiply(a, N, n):
    """Controlled modular multiplication by a mod N.
    The control qubit comes from the phase estimation register.
    """
    return modular_multiply_const(a, N, n).control()


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
    return controlled_modular_multiply(a_power, N, n)


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
N = 21
n = int(np.ceil(np.log2(N)))  # register size scales with log2(N)
N_COUNT = n + 2  # was 2n but causing issues with N >15
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

def build_qpe_circuit(a, N, N_COUNT=None):
    """Construct the QPE circuit for Shor's algorithm (no execution).

    Returns (qc, N_COUNT).
    """
    n = int(np.ceil(np.log2(N)))
    if N_COUNT is None:
        N_COUNT = 2 * n
    # Create QuantumCircuit with N_COUNT counting qubits
    # plus work + ancilla qubits for modular multiplication
    # Work register: n qubits
    # Ancillas: acc (n), k (n), carry (n+1), flag (1)
    qc = QuantumCircuit(N_COUNT + 4 * n + 2, N_COUNT)
    # Initialize counting qubits in state |+>
    for q in range(N_COUNT):
        qc.h(q)
    # Work register in state |1>
    qc.x(N_COUNT)
    # Modular exponentiation via repeated squaring
    for q in range(N_COUNT):
        x = [N_COUNT + i for i in range(n)]
        acc = [N_COUNT + n + i for i in range(n)]
        kreg = [N_COUNT + 2 * n + i for i in range(n)]
        carry = [N_COUNT + 3 * n + i for i in range(n + 1)]
        flag = N_COUNT + 4 * n + 1
        qc.append(c_amodN(a, q, N, n),
                 [q] + x + acc + kreg + carry + [flag])
    # Inverse QFT and measurements
    qc.append(qft_dagger(N_COUNT), range(N_COUNT))
    qc.measure(range(N_COUNT), range(N_COUNT))
    return qc, N_COUNT


# # In[8]:


# Build QPE circuit (construction only; execution is separate)
qc, N_COUNT = build_qpe_circuit(a, N, N_COUNT)
# qc.draw(fold=-1)  # -1 means 'do not fold'


# # In[9]:


# Execute and collect counts
counts = run_qpe_counts(qc, shots=128, backend=backend, n_count=N_COUNT)
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
def qpe_amodN(a, N, shots=128, backend=None, rng=None):
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
    if rng is None:
        rng = np.random.default_rng()

    if backend is None:
        backend = get_backend(RUN_MODE)
    counts = run_qpe_counts(qc, shots=shots, backend=backend, n_count=N_COUNT)
    if rng is None:
        rng = np.random.default_rng()
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
MAX_ATTEMPTS = 5
FACTOR_FOUND = False

for ATTEMPT in range(1, MAX_ATTEMPTS + 1):
    print(f"\nATTEMPT {ATTEMPT}:")
    if g != 1:
        print(f"*** Non-trivial factor found classically: {g} ***")
        FACTOR_FOUND = True
        break
    phase = qpe_amodN(a, N, shots=128, rng=rng) # Phase = s/r
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