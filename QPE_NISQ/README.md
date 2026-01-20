This project implements Quantum Phase Estimation (QPE) using Qiskit, with a focus on understanding the algorithm at the circuit level and evaluating its behavior under both ideal and realistic execution conditions.

The same QPE circuit is executed in three contexts: ideal simulation, simulation with hardware-derived noise, real-hardware execution path (documented)

The goal of the project is to understand how phase information is encoded, extracted, and degraded as execution moves from theory to NISQ-era conditions.

⸻

QPE Circuit Overview

The QPE circuit estimates the phase associated with an eigenstate of a unitary operator.

At a high level, the circuit: prepares multiple ancilla qubits to store phase information, prepares a system qubit in an eigenstate of the unitary, uses controlled phase operations to apply phase kickback, applies an inverse Quantum Fourier Transform (QFT) to decode the phase, measures the ancilla qubits to obtain a binary phase estimate

The circuit structure itself is independent of how it is executed. Only the execution backend changes between ideal, noisy, and hardware-oriented runs.

⸻

Circuit Construction

Registers: Four ancilla qubits are used to represent the phase in binary form. One system qubit is used as the target of the controlled unitary operations. A classical register stores measurement results from the ancilla qubits.

Phase Kickback

The system qubit is prepared in a known eigenstate so that controlled phase operations transfer phase information onto the ancilla qubits. Each ancilla controls a phase rotation whose exponent increases by powers of two, encoding the phase across the ancilla register.

Inverse QFT

An inverse Quantum Fourier Transform is applied to the ancilla qubits to convert accumulated phase information into a measurable binary representation.

⸻

Ideal Execution (Textbook Behavior)

The circuit is first executed on an ideal simulator with no noise, decoherence, or measurement error.

Purpose to verify correctness of the QPE circuit & confirm that the expected phase bitstring is recovered

In this setting, the output distribution is sharply peaked at the correct binary representation of the phase. This reflects the theoretical behavior of QPE.

⸻

Noisy Execution (Hardware-Derived Simulation)

The same circuit is then executed on a simulator augmented with a noise model extracted from a real IBM quantum device.

The noise model is built from device calibration data, including gate error rates/decoherence effects/readout error

Purpose to evaluate QPE under realistic NISQ conditions & observe how noise degrades phase resolution

Compared to the ideal case, the output distribution shows reduced peak height and increased spread across nearby bitstrings. This highlights the sensitivity of phase-based algorithms to noise.

⸻

Real Hardware Execution Path

The project also includes the execution path required to run QPE on real IBM quantum hardware using the Runtime primitives interface.

This involves: selecting a physical backend, transpiling the circuit to hardware-native gates, executing via a Runtime Session and the Sampler primitive, collecting quasi-probability measurement results

This section documents the correct transition from simulation to physical hardware execution, where noise and imperfections arise from the quantum system itself rather than a model.

⸻

Project Focus

This project emphasizes: phase kickback as a central quantum primitive, the role of the inverse QFT in phase extraction, differences between ideal, noisy, and hardware-oriented execution, practical limitations of QPE in the NISQ regime

The work is intended to build intuition for phase-based quantum algorithms and their behavior under realistic constraints, rather than treating QPE as a black-box routine.
