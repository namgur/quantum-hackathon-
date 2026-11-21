
import stim
import pymatching
import numpy as np

# Simulation parameters
distance = 3
shots = 10000
p = 0.005

# Generate a noisy surface-code circuit
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=distance,
    rounds=distance,
    after_clifford_depolarization=p,
    before_round_data_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p,
)

# Build a detector error model
model = circuit.detector_error_model(
    decompose_errors=True
)

# Initialize the MWPM decoder
decoder = pymatching.Matching.from_detector_error_model(model)

# Run the experiment
sampler = circuit.compile_detector_sampler()
syndromes, actual = sampler.sample(
    shots=shots,
    separate_observables=True
)

# Predict logical observable flips
predicted = decoder.decode_batch(syndromes)

# Calculate the logical failure rate
failures = np.any(predicted != actual, axis=1)
logical_error_rate = np.mean(failures)

print("Code distance:", distance)
print("Physical error probability:", p)
print("Simulation shots:", shots)
print("Logical failures:", np.sum(failures))
print("Logical error rate:", logical_error_rate)
