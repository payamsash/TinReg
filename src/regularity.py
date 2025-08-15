import serial
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']  # Use PTB for precision audio
from psychopy import sound, core, gui, logging, event, visual
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# Initialize the serial port for trigge rs
ser = serial.Serial('COM4', baudrate=19200, timeout=1)

# Log save path
log_save_path = "experiment_log.json"

# Define standard frequencies
sfrq = [440, 587, 782, 1043]

# Compute octave range of standard frequencies
octave_range = np.log2(max(sfrq) / min(sfrq))

# Compute approximate central f0 for standard test, will be displayed in print later on....
f0_standard = np.sqrt(min(sfrq) * max(sfrq))

# Trigger sending function
def send_trigger(trigger_number):
    ser.write(bytes([trigger_number]))
    core.wait(0.001)
    ser.write(bytes([0]))  # Reset trigger
    print(f"Trigger {trigger_number} sent")

# Convert dB SPL to amplitude with 25dB attenuation for calibrated presentation
def db_to_amplitude(target_dB_SPL, reference_dB_SPL=100):
    return 10 ** ((target_dB_SPL - reference_dB_SPL - 25) / 20)  # Added -25dB attenuation

# Generate logarithmically spaced frequencies within the standard octave range
def generate_custom_frequencies(f0, octave_range):
    log_spacing = octave_range / 4  # Equal spacing in log2 domain
    f1 = round(f0 * 2**(-2 * log_spacing))
    f2 = round(f0 * 2**(-log_spacing))
    f3 = round(f0)
    f4 = round(f0 * 2**(log_spacing))
    f5 = round(f0 * 2**(2 * log_spacing))
    return [f1, f2, f3, f4, f5]

# Function to display start instructions
def show_start_instructions():
    instructions = visual.TextStim(win, text="Start experiment with spacebar when ready.", pos=(0, 0))
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=['space'])


# Get user-defined f0 and study ID
info = {'Enter f0 (Hz)': '1000', 'Study Shortname': ''}
dlg = gui.DlgFromDict(dictionary=info, title="Experiment Setup")
if not dlg.OK:
    core.quit()

# Convert f0 to float and get study ID
f0 = float(info['Enter f0 (Hz)'])
study_id = str(info['Study Shortname'])  # Get study ID from GUI

# Generate custom frequency set based on standard octave range
cfrq = generate_custom_frequencies(f0, octave_range)

# Define frequency labels, for factors orderliness and frequency type, à 4 tones, 2*2*4 = 16
frequency_labels = [
    "f1_std_or", "f2_std_or", "f3_std_or", "f4_std_or",
    "f1_std_rndm", "f2_std_rndm", "f3_std_rndm", "f4_std_rndm",
    "f1_tin_or", "f2_tin_or", "f3_tin_or", "f4_tin_or",
    "f1_tin_rndm", "f2_tin_rndm", "f3_tin_rndm", "f4_tin_rndm"
]

# Transition matrix for ordered sequence
trans_matrix_or = {
    'f4_std_or': {'f4_std_or': 0.25, 'f3_std_or': 0.75, 'f2_std_or': 0.00, 'f1_std_or': 0.00},
    'f3_std_or': {'f4_std_or': 0.00, 'f3_std_or': 0.25, 'f2_std_or': 0.75, 'f1_std_or': 0.00},
    'f2_std_or': {'f4_std_or': 0.00, 'f3_std_or': 0.00, 'f2_std_or': 0.25, 'f1_std_or': 0.75},
    'f1_std_or': {'f4_std_or': 0.75, 'f3_std_or': 0.00, 'f2_std_or': 0.00, 'f1_std_or': 0.25}
}

# Transition matrix for random sequence
trans_matrix_rndm = {
    'f4_std_rndm': {'f4_std_rndm': 0.25, 'f3_std_rndm': 0.25, 'f2_std_rndm': 0.25, 'f1_std_rndm': 0.25},
    'f3_std_rndm': {'f4_std_rndm': 0.25, 'f3_std_rndm': 0.25, 'f2_std_rndm': 0.25, 'f1_std_rndm': 0.25},
    'f2_std_rndm': {'f4_std_rndm': 0.25, 'f3_std_rndm': 0.25, 'f2_std_rndm': 0.25, 'f1_std_rndm': 0.25},
    'f1_std_rndm': {'f4_std_rndm': 0.25, 'f3_std_rndm': 0.25, 'f2_std_rndm': 0.25, 'f1_std_rndm': 0.25}
}

# Transition matrix for ordered sequence (custom frequencies)
trans_matrix_or_tin = {
    'f4_tin_or': {'f4_tin_or': 0.25, 'f3_tin_or': 0.75, 'f2_tin_or': 0.00, 'f1_tin_or': 0.00},
    'f3_tin_or': {'f4_tin_or': 0.00, 'f3_tin_or': 0.25, 'f2_tin_or': 0.75, 'f1_tin_or': 0.00},
    'f2_tin_or': {'f4_tin_or': 0.00, 'f3_tin_or': 0.00, 'f2_tin_or': 0.25, 'f1_tin_or': 0.75},
    'f1_tin_or': {'f4_tin_or': 0.75, 'f3_tin_or': 0.00, 'f2_tin_or': 0.00, 'f1_tin_or': 0.25}
}

# Transition matrix for random sequence (custom frequencies)
trans_matrix_rndm_tin = {
    'f4_tin_rndm': {'f4_tin_rndm': 0.25, 'f3_tin_rndm': 0.25, 'f2_tin_rndm': 0.25, 'f1_tin_rndm': 0.25},
    'f3_tin_rndm': {'f4_tin_rndm': 0.25, 'f3_tin_rndm': 0.25, 'f2_tin_rndm': 0.25, 'f1_tin_rndm': 0.25},
    'f2_tin_rndm': {'f4_tin_rndm': 0.25, 'f3_tin_rndm': 0.25, 'f2_tin_rndm': 0.25, 'f1_tin_rndm': 0.25},
    'f1_tin_rndm': {'f4_tin_rndm': 0.25, 'f3_tin_rndm': 0.25, 'f2_tin_rndm': 0.25, 'f1_tin_rndm': 0.25}
}

# Define the tones for both standard and custom frequency sequences
frequencies = [
    "f1_std_or", "f2_std_or", "f3_std_or", "f4_std_or",
    "f1_std_rndm", "f2_std_rndm", "f3_std_rndm", "f4_std_rndm",
    "f1_tin_or", "f2_tin_or", "f3_tin_or", "f4_tin_or",
    "f1_tin_rndm", "f2_tin_rndm", "f3_tin_rndm", "f4_tin_rndm"
]

# Function to generate a sequence based on the transition matrix using MCMC
def generate_sequence_mcmc(start_freq='f4_std_or', length=1500, trans_matrix=trans_matrix_or):
    sequence = [start_freq]
    for _ in range(length - 1):
        current_freq = sequence[-1]
        next_freq = np.random.choice(list(trans_matrix[current_freq].keys()), p=list(trans_matrix[current_freq].values()))
        sequence.append(next_freq)
    return sequence

# Function to count transitions in a sequence
def count_transitions(sequence):
    transition_counts = {f: {t: 0 for t in frequencies} for f in frequencies}
    for i in range(len(sequence) - 1):
        current_freq = sequence[i]
        next_freq = sequence[i + 1]
        transition_counts[current_freq][next_freq] += 1
    return transition_counts

# Function to calculate transition probabilities from counts
def calculate_transition_probabilities(transition_counts):
    transition_probabilities = {f: {t: 0 for t in frequencies} for f in frequencies}
    for current_freq, next_freqs in transition_counts.items():
        total_transitions = sum(next_freqs.values())
        if total_transitions == 0:
            transition_probabilities[current_freq] = {f: 0 for f in next_freqs.keys()}
        else:
            transition_probabilities[current_freq] = {f: count / total_transitions for f, count in next_freqs.items()}
    return transition_probabilities

# Function to calculate differences between observed and expected transition probabilities
def calculate_differences(observed, expected):
    differences = {}
    for current_freq in frequencies:
        differences[current_freq] = {f: observed.get(current_freq, {}).get(f, 0) - expected.get(current_freq, {}).get(f, 0) for f in frequencies}
    return differences

# Function to calculate the total error from differences
def calculate_total_error(differences):
    return sum(abs(diff) for diffs in differences.values() for diff in diffs.values())

# Function to optimize the sequence to match the original transition matrix
def optimize_sequence(start_freq='f4_std_or', length=1500, iterations=100, trans_matrix=trans_matrix_or):
    best_sequence = generate_sequence_mcmc(start_freq, length, trans_matrix)
    best_error = float('inf')

    for i in range(iterations):
        new_sequence = generate_sequence_mcmc(start_freq, length, trans_matrix)
        
        # Split the sequence into three blocks of 500 trials each
        sequence1 = new_sequence[:500]
        sequence2 = new_sequence[500:1000]
        sequence3 = new_sequence[1000:1500]

        # Calculate transition probabilities and differences for each block
        transition_counts1 = count_transitions(sequence1)
        transition_counts2 = count_transitions(sequence2)
        transition_counts3 = count_transitions(sequence3)
        
        transition_probabilities1 = calculate_transition_probabilities(transition_counts1)
        transition_probabilities2 = calculate_transition_probabilities(transition_counts2)
        transition_probabilities3 = calculate_transition_probabilities(transition_counts3)
        
        differences1 = calculate_differences(transition_probabilities1, trans_matrix)
        differences2 = calculate_differences(transition_probabilities2, trans_matrix)
        differences3 = calculate_differences(transition_probabilities3, trans_matrix)
        
        error1 = calculate_total_error(differences1)
        error2 = calculate_total_error(differences2)
        error3 = calculate_total_error(differences3)
        
        total_error = error1 + error2 + error3

        if total_error < best_error:
            best_sequence = new_sequence
            best_error = total_error

    print(f"Final Block Errors: {error1}, {error2}, {error3}, Total Error: {total_error}")
    return best_sequence

# Generate and optimize sequences for standard and custom frequencies
sequence_or_std = optimize_sequence(start_freq='f4_std_or', length=1500, iterations=100, trans_matrix=trans_matrix_or)
sequence_rndm_std = optimize_sequence(start_freq='f4_std_rndm', length=1500, iterations=100, trans_matrix=trans_matrix_rndm)

# CHANGED: Generate truly new sequences for custom frequencies
sequence_or_tin = optimize_sequence(start_freq='f4_tin_or', length=1500, iterations=100, trans_matrix=trans_matrix_or_tin)
sequence_rndm_tin = optimize_sequence(start_freq='f4_tin_rndm', length=1500, iterations=100, trans_matrix=trans_matrix_rndm_tin)

# Split the optimized sequences into three blocks of 500 trials each
standard_orderly_blocks = [sequence_or_std[i:i+500] for i in range(0, 1500, 500)]
custom_orderly_blocks = [sequence_or_tin[i:i+500] for i in range(0, 1500, 500)]  # CHANGED: Added custom orderly blocks
standard_random_blocks = [sequence_rndm_std[i:i+500] for i in range(0, 1500, 500)]
custom_random_blocks = [sequence_rndm_tin[i:i+500] for i in range(0, 1500, 500)]  # CHANGED: Added custom random blocks

# Combine blocks
# CHANGED: Updated experiment blocks to include custom sequences
experiment_blocks = [
    ("Standard Orderly 1", standard_orderly_blocks[0]),
    ("Standard Random 1", standard_random_blocks[0]),
    ("Standard Orderly 2", standard_orderly_blocks[1]),
    ("Standard Random 2", standard_random_blocks[1]),
    ("Standard Orderly 3", standard_orderly_blocks[2]),
    ("Standard Random 3", standard_random_blocks[2]),
    ("Custom Orderly 1", custom_orderly_blocks[0]),
    ("Custom Random 1", custom_random_blocks[0]),
    ("Custom Orderly 2", custom_orderly_blocks[1]),
    ("Custom Random 2", custom_random_blocks[1]),
    ("Custom Orderly 3", custom_orderly_blocks[2]),
    ("Custom Random 3", custom_random_blocks[2])
]

# Mapping of names to triggers
trigger_dict = {
    "f1_std_or": 1, "f2_std_or": 2, "f3_std_or": 3, "f4_std_or": 4,
    "f1_std_rndm": 5, "f2_std_rndm": 6, "f3_std_rndm": 7, "f4_std_rndm": 8,
    "f1_tin_or": 11, "f2_tin_or": 12, "f3_tin_or": 13, "f4_tin_or": 14,
    "f1_tin_rndm": 15, "f2_tin_rndm": 16, "f3_tin_rndm": 17, "f4_tin_rndm": 18
}

log_dir = "D:\\ANTINOMICS\\data\\log"


# Generate log filename with study ID and regularity
log_filename = f"{log_dir}\\{study_id}_regularity_log.json"

# Save experiment log (replace existing save code)
with open(log_filename, "w") as f:
    json.dump({
        "study_id": study_id,
        "f0": f0,
        "blocks": experiment_blocks, 
        "triggers": trigger_dict
    }, f, indent=4)

print(f"Experiment setup complete. Sequences saved. Approximate standard f0: {f0_standard:.2f} Hz")

# Function to plot sequences for all experimental blocks
def plot_experiment_blocks(experiment_blocks, frequency_map):
    plt.figure(figsize=(12, 8))
    for i, (block_name, sequence) in enumerate(experiment_blocks):
        frequencies = [frequency_map[freq_label] for freq_label in sequence]
        plt.plot(range(len(frequencies)), frequencies, marker='o', label=block_name)
    plt.xlabel("Trial")
    plt.ylabel("Frequency (Hz)")
    plt.title("Sequences for Experimental Blocks")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Mapping of frequency labels to actual frequencies
frequency_map = {
    "f1_std_or": sfrq[0], "f2_std_or": sfrq[1], "f3_std_or": sfrq[2], "f4_std_or": sfrq[3],
    "f1_std_rndm": sfrq[0], "f2_std_rndm": sfrq[1], "f3_std_rndm": sfrq[2], "f4_std_rndm": sfrq[3],
    "f1_tin_or": cfrq[0], "f2_tin_or": cfrq[1], "f3_tin_or": cfrq[2], "f4_tin_or": cfrq[3],
    "f1_tin_rndm": cfrq[0], "f2_tin_rndm": cfrq[1], "f3_tin_rndm": cfrq[2], "f4_tin_rndm": cfrq[3],
}

# Call the function to plot the sequences before the experiment starts
plot_experiment_blocks(experiment_blocks, frequency_map)

# Create a PsychoPy window (this happens after the plot is shown)
win = visual.Window(
    size=(1920, 1080),
    screen=1,  # Use second display
    color='black',
    fullscr=True,
    monitor="testMonitor",
    units='pix'
)

# Function to generate a sinusoidal tone with ramps
def generate_tone(frequency, duration, ramp=0.005, sampling_rate=44100):
    t = np.linspace(0, duration, int(sampling_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    ramp_samples = int(sampling_rate * ramp)
    
    # Apply Hann window for smooth start and end
    hann_window = np.hanning(2 * ramp_samples)
    ramp_up = hann_window[:ramp_samples]
    ramp_down = hann_window[ramp_samples:]
    
    ramp_window = np.ones_like(wave)
    ramp_window[:ramp_samples] = ramp_up
    ramp_window[-ramp_samples:] = ramp_down
    
    return wave * ramp_window

# Generate PsychoPy sound objects for standard frequencies
sounds_standard = {f: sound.Sound(value=generate_tone(f, 0.1, 0.005), sampleRate=44100) for f in sfrq}

# Generate PsychoPy sound objects for custom frequencies
sounds_custom = {f: sound.Sound(value=generate_tone(f, 0.1, 0.005), sampleRate=44100) for f in cfrq}

# Generate PsychoPy sound objects for standard frequencies
amplitude_factor = db_to_amplitude(100)  # 100dB -> 75dB (25dB attenuation
sounds_standard = {
    f: sound.Sound(value=generate_tone(f, 0.1, 0.005) * amplitude_factor,
                 sampleRate=44100)
    for f in sfrq
}

# Generate PsychoPy sound objects for custom frequencies
sounds_custom = {
    f: sound.Sound(value=generate_tone(f, 0.1, 0.005) * amplitude_factor,
                 sampleRate=44100)
    for f in cfrq
}


def show_short_break():
    break_text = visual.TextStim(win, text="Short break. Please wait...")
    break_text.draw()
    win.flip()
    core.wait(10)

def show_long_break():
    break_text = visual.TextStim(win, text="Long break. Press spacebar to continue.")
    break_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])


show_start_instructions()


# Function to play a sequence of sounds with key press to jump to the next sound
def play_sequence(sequence, sounds_standard, sounds_custom, stim_interval=0.327):
    for freq_label in sequence:
        display_dot()
        trigger_number = trigger_dict[freq_label]
        send_trigger(trigger_number)  # Commented out as hardware is not connected
        print(f"Trigger {trigger_number} sent for {freq_label}")
        if "std" in freq_label:
            freq_index = int(freq_label[1]) - 1
            freq = sfrq[freq_index]
            sounds_standard[freq].play()
        else:  # CHANGED: Ensure custom frequencies are played
            freq_index = int(freq_label[1]) - 1
            freq = cfrq[freq_index]
            sounds_custom[freq].play()
        core.wait(0.1)  # Duration of the sound
        keys = event.getKeys()
        if 'n' in keys:
            print("Skipping to next sound...")
            return False
        core.wait(0.227)  # Remaining interval time (adjusted to make total 0.327)
    return True


# Update display_dot function with fixation cross styling
def display_dot():
    fixation = visual.TextStim(win, text="•", color='white', pos=(0, 0), height=100)
    fixation.draw()
    win.flip()

# Update start instructions for fullscreen
def show_start_instructions():
    instructions = visual.TextStim(win, 
        text="Start experiment with spacebar when ready.\n\nFocus on the white dot during trials.",
        color='white',
        height=40,
        wrapWidth=1000
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

#  Break functions for fullscreen
def show_short_break():
    break_text = visual.TextStim(win, 
        text="Short break\n\nPlease wait...",
        color='white',
        height=60,
        pos=(0, 0)
    )
    break_text.draw()
    win.flip()
    core.wait(10)

def show_long_break():
    break_text = visual.TextStim(win,
        text="Long break\n\nPress spacebar to continue",
        color='white',
        height=60,
        pos=(0, 0)
    )
    break_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

# Update end message
end_text = visual.TextStim(win,
    text="Experiment complete\n\nThank you!",
    color='white',
    height=60,
    pos=(0, 0)
)

# Play sequences
block_count = 0
total_blocks = len(experiment_blocks)

for name, sequence in experiment_blocks:
    print(f"Playing {name}...")
    if not play_sequence(sequence, sounds_standard, sounds_custom):
        print("Skipping to next set...")
        continue
    print(f"{name} playback complete.")
    
    block_count += 1
    
    if block_count % 3 == 0 and block_count < total_blocks:
        show_long_break()
    elif block_count < total_blocks:
        show_short_break()

# End of experiment
end_text = visual.TextStim(win, text="Experiment complete. Thank you!")
end_text.draw()
win.flip()
core.wait(3)
win.close()
core.quit()