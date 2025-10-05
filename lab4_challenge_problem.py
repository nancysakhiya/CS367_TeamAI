import random
import math
import pygame
import numpy as np

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Define the notes of Raag Bhairav in Indian notation
raag_bhairav = ['Sa', 'Re (komal)', 'Ga', 'Ma', 'Pa', 'Dha (komal)', 'Ni', 'Sa`']

# Define frequencies for each note (in Hz)
note_frequencies = {
    'Sa': 261.63,    # C (Middle C)
    'Re (komal)': 277.18, # Db
    'Ga': 329.63,    # E
    'Ma': 349.23,    # F
    'Pa': 392.00,    # G
    'Dha (komal)': 415.30, # Ab
    'Ni': 493.88,    # B
    'Sa`': 523.25  # Higher octave C (Double frequency of Sa)
}

def generate_initial_melody(length):
    # Generate a melody starting and ending with 'Sa' and 'Sa`'
    melody = [raag_bhairav[0]]  # Start with 'Sa'
    for _ in range(1, length - 1):
        melody.append(random.choice(raag_bhairav[1:-1]))  # Avoid first and last Sa
    melody.append('Sa`')  # End with higher 'Sa'
    return melody

def calculate_energy(melody):
    energy = 0
    
    # Penalize for not starting or ending with Sa and Sa`
    if melody[0] != 'Sa' or melody[-1] != 'Sa`':
        energy += 10
    
    # Penalize for not emphasizing Ga and Ni
    if 'Ga' not in melody or 'Ni' not in melody:
        energy += 15
    
    # Penalize for not having enough variety
    if len(set(melody)) < 5:
        energy += 10
    
    # Penalize for repeated notes
    for i in range(len(melody) - 1):
        if melody[i] == melody[i+1]:
            energy += 2
    
    return energy

def get_neighbor(melody):
    new_melody = melody.copy()
    # Choose a random index except the first and last (since those are fixed as Sa and Sa`)
    index = random.randint(1, len(melody) - 2)
    new_melody[index] = random.choice(raag_bhairav[1:-1])  # Avoid first and last Sa
    return new_melody

def simulated_annealing(length, initial_temp, cooling_rate, num_iterations):
    current_melody = generate_initial_melody(length)
    current_energy = calculate_energy(current_melody)
    best_melody = current_melody
    best_energy = current_energy
    temperature = initial_temp

    for _ in range(num_iterations):
        neighbor = get_neighbor(current_melody)
        neighbor_energy = calculate_energy(neighbor)
        
        if neighbor_energy < current_energy or random.random() < math.exp((current_energy - neighbor_energy) / temperature):
            current_melody = neighbor
            current_energy = neighbor_energy
        
        if current_energy < best_energy:
            best_melody = current_melody
            best_energy = current_energy
        
        temperature *= cooling_rate

    return best_melody

def play_note(note, duration=0.5):
    frequency = note_frequencies[note]
    sample_rate = 44100
    samples = int(duration * sample_rate)
    
    t = np.linspace(0, duration, samples, False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Apply a simple envelope
    envelope = np.linspace(1, 0, samples)**2
    tone = (tone * envelope * 32767).astype(np.int16)
    
    # Create a stereo sound by duplicating the mono channel
    stereo_sound = np.column_stack((tone, tone))
    
    sound = pygame.sndarray.make_sound(stereo_sound)
    sound.play()
    pygame.time.wait(int(duration * 1000))

def play_melody(melody):
    for note in melody:
        play_note(note)
        pygame.time.wait(100)  # Short pause between notes

# Generate a melody with 8 notes
melody = simulated_annealing(length=8, initial_temp=100, cooling_rate=0.995, num_iterations=10000)

# Print Aaroh (original melody)
print("Generated Raag Bhairav Melody (Aaroh, ascending order):")
print(' '.join(melody))

# Print Avroah (reversed melody)
avroah = melody[::-1]
print("Generated Raag Bhairav Melody (Avroah, descending order):")
print(' '.join(avroah))

# Play the Aaroh (ascending melody)
play_melody(melody)

# Play the Avroah (descending melody)
play_melody(avroah)

pygame.quit()
