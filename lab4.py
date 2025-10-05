import numpy as np
import random
import matplotlib.pyplot as plt

# =========================
# Load the scrambled image from Octave text-based .mat
# =========================
filename = "scrambled_lena.mat"

with open(filename, 'r') as f:
    lines = f.readlines()

# Remove empty lines and strip spaces
lines = [line.strip() for line in lines if line.strip()]

# Find the line with dimensions (first line that has exactly two numbers)
for idx, line in enumerate(lines):
    if len(line.split()) == 2 and all(s.isdigit() for s in line.split()):
        rows, cols = map(int, line.split())
        dim_index = idx
        break

# All lines after the dimension line are pixel values
data_lines = lines[dim_index + 1:]

# Convert all remaining lines to integers
data_numbers = [int(line) for line in data_lines]

# Reshape into 2D array
image_data = np.array(data_numbers, dtype=np.uint8).reshape(rows, cols)

# =========================
# Downscale the image for faster computation
# =========================
scale_factor = 4  # reduce size (128x128)
image_data = image_data[::scale_factor, ::scale_factor]

# =========================
# Puzzle parameters
# =========================
TILE_COUNT = 9  # 3x3 puzzle
START_TEMP = 500
TEMP_DECAY = 0.95
ITERATION_LIMIT = 1000

# =========================
# Utility functions
# =========================
def show_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def cut_image_into_tiles(image, num_tiles):
    height, width = image.shape
    tile_height, tile_width = height // num_tiles, width // num_tiles
    return [image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
            for i in range(num_tiles) for j in range(num_tiles)]

def reconstruct_image(arrangement, pieces, num_tiles):
    tile_height, tile_width = pieces[0].shape
    reconstructed_image = np.zeros((tile_height * num_tiles, tile_width * num_tiles), dtype=pieces[0].dtype)
    for row in range(num_tiles):
        for col in range(num_tiles):
            reconstructed_image[row*tile_height:(row+1)*tile_height,
                                col*tile_width:(col+1)*tile_width] = pieces[arrangement[row*num_tiles + col]]
    return reconstructed_image

# =========================
# Vectorized fitness function
# =========================
def evaluate_puzzle_fitness(arrangement, pieces):
    """Compute fitness using only edge pixels (vectorized for speed)."""
    grid_size = int(np.sqrt(len(pieces)))
    fitness = 0

    # Precompute top, bottom, left, right edges
    top_edges = np.array([pieces[i][0, ::5] for i in range(len(pieces))])
    bottom_edges = np.array([pieces[i][-1, ::5] for i in range(len(pieces))])
    left_edges = np.array([pieces[i][::5, 0] for i in range(len(pieces))])
    right_edges = np.array([pieces[i][::5, -1] for i in range(len(pieces))])

    # Compare top-bottom edges
    for row in range(1, grid_size):
        for col in range(grid_size):
            curr = arrangement[row * grid_size + col]
            above = arrangement[(row-1) * grid_size + col]
            fitness += np.sum(np.abs(top_edges[curr] - bottom_edges[above]))

    # Compare left-right edges
    for row in range(grid_size):
        for col in range(1, grid_size):
            curr = arrangement[row * grid_size + col]
            left = arrangement[row * grid_size + (col-1)]
            fitness += np.sum(np.abs(left_edges[curr] - right_edges[left]))

    return fitness

def swap_random_tiles(arrangement):
    new_arr = arrangement.copy()
    a, b = random.sample(range(len(arrangement)), 2)
    new_arr[a], new_arr[b] = new_arr[b], new_arr[a]
    return new_arr

def optimize_puzzle_arrangement(pieces):
    current_state = list(range(len(pieces)))
    current_fitness = evaluate_puzzle_fitness(current_state, pieces)
    temperature = START_TEMP

    for step in range(ITERATION_LIMIT):
        new_state = swap_random_tiles(current_state)
        new_fitness = evaluate_puzzle_fitness(new_state, pieces)

        if new_fitness < current_fitness or random.random() < np.exp((current_fitness - new_fitness) / temperature):
            current_state = new_state
            current_fitness = new_fitness

        temperature *= TEMP_DECAY

        if step % 100 == 0:
            print(f"Step {step}, Fitness: {current_fitness}, Temp: {temperature}")

        if current_fitness == 0:
            break

    return current_state

# =========================
# Main
# =========================
if __name__ == "__main__":
    show_image(image_data, "Scrambled Puzzle")

    grid_size = int(np.sqrt(TILE_COUNT))
    puzzle_tiles = cut_image_into_tiles(image_data, grid_size)

    final_order = optimize_puzzle_arrangement(puzzle_tiles)
    solved_image = reconstruct_image(final_order, puzzle_tiles, grid_size)

    show_image(solved_image, "Solved Puzzle")
