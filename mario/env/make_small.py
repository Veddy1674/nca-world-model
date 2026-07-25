from PIL import Image
import numpy as np

WHOLE_MAP = "mario/env/map3376x240.png" # input
SMALL_MAP = "mario/env/map211x15.png" # output
COLOR_MAP = "mario/env/color_map.npz" # output

# HWC - uint8
map = np.array(Image.open(WHOLE_MAP).convert("RGB"), dtype=np.uint8)

H_tiles, W_tiles = map.shape[0] // 16, map.shape[1] // 16
iterations = W_tiles * H_tiles
print(f"Map is {map.shape[1]}x{map.shape[0]} ({iterations:,} iterations)...")

small_map = np.empty((H_tiles, W_tiles, 3), dtype=np.uint8)

# Dizionari per tracciare lo stato globale
tile_to_color = {}    # {tile_hash: mean_color_numpy}
color_to_sprite = {}  # {(R, G, B): tile_numpy 16x16x3}
used_mean_hashes = set()

def get_mean_color(tile: np.ndarray):
    tile_hash = hash(tile.tobytes())
    
    # 1. Se abbiamo già processato QUESTO esatto tile graficamente, restituisci il colore già assegnato
    if tile_hash in tile_to_color:
        return tile_to_color[tile_hash]
        
    # 2. Altrimenti calcola la media per il nuovo tile
    mean = tile.mean(axis=(0, 1)).astype(np.uint8)
    mean_hash = hash(mean.tobytes())
    
    # 3. Se il colore medio è già usato da un ALTRO tile, modificalo finché non è libero
    if mean_hash in used_mean_hashes:
        print("NOTE: Collisione rilevata! Colore medio già in uso da un altro tile, modifico...")
        mean_int = mean.astype(int)
        while True:
            mean_int = (mean_int + 1) % 256
            mean_hash = hash(mean_int.astype(np.uint8).tobytes())
            if mean_hash not in used_mean_hashes:
                break
        mean = mean_int.astype(np.uint8)
        
    # 4. Registra il colore come usato e associalo a questo tile
    used_mean_hashes.add(mean_hash)
    tile_to_color[tile_hash] = mean
    color_to_sprite[tuple(mean)] = tile
    
    return mean

# iterate to each tile (step by 16), 'iterations' times
for y in range(0, map.shape[0], 16):
    for x in range(0, map.shape[1], 16):
        tile = map[y:y+16, x:x+16] # 16x16 numpy

        # mean color is the unique identifier
        mean_color = get_mean_color(tile)

        # set each pixel in small_map
        small_map[y // 16, x // 16, :] = mean_color

print("Done, saving...")

Image.fromarray(small_map).save(SMALL_MAP)
print(f"Saved small map as '{SMALL_MAP}' as shape: {small_map.shape}")

# create color map:
colors = np.unique(small_map.reshape(-1, 3), axis=0)
sprites = np.array([color_to_sprite[tuple(c)] for c in colors], dtype=np.uint8) # allinea

np.savez(COLOR_MAP, colors=colors, sprites=sprites)

num_colors = colors.shape[0]
print(f"Saved {num_colors} unique colors in '{COLOR_MAP}' as shape: {colors.shape}")
print(f"Each {small_map.shape[0]}x{small_map.shape[0]} map slice as one-hot float32 will be ~{(4 * num_colors * small_map.shape[0] * small_map.shape[0]) / 1000:.2f}KB")