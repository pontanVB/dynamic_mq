import argparse
import random
from pathlib import Path

# Argument parser for -i and -o
parser = argparse.ArgumentParser(description='Convert METIS to DIMACS format with random weights.')
parser.add_argument('-i', '--input', required=True, help='Path to input .metis file')
parser.add_argument('-o', '--output_dir', required=True, help='Output directory for .gr file')
args = parser.parse_args()

input_file = args.input
output_dir = args.output_dir

# Read input METIS file
with open(input_file, 'r') as f:
    first_line = f.readline().strip()
    num_nodes, _, _ = map(int, first_line.split())

    edges = set()

    for i, line in enumerate(f, 1):
        neighbors = map(int, line.strip().split())
        for neighbor in neighbors:
            u, v = min(i, neighbor), max(i, neighbor)
            if u != v:
                edges.add((u, v))

# Add random weights between 1 and 256
edges_with_weights = [(u, v, random.randint(1, 256)) for u, v in edges]

# Prepare output path
output_path = Path(output_dir) / (Path(input_file).stem + '.gr')

# Write DIMACS output
with open(output_path, 'w') as out:
    out.write('c Converted from METIS format\n')
    out.write(f'p sp {num_nodes} {2 * len(edges)}\n')
    for u, v, w in edges_with_weights:
        out.write(f'a {u} {v} {w}\n')
        out.write(f'a {v} {u} {w}\n')

print(f"Written to {output_path}")