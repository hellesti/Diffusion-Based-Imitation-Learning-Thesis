import click
import pathlib
import numpy as np

@click.command()
@click.option('--eval_folder', '-e', required=True, help='Path to the eval folder')
def calculate_average_latency(eval_folder):
    eval_path = pathlib.Path(eval_folder)
    latency_file = eval_path / 'videos' / 'inference_latencies.txt'
    
    if not latency_file.exists():
        print(f"Latency file not found at {latency_file}")
        return
    
    latencies = []
    with open(latency_file, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            if len(parts) == 2:
                latency = float(parts[1].split(': ')[1])
                latencies.append(latency)
    
    if latencies:
        average_latency = np.mean(latencies)
        print(f"Average inference latency: {average_latency:.6f} seconds")
    else:
        print("No valid latency data found in the file.")

if __name__ == '__main__':
    calculate_average_latency()