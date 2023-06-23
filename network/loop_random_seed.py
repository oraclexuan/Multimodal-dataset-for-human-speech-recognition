import sys
import subprocess

start_seed = 0  
end_seed = 1   

for seed in range(start_seed, end_seed + 1):
    args = [sys.executable, r'F:\Scientific_data\radar_data\Python\Multimodal-dataset-for-human-speech-recognition-main\network\train2.py', '--seed', str(seed)]
    subprocess.run(args)