import os
import subprocess
import time

# Directory where your YAML files are located
yaml_dir = 'data/tests/epicenter_energy_tests'

# List all YAML files in the directory
yaml_files = [f for f in os.listdir(yaml_dir) if f.endswith('.yaml')]
#train-distance
#train-backazimuth
#train-costero
#train-multitask
#test-distance

# Loop through each YAML file and run the command
for yaml_file in yaml_files:
    yaml_path = os.path.join(yaml_dir, yaml_file)
    command = f'pdm train-distance --test_file={yaml_path} --extract_features'

    print(f'Running command: {command}')
    time.sleep(2)
    # Execute the command
    result = subprocess.run(command, shell=True)
    
