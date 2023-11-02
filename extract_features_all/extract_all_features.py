import gc
import multiprocessing as mp
import os
import time
import run_self_har
import subprocess
files = ['test_run/processed_datasets/hhar']
mapping = {
    'test_run/processed_datasets/hhar/Phones_gyroscope.csv.pkl': 'test_run/processed_datasets/hhar/Phones_gyroscope.csv.pkl',
    'test_run/processed_datasets/hhar/Phones_accelerometer.csv.pkl': 'test_run/processed_datasets/hhar/Phones_accelerometer.csv.pkl',
}


def extract_features(*args):
    print('Extracting Features from ', *args)
    try:

        command = [
            "python",  # Path to the Python interpreter
            "run_self_har.py",  # The name of the script you want to run
            "--working_directory", "test_run",
            "--config", "sample_configs/transformation_discrimination_ft_extract.json",
            "--labelled_dataset_path", args[1],
            "--unlabelled_dataset_path", args[0],
            "--window_size", "400",
            "--max_unlabelled_windows", "40000"
        ]
        print(command)
        subprocess.run(command)
    except Exception as e:
        print(e)



def print_command():

    for key in mapping.keys():

        command = [
            "python",  # Path to the Python interpreter
            "run_self_har.py",  # The name of the script you want to run
            "--working_directory", "test_run",
            "--config", "sample_configs/transformation_discrimination_ft_extract.json",
            "--labelled_dataset_path",key,
            "--unlabelled_dataset_path", key,
            "--window_size", "400",
            "--max_unlabelled_windows", "40000"
        ]
        print(' '.join(command))


def main():
    num_modal = 1
    print('Extracting Features from all datasets')
    with mp.Pool(num_modal) as pool:
        # now go through each item in the file and extract the features
        # pair the modalities with the other hhar/gyro with motionsense gyroscope
        for file in files:
           for modality in os.listdir(file):
               modality_file = os.path.join(file, modality)
               if modality_file in mapping:
                    print(mapping[modality_file])
                    extract_features(modality_file, mapping[modality_file])
                    # pool.aopp(extract_features, (modality_file, mapping[modality_file]))
                    gc.collect()

        pool.close()
        pool.join() # this will wait for the processes to finish







if __name__ == "__main__":
    print_command()