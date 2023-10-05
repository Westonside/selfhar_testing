import argparse
import wisdm_processor
import hhar_processor
import motionsense_processor
import numpy as np
import os
dataset_processors = {
    "wisdm":  wisdm_processor.wisdm_process,
    "hhar": hhar_processor.hhar_process,
    "motionsense": motionsense_processor.motionsense_process,
}

def get_parser():
   parser = argparse.ArgumentParser(description='Preprocess all data')
   parser.add_argument('--data_dir', type=str, default='./data', help='path to data directory')
   return parser

def process_modality(modality_dir, output_dir):
    pass
def get_modalities(device_dir) -> list:
    modalities = []
    #get all the folders in the directory
    for folder in os.listdir(device_dir):
        print(folder) # get the folders that will correspond with devices
        # for each device there will be a number of modalities



def main():
    # TODO: complete processing of modalities and devices
    parser = get_parser()
    args = parser.parse_args()
    output_dir = "./test_run"
    data_dir = args.data_dir
    print("Preprocessing data...")




if __name__ == "__main__":
    main()