import argparse
import gc

import raw_data_processing
import wisdm_processor
import hhar_processor
import motionsense_processor
import numpy as np
import os
import pickle


strings = {
    'device': ['phone'],
    'modality': ['accel', 'gyro'],
    'exclude': ['.zip', '.DS_Store', '__MACOSX', '.pdf', '.txt'],
    'include': ['Phones_accelerometer.csv', 'Phones_gyroscope' ],
    'process_funcs': [raw_data_processing.process_hhar, raw_data_processing.process_wisdm]

}
def get_parser():
   parser = argparse.ArgumentParser(description='Preprocess all data')
   parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset directory')
   return parser

def process_dataset(dataset, output_dir, first_run = False, seen_goal = False):
    ds = dataset[1]
    print(os.getcwd())
    #create a folder to hold the modalities for each dataset
    if first_run and not os.path.exists(os.path.join(output_dir, ds)):
        os.mkdir(os.path.join(output_dir,ds))

    if ds == 'wisdm':
        # wisdm processing
        #get all the modalities
        labels = raw_data_processing.get_labels_from_file('../test_run/original_datasets/wisdm/wisdm/wisdm-dataset/activity_key.txt')
        for modality in os.listdir(dataset[0]):
            if modality in strings['exclude']:
                continue
            print(modality)
            if modality == 'accel':
                #process the accelerometer data
                print('accel')
                user_datasets = raw_data_processing.process_wisdm(os.path.join(dataset[0],modality), labels)
                #save the data
                pickle.dump(user_datasets, open(os.path.join(output_dir, 'wisdm', 'accel.pkl'), 'wb'))
            elif modality == 'gyro':
                print('gyro')
                #process the gyroscope data
                user_datasets = raw_data_processing.process_wisdm(os.path.join(dataset[0],modality), labels)
                #save the data
                pickle.dump(user_datasets, open(os.path.join(output_dir, 'wisdm', 'gyro.pkl'), 'wb'))
    elif ds == 'hhar':
        for modality in os.listdir(dataset[0]):
            # if the modality does not contain a string in the device list then skip it
            if not any([device in modality.lower() for device in strings['device']]):
                continue
            #now that you have the device, get the modalities
            # now you will process each modality by going through the include list
            res = list(filter(lambda x: x in modality.lower(), strings['modality']))
            #process the input and pickle it to the correct location
            user_dataset = raw_data_processing.process_hhar(dataset[0], modality)

            pickle.dump(user_dataset, open(os.path.join(output_dir, 'hhar', modality + '.pkl'), 'wb'))
            print(res)

        print('testing')

    gc.collect()


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
    output_dir = "../test_run/processed_datasets"
    data_dir = args.data_dir
    folders = [
        # ('../test_run/original_datasets/wisdm/wisdm/wisdm-dataset/raw/phone', 'wisdm'),
               ('../test_run/original_datasets/hhar/Activity recognition exp', 'hhar')]
    for f in folders:
        process_dataset(f, output_dir, True, False)

    print("Preprocessing data...")




if __name__ == "__main__":
    main()