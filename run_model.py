#!/usr/bin/env python

# Please do *not* edit this script. We will discard any changes so that we can run the trained models consistently.

# This file contains functions for running your model for the Challenge. You can run it as follows:
#
#   python run_model.py -d data -m model -o outputs -v
#
# where 'data' is a folder containing the Challenge data, 'model' is a folder containing your trained model, 'outputs' is a folder
#  for saving your model's outputs, and -v is an optional verbosity flag.

import argparse
import os
import sys

from helper_code import *
from team_code import load_model, run_model

# Parse arguments.
def get_parser():
    description = 'Run the trained Challenge models.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--allow_failures', action='store_true')
    return parser

# Run the code.
def run(args):
    # Load the models.
    if args.verbose:
        print('Loading the Challenge model...')

    # You can use these functions to perform tasks, such as loading your model, that you only need to perform once.
    model = load_model(args.model_folder, args.verbose) ### Teams: Implement this function!!!

    # Find the Challenge data.
    if args.verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(args.data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise Exception('No data were provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(args.output_folder, exist_ok=True)

    # Run the team's model on the Challenge data.
    if args.verbose:
        print('Running the Challenge model on the Challenge data...')

    # Initialize a dictionary to hold all results.
    results = {}

    # Iterate over the patients.
    for i in range(num_records):
        record = patient_metadata_list[i]
        patient_id = record[HEADERS['bids_folder']]
        site_id    = record[HEADERS['site_id']]
        session_id = record[HEADERS['session_id']]
        
        if args.verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {patient_id} (Session {session_id})...')

        # Allow or disallow the model to fail on parts of the data; this can be helpful for debugging.
        try:
            binary_output, probability_output = run_model(model, record, args.data_folder, args.verbose) ### Teams: Implement this function!!!
            assert(is_boolean(binary_output) or is_nan(binary_output))
            assert(is_number(probability_output))
        except:
            if args.allow_failures:
                if args.verbose:
                    print('... failed.')
                binary_output, probability_output = float('nan'), float('nan')
            else:
                raise

        # Store the results.
        results[patient_id] = (binary_output, probability_output)

    # Update the demographics table with the model outputs.
    if args.verbose:
        print('Updating demographics table with model outputs...')
    
    patient_data_file = os.path.join(args.data_folder, DEMOGRAPHICS_FILE)
    output_table_path = update_demographics_table(patient_data_file, args.output_folder, results)

    if args.verbose:
        print(f'Results saved to: {output_table_path}')
        print('Done.')

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
