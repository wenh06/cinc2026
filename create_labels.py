#!/usr/bin/env python

# Do *not* edit this script. 
# This file contains functions for creating labels for the Challenge 2026. You can run it as follows:
#
#   python create_labels.py -i demographics.csv -c icd_codes_CI.csv -o demographics_with_CI.csv
#
# where 'demographics.csv' is a CSV file containing the subjects' demographic information, 'icd_codes_CI.csv' is a CSV file
# containing ICD-9 and/or ICD-10 codes related to cognitive impairment diagnoses, amd 'demographics_with_CI.csv' is CSV file 
# containing the subjects' demographic information and additional columns, 'Cognitive_Impairment' and 'Time_to_Event', that describe
# a positive or negative label and, for positive subjects, the number of days from the date of the PSG to the date of the first ICD
# code for a cognitive impairment diagnosis.

import argparse
import numpy as np
import pandas as pd
import sys
from datetime import timedelta

id_patients = 'BDSPPatientID'
id_site = 'SiteID'
id_labels = 'Cognitive_Impairment'
id_time_to_event = 'Time_to_Event'
lower_bound = 1 # Lower bound of interval for CI diagnosis in years
upper_bound = 6 # Upper bound of interval for CI diagnosis in years

# Parse arguments.
def get_parser():
    description = 'Create cognitive impairment labels'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_demographics_file', type=str, required=True)
    parser.add_argument('-c', '--icd_file', type=str, required=True)
    parser.add_argument('-o', '--output_demographics_file', type=str, required=True)
    return parser


# Create labels for the models.
def create_labels(demographics_file, icd_file, output_demographics_file):
    # Read demographics
    demographics = pd.read_csv(demographics_file)
    
    # Convert CreationTime to datetime variable
    demographics['CreationTime'] = pd.to_datetime(demographics['CreationTime'], format = 'mixed')

    # Read ICD codes for CI (cognitive impairment)    
    ci = pd.read_csv(icd_file)

    # Create column combining SiteID and BDSPPatientID
    ci['SiteIDBDSPPatientID'] = ci['SiteID'] + ci['BDSPPatientID'].astype(str)
    demographics['SiteIDBDSPPatientID'] = demographics['SiteID'] + demographics['BDSPPatientID'].astype(str)

    # Filter ICD codes subject-wise, include first ICD date, last ICD date, number of ICDs and list of ICDs
    ci['ICDDate'] = pd.to_datetime(ci['ICDDate'], format = 'mixed')
    ci = (ci.groupby('SiteIDBDSPPatientID').agg(
            BDSPPatientID = ('BDSPPatientID', 'first'),
            SiteID = ('SiteID', 'first'),
            ICDDateFirst=('ICDDate', 'min'),
            ICDDateLast=('ICDDate', 'max'),
            ICDCount=('ICDDate', 'count'),
            ICD9 =('ICD9', lambda x: x.dropna().tolist()),
            ICD10 =('ICD10', lambda x: x.dropna().tolist())
        ).reset_index())

    # Merge with demographics
    demographics = pd.merge(demographics, ci[['SiteIDBDSPPatientID','ICDDateFirst', 'ICDDateLast', 'ICDCount']], on = 'SiteIDBDSPPatientID', how = 'left')

    # Calculate time difference between last and first ICD code (condition: at least 7 days)
    demographics['ICD_difference'] = (demographics['ICDDateLast'] - demographics['ICDDateFirst']).apply(lambda x: x.days)
    # Calculate time difference between first ICD code and PSG date
    demographics['Time_to_Event'] = (demographics['ICDDateFirst'] - demographics['CreationTime']).apply(lambda x: x.days)

    # Create Cognitive_Impairment column
    demographics['Cognitive_Impairment'] = False

    # Find which subjects have CI diagnosed in the given interval after the PSG and have at least 7 days between first and last ICD code
    indexes_positive = (
        (demographics['ICDDateLast'] - demographics['ICDDateFirst'] >= timedelta(days=7))
        &
        (demographics['ICDDateFirst'] - demographics['CreationTime'] >= timedelta(days=365.25 * lower_bound))
        &
        (demographics['ICDDateFirst'] - demographics['CreationTime'] <= timedelta(days=365.25 * upper_bound))
    )
    demographics.loc[indexes_positive, 'Cognitive_Impairment'] = True

    # Drop unnecessary columns    
    demographics = demographics.drop(columns = {'SiteIDBDSPPatientID','ICD_difference', 'ICDDateFirst', 'ICDDateLast', 'ICDCount'})

    # Store demographics
    demographics.to_csv(output_demographics_file, index = False)

# Run the code.
def run(args):
    # Create labels
    create_labels(args.input_demographics_file, args.icd_file, args.output_demographics_file)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
