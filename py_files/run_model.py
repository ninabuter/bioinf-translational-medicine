#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys

# Start your coding

# import the library you need here
import pandas as pd
import pickle

# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding

    # Step 1: load the model from the model file
    with open(args.model_file, "rb") as file:
        model = pickle.load(file)
    unlabelled_df = pd.read_table(args.input_file, sep='\t')
    test_samples = unlabelled_df['Sample']
    test_data = unlabelled_df.drop(['Sample'], axis=1)
    # Step 2: apply the model to the input file to do the prediction
    predictions = model.predict(test_data)
    # Step 3: write the prediction into the designated output file
    predictions_df = pd.DataFrame({'Sample': test_samples, 'Subgroup': predictions})
    with open(args.output_file, "w") as outfile:
        outfile.write('"Sample"\t"Subgroup"\n')
        for index, row in predictions_df.iterrows():
            outfile.write(f'"{row["Sample"]}"\t"{row["Subgroup"]}"\n')

    # End your coding


if __name__ == '__main__':
    main()
