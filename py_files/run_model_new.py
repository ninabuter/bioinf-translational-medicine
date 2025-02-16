#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model_new.py -i unlabelled_sample.txt -m model.pkl -o output.txt

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

    # loading the validation data
    unlabelled_df = pd.read_table(args.input_file, sep='\t')

    # transforming the validation data
    df_call = unlabelled_df

    # Combine the first four columns of df_call to one column, save that column in column_names, and remove the first
    # five columns
    df_call["Chromosome_start_end_nclone"] = (df_call["Chromosome"].astype(str) + "-"
                                              + df_call["Start"].astype(str) + "-"
                                              + df_call["End"].astype(str) + "-"
                                              + df_call["Nclone"].astype(str))
    column_names = df_call["Chromosome_start_end_nclone"]
    df_call = df_call.drop(columns=["Chromosome", "Start", "End", "Nclone", "Chromosome_start_end_nclone"])

    # Transpose df_call, add the column_names, reset index, and rename "index" to "Sample"
    df_call_transposed = df_call.transpose()
    df_call_transposed.columns = column_names.tolist()
    df_call_transposed.reset_index(drop=False, inplace=True)
    df_call_transposed = df_call_transposed.rename(columns={"index": "Sample"})

    # Select features that are selected by the RFE_F method in the training set
    selected_features = ['6-135286400-135498058-20', '12-64727853-66012212-99', '12-70574871-71644041-87',
                         '12-72645832-73204852-27', '12-84542006-85443011-85', '12-85450052-85962613-38',
                         '17-35076296-35282086-29', '17-39280577-40847517-203', '17-41062669-41447005-37',
                         '17-41458670-41494331-6']

    # Keep only the selected features and the 'Sample' column
    df_call_transposed = df_call_transposed[['Sample'] + selected_features]
    test_samples = df_call_transposed['Sample']
    test_data = df_call_transposed.drop(['Sample'], axis=1)

    """
    df1 = unlabelled_df.T  # transposing the Train_call data
    meta_data = df1.head(4)
    df1 = df1.iloc[4:]
    df1 = df1.reset_index()
    df1 = df1.rename(columns={'index': 'Sample'})
    meta_data = meta_data.T

    meta_data['Features'] = meta_data[meta_data.columns[0:3]].apply(
        lambda x: '-'.join(x.dropna().astype(str)),
        axis=1
    )

    features = meta_data["Features"].tolist()

    column_names = features.copy()

    column_names.insert(0, "Sample")
    unlabelled_df = df1
    test_samples = unlabelled_df['Sample']
    test_data = unlabelled_df.drop(['Sample'], axis=1)
    """

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
