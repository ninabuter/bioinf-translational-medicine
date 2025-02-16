""""
This script transforms the original evaluation dataset into a format suitable for the prediction task.
"""

import pandas as pd

# Import arrayCGH copy number data and label data
df_call = pd.read_table("Validation_call.txt", sep="\t")

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

# Drop features that were selected by the RFE_F method in the training set
selected_features = ['6-135286400-135498058-20', '12-64727853-66012212-99', '12-70574871-71644041-87',
                     '12-72645832-73204852-27', '12-84542006-85443011-85', '12-85450052-85962613-38',
                     '17-35076296-35282086-29', '17-39280577-40847517-203', '17-41062669-41447005-37',
                     '17-41458670-41494331-6']

# Keep only the selected features and the 'Sample' column
df_call_transposed = df_call_transposed[['Sample'] + selected_features]

# CSVs
df_call_transposed.to_csv("df_call_eval_RFE", sep='\t', index=False)