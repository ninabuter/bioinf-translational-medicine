""""
This script transforms the original call dataset, changes the column names to the chromosomal data, merges the
subgroup column from the clinical dataset, and creates a csv file.
"""

import pandas as pd

# Import arrayCGH copy number data and label data
df_call = pd.read_table("Train_call.txt", sep="\t")
df_clinical = pd.read_table("Train_clinical.txt", sep="\t")

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

# Merge the "Subgroup" column from df_clinical to df_call_transposed
df_call_transposed = pd.merge(df_clinical, df_call_transposed, on="Sample")

# CSV
df_call_transposed.to_csv("df_call_transposed", sep='\t', index=False)

