import pandas as pd

# Read the file into a DataFrame
df = pd.read_csv("Train_clinical.txt", sep="\t")

# Group by 'Subgroup' and count the frequency of each subgroup
subgroup_frequency = df.groupby('Subgroup').size().reset_index(name='Frequency')

print(subgroup_frequency)