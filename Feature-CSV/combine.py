import pandas as pd

# Load CSV files
df1 = pd.read_csv('LBP_max_GLCM(D2_2).csv')       # e.g. your old features
df2 = pd.read_csv('features_AlexNet_conv1_48.csv')   # new features

# Drop duplicate label if present in df2
if 'label' in df2.columns:
    df2 = df2.drop(columns=['label'])

# Concatenate side-by-side
df_combined = pd.concat([df1, df2], axis=1)

# Save combined features
df_combined.to_csv('features_102_Alex_den.csv', index=False)
