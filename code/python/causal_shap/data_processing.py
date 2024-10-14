# data_processing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data_metabolites(self):
        self.df = pd.read_excel(self.data_path)
        self.df = self.df.drop(columns=['HAD_Anxiety', 'Patient', 'Batch_metabolomics', 'BH', 'Sex', 'Age', 'BMI','Race','Education','HAD_Depression','STAI_Tanxiety', 'Diet_Category','Diet_Pattern'])
        return self.df

    def preprocess_raw_data(self, raw_data_path):
        raw_df = pd.read_excel(raw_data_path)
        raw_df = raw_df.drop(columns=['No.', 'RT', 'mz', 'charge', 'quality', 'identifications',
                                      'chemical_formula', 'exp_mass_to_charge', 'RT_mz'])
        raw_df = raw_df.groupby('opt_global_neutral_mass').agg('mean').reset_index()
        raw_df = raw_df.dropna()
        return raw_df

    def encode_labels(self, df, label_column):
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])
        return df, le
