import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor


class Preprocessor:

    def remove_spaces(self, df_path, output_path):
          """
          Remove unnecessary spaces in dataset file.
          params:
          df_path - path to dataset file
          output_path - path where we save processed dataset
          """
          with open(df_path, 'r') as input_file:
            with open(output_path, 'w') as output_file:
                for line in input_file:
                    stripped_line = line.strip()
                    if stripped_line:
                        output_file.write(stripped_line + '\n')
    
    def nan_values_percentage(self, df, thresh=0.1):
        """
        Get percentages of NaN values in each column of dataframe.
        params:
        df - dataframe we process
        thresh - threshold for selecting column (if column has less NaN values than 'thresh', it is appended to 'dc' list)
        return:
        d - dictionary containing {name: NaN percentage} key-value pairs.
        dc - list containing colnames of columns, in which the NaN value precentage is lesser than thresh
        """
        d, dc = dict(), list()
        for name in df.columns:
            nan_values = df[name].isna().sum()
            percentage = nan_values/df.shape[0]
            d[name]= percentage
            if percentage < thresh:
                dc.append(name)
        return d, dc

    def vif(self, df):
        vif_coefs = pd.DataFrame()
        vif_coefs["variables"] = df.columns
        vif_coefs["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return vif_coefs

    def get_cat_num_colnames(self, df):
        """
        Get column names of categorical and numerical features
        params:
        df - dataframe we process
        return:
        categorical_cols, numerical_cols - lists of names of categorical and numerical columns
        """
        categorical_cols, numerical_cols = df.columns[df.dtypes == 'object'].tolist(), df.columns[df.dtypes != 'object'].tolist()
        return categorical_cols, numerical_cols
    
    def data_preprocess(self, df, categorical_cols, numerical_cols, num_imputer_strategy='mean'):
        """
        Preprocess dataset before fitting machine learning model. This includes:
        1. Dealing with NaN values - lets be precise ...
        2. One hot encoding of categorical features
        3. Scaling of numerical features
        params:
        df - dataframe we process
        categorical_cols - list of names of categorical columns
        numerical_cols - list of names of numerical columns
        imputer_strategy - strategy of dealing with NaN values (default - replacing with mean value)
        return:
        X - preprocessed dataset, ready for passing to ML model
        """
        binary_cols = [col for col in categorical_cols if df[col].nunique(dropna=True) == 2]
        multivalue_cols = list(set(categorical_cols) - set(binary_cols))
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
            ('scaler', MinMaxScaler())
            ])

        multivalue_transformer = OneHotEncoder()

        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='if_binary'))
            ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('mul', multivalue_transformer, multivalue_cols),
            ('bin', binary_transformer, binary_cols)
            ])

        processed_data = preprocessor.fit_transform(df)
        columns = numerical_cols + list(preprocessor.transformers_[1][1].get_feature_names_out()) + binary_cols

        X = pd.DataFrame(processed_data, columns=columns)

        return X