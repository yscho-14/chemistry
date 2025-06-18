
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
                   
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
        
import warnings
warnings.filterwarnings("ignore", category=Warning)

# 1. Data Loading
sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head()

test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
print(test.shape)
test.head()

# a hidden test set

train= pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
print(train.shape)
train.head()

"""
2. Prediction Target
Tg : Glass Transition Temperature
FFV : Fractional Free Volume
Tc : Thermal Conductivity
Density
Rg : Radius of Gyration
"""

"""
3. QSPR(Quantitative Structure–Property Relationship)-SVM
SMILES → Molecualr Descriptors(using RDKit) → SVM Regression
---

descriptor_names_tg = [
    'MolWt',                # Molecular weight
    'MolLogP',              # Hydrophobicity/Polarity
    'NumHDonors',           # Number of hydrogen bond donors
    'NumHAcceptors',        # Number of hydrogen bond acceptors
    'TPSA',                 # Topological polar surface area
    'NumRotatableBonds',    # Number of rotatable bonds (molecular flexibility)
    'RingCount',            # Number of rings
    'HeavyAtomCount',       # Number of heavy atoms
    'FractionCSP3',         # Fraction of sp3 carbons (structural diversity)
    'BalabanJ',             # Topological index (molecular complexity)
    'Chi0',                 # Connectivity index
    'Kappa3',               # Shape index
]

descriptor_names_ffv = [
    'MolWt',
    'TPSA',
    'LabuteASA',           # Accessible surface area
    'MolMR',               # Molecular refractivity (volume-related)
    'NumAliphaticRings',   # Number of aliphatic rings
    'NumAromaticRings',    # Number of aromatic rings
    'FractionCSP3',
    'NumRotatableBonds',
    'HeavyAtomCount',
    'PEOE_VSA1',           # Partial charge surface area 1
    'PEOE_VSA6',           # Partial charge surface area 6
]

descriptor_names_tc = [
    'MolWt',
    'MolLogP',
    'TPSA',
    'LabuteASA',
    'NumAromaticRings',
    'NumAliphaticRings',
    'BalabanJ',
    'Chi3n',               # Connectivity index
    'Kappa2',              # Shape index
    'HeavyAtomCount',
    'EState_VSA1',         # EState surface area 1
    'EState_VSA9',         # EState surface area 9
]

descriptor_names_density = [
    'MolWt',
    'MolMR',
    'TPSA',
    'LabuteASA',
    'HeavyAtomCount',
    'FractionCSP3',
    'NumAliphaticRings',
    'NumAromaticRings',
    'PEOE_VSA2',
    'PEOE_VSA8',
    'Kappa1',
]

descriptor_names_rg = [
    'MolWt',
    'NumRotatableBonds',
    'HeavyAtomCount',
    'FractionCSP3',
    'LabuteASA',
    'BalabanJ',
    'Kappa3',
    'Chi1v',
    'TPSA',
    'NumAliphaticCarbocycles',
    'NumAromaticRings',
]

# RDKit library
!pip install rdkit-pypi scikit-learn pandas tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# descriiptors to use
descriptor_names_all= list(set(
    descriptor_names_tg +
    descriptor_names_ffv +
    descriptor_names_tc +
    descriptor_names_density +
    descriptor_names_rg
))

calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names_all)
calc

# convert smiles to descriptors
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan]*len(descriptor_names_all)
    return list(calc.CalcDescriptors(mol))

smiles_to_descriptors

# Apply descriptors to the train dataset
train_desc_list = []
for smi in tqdm(train['SMILES'], desc='Calculating descriptors'):
    train_desc_list.append(smiles_to_descriptors(smi))
train_desc_df = pd.DataFrame(train_desc_list, columns=descriptor_names_all)

# combine train dataset
train_desc = pd.concat([train, train_desc_df], axis=1)

print(train_desc.shape)
train_desc.head()

# Apply descriptors to the test dataset
test_desc_list = []
for smi in tqdm(test['SMILES'], desc='Calculating descriptors'):
    test_desc_list.append(smiles_to_descriptors(smi))
test_desc_df = pd.DataFrame(test_desc_list, columns=descriptor_names_all)

# combine test dataset
test_desc = pd.concat([test, test_desc_df], axis=1)

print(test_desc.shape)
test_desc.head()

# 3-1. Prediction for Tg, Glass Transition Temperature

# 1. features and target data for Tg
train_Tg = train_desc[~train_desc['Tg'].isna()].copy()

train_Tg_features = train_Tg.drop(['id', 'SMILES',	'Tg',	'FFV',	'Tc',	'Density',	'Rg'], axis=1)
train_Tg_target = train_Tg['Tg']

print(train_Tg_features.shape, train_Tg_target.shape)
train_Tg_features.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# 2. Train & Validation set 
X_train_Tg, X_val_Tg, y_train_Tg, y_val_Tg = train_test_split(
    train_Tg_features, train_Tg_target, test_size=0.2, random_state=42
)

# 3. Scaling
scaler = StandardScaler()
X_train_Tg_scaled = scaler.fit_transform(X_train_Tg)
X_val_Tg_scaled = scaler.transform(X_val_Tg)

X_train_Tg_scaled.shape, X_val_Tg_scaled.shape

# 5. SVM Regression Model
svm_Tg = SVR(kernel='rbf', C=300, epsilon=1, gamma=0.01)
svm_Tg.fit(X_train_Tg_scaled, y_train_Tg)

# 6. Prediction
y_pred_Tg = svm_Tg.predict(X_val_Tg_scaled)

# 7. Evaluation
mae = mean_absolute_error(y_val_Tg, y_pred_Tg)
r2 = r2_score(y_val_Tg, y_pred_Tg)
print(f"Validation MAE: {mae:.3f}")
print(f"Validation R2: {r2:.3f}")

# 8. Predicted Values
import pandas as pd
result_df = pd.DataFrame({
    "True_Tg": y_val_Tg,
    "Pred_Tg": y_pred_Tg
})
print(result_df.head())

"""
Validation MAE: 51.008
Validation R2: 0.553
"""

# 3-2. Prediction for FFV, Fractional Free Volume
# 1. features and target data for FFV
train_FFV = train_desc[~train_desc['FFV'].isna()].copy()

train_FFV_features = train_FFV.drop(['id', 'SMILES', 'Tg', 'FFV', 'Tc',	'Density',	'Rg'], axis=1)
train_FFV_target = train_FFV['FFV']

print(train_FFV_features.shape, train_FFV_target.shape)
train_FFV_features.head()


# 2. Train & Validation set 
X_train_FFV, X_val_FFV, y_train_FFV, y_val_FFV = train_test_split(
    train_FFV_features, train_FFV_target, test_size=0.2, random_state=42
)

# 3. Scaling
scaler = StandardScaler()
X_train_FFV_scaled = scaler.fit_transform(X_train_FFV)
X_val_FFV_scaled = scaler.transform(X_val_FFV)

X_train_FFV_scaled.shape, X_val_FFV_scaled.shape

# 5. SVM Regression Model
svm_FFV = SVR(kernel='rbf', C=10, epsilon=0.001, gamma='scale'
             )
svm_FFV.fit(X_train_FFV_scaled, y_train_FFV)

# 6. Prediction
y_pred_FFV = svm_FFV.predict(X_val_FFV_scaled)

# 7. Evaluation
mae = mean_absolute_error(y_val_FFV, y_pred_FFV)
r2 = r2_score(y_val_FFV, y_pred_FFV)
print(f"Validation MAE: {mae:.3f}")
print(f"Validation R2: {r2:.3f}")

# 8. Predicted Values
import pandas as pd
result_df = pd.DataFrame({
    "True_FFV": y_val_FFV,
    "Pred_FFV": y_pred_FFV
})
print(result_df.head())
"""
Validation MAE: 0.009
Validation R2: 0.597
"""

# 4. Submission

test_features = test_desc.drop(['id', 'SMILES'], axis=1)
test_features_scaled = scaler.fit_transform(test_features)
test_features_scaled = pd.DataFrame(test_features_scaled, columns=test_features.columns)
test_features_scaled.head()

test_pred_Tg = svm_Tg.predict(test_features_scaled)
test_pred_FFV = svm_FFV.predict(test_features_scaled)
test_pred_Tc = svm_Tc.predict(test_features_scaled)
test_pred_Density = svm_Density.predict(test_features_scaled)
test_pred_Rg = svm_Rg.predict(test_features_scaled)
test_pred_Tg

test_pred_Tg_orig = scaler.inverse_transform(test_pred_Tg.reshape(-1, 1)).flatten()
test_pred_FFV_orig = scaler.inverse_transform(test_pred_FFV.reshape(-1, 1)).flatten()
test_pred_Tc_orig = scaler.inverse_transform(test_pred_Tc.reshape(-1, 1)).flatten()
test_pred_Density_orig = scaler.inverse_transform(test_pred_Density.reshape(-1, 1)).flatten()
test_pred_Rg_orig = scaler.inverse_transform(test_pred_Rg.reshape(-1, 1)).flatten()
