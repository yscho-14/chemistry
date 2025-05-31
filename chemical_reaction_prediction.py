# 1. Import libraries and set up the environment
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# 2. Load the USPTO reaction SMILES dataset
reaction_smiles_df = pd.read_csv("/kaggle/input/reaction-smiles-uspto-year-2023/reactionSmilesFigShareUSPTO2023.txt", 
                                 names=['reaction_smiles'])

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(reaction_smiles_df.shape)
reaction_smiles_df.head()

# reaction_smiles
# Reactant1.Reactant2>Condition>Product1.Product2

%pip install scikit-mol

# 3. Example: check the structure of the first row (reactant, condition, product)
# 1st row data structure
from rdkit import Chem
smiles_strings_0 = ['C(=O)(OC(C)(C)C)OC(=O)OC(C)(C)C.NC=1C=CC(=NC1)C(C#N)(C)C', #reactants
                    'O.O1CCOCC1', #conditions(reagent, solvent, catalyst, ...)
                    'C(#N)C(C)(C)C1=CC=C(C=N1)NC(OC(C)(C)C)=O'] #products
mols_0 = [Chem.MolFromSmiles(smiles) for smiles in smiles_strings_0]
mols_0

# 4. Visualize the molecular structures for the first row
# mol structure
from rdkit.Chem import Draw
from IPython.display import Image, display

mols_img_0 = Draw.MolsToGridImage(mols_0, molsPerRow=len(mols_0), subImgSize=(300, 200))
display(mols_img_0)
# reactants, reagent/solvent, product

# 5. Example: check the structure of the second row
# 2nd row data structure
reaction_smiles_df.reaction_smiles[1]

# 6. Convert the second row's SMILES strings to RDKit Mol objects
# ROMol
smiles_strings_1 = ['BrC1=CC=C(C=C1)CBr.C(#N)C(C)(C)C1=CC=C(C=N1)NC(OC(C)(C)C)=O', 
                    'N(C)(C)C=O.[H-].[Na+].[Cl-].[NH4+]', 
                    'BrC1=CC=C(CN(C(OC(C)(C)C)=O)C=2C=NC(=CC2)C(C)(C)C#N)C=C1']
mols_1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_strings_1]
mols_1

# 7. Visualize the molecular structures for the second row
# structure
mols_img_1 = Draw.MolsToGridImage(mols_1, molsPerRow=len(mols_1), subImgSize=(300, 200))
display(mols_img_1)
# reactants, reagents/solvents, product
# H- : Metal Hydride such as NaH

# 8. Split reaction_smiles into separate columns: reactant, condition, and product
# sep smiles_set to reactants, condition and product col 
reaction_smiles_df = reaction_smiles_df['reaction_smiles'].str.split('>', expand=True)
reaction_smiles_df.columns = ['reactant', 'condition', 'product']
print(reaction_smiles_df.shape)
reaction_smiles_df.head()

# 9. Generate RDKit Mol objects (ROMol) for reactant and product columns
# generate reactant_ROMol & product_ROMol
from rdkit.Chem import AllChem, PandasTools, Descriptors

PandasTools.AddMoleculeColumnToFrame(frame=reaction_smiles_df, smilesCol='reactant', 
                                     molCol='reactant_ROMol')

# generate product_ROMol column
PandasTools.AddMoleculeColumnToFrame(frame=reaction_smiles_df, smilesCol='product', 
                                     molCol='product_ROMol')

print(reaction_smiles_df.shape)
reaction_smiles_df.head()

reaction_smiles_df.info()

# 10. Drop rows with missing values (NaN)
reaction_smiles_df = reaction_smiles_df.dropna()
reaction_smiles_df.shape 

# 11. Split data into training and validation sets
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(reaction_smiles_df, test_size=0.2, random_state=14)
train_df['set'] = 'train'
val_df['set'] = 'val'

print(train_df.shape, val_df.shape)
train_df.head(2)

val_df.head(2)

reaction_df = pd.concat([train_df, val_df], ignore_index=False)
reaction_df = reaction_df.sort_index()

print(reaction_df.shape)
reaction_df.head(2)

# 12. Define a SMILES tokenizer (character-level vectorizer)
# RDKit based
class SmilesIndexer:
    def __init__(self, start_char='[', end_char=']', canonical=True, isomericSmiles=True):
        self.startchar = start_char
        self.endchar = end_char
        self.canonical = canonical
        self.isomericSmiles = isomericSmiles
        self.char_to_int = {}
        self.int_to_char = {}
        self.dims = None
        self.charset = ""

    def fit(self, mols):
        """Builds the vocabulary and determines the maximum length of SMILES strings."""
        vocabulary = set()
        smiles_list = []
        max_len = 0

        for mol in mols:
            if isinstance(mol, Chem.rdchem.Mol):
                smiles = Chem.MolToSmiles(mol, canonical=self.canonical, isomericSmiles=self.isomericSmiles)
                smiles_with_special_chars = self.startchar + smiles + self.endchar
                smiles_list.append(smiles_with_special_chars)
                max_len = max(max_len, len(smiles_with_special_chars))
                for char in smiles_with_special_chars:
                    vocabulary.add(char)
            elif isinstance(mol, str):
                rdkit_mol = Chem.MolFromSmiles(mol)
                if rdkit_mol is not None:
                    smiles = Chem.MolToSmiles(rdkit_mol, canonical=self.canonical, isomericSmiles=self.isomericSmiles)
                    smiles_with_special_chars = self.startchar + smiles + self.endchar
                    smiles_list.append(smiles_with_special_chars)
                    max_len = max(max_len, len(smiles_with_special_chars))
                    for char in smiles_with_special_chars:
                        vocabulary.add(char)
                else:
                    print(f"Warning: Could not parse SMILES: {mol} during fitting.")

        vocabulary = sorted(list(vocabulary))
        self.char_to_int = {char: idx for idx, char in enumerate(vocabulary)}
        self.int_to_char = {idx: char for idx, char in enumerate(vocabulary)}
        self.charset = "".join(vocabulary)
        self.dims = (len(vocabulary), max_len)

    def tokenize(self, mols, augment=None, canonical=None):
        tokenized = []

        if augment is None:
            augment = False  
        if canonical is None:
            canonical = self.canonical

        for mol in mols:
            if isinstance(mol, Chem.rdchem.Mol):
                smiles = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=self.isomericSmiles)
            elif isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    print(f"Warning: Could not parse SMILES: {mol} for tokenization.")
                    continue
                smiles = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=self.isomericSmiles)
            else:
                print(f"Warning: Invalid input type for mol: {type(mol)}")
                continue

            smiles = "%s%s%s"%(self.startchar, smiles, self.endchar)

            tokens = torch.tensor([self.char_to_int.get(char, self.char_to_int.get('<unk>', 0)) for char in smiles], dtype=torch.long)
            tokenized.append(tokens)

        return tokenized

    def reverse_tokenize(self, vect, strip=True):
        smiles = []
        for v in vect:
            chars = [self.int_to_char.get(i.item(), '') for i in v]
            smile = "".join(chars)
            if strip:
                smile = smile.strip(self.startchar + self.endchar)
            smiles.append(smile)
        return np.array(smiles)

tokenizer = SmilesIndexer()
tokenizer.fit(np.concatenate([reaction_df.reactant_ROMol.values, reaction_df.product_ROMol.values]))
tokenizer.charset = " %0" + tokenizer.charset

print("Dimensions:\t%s"%(str(tokenizer.dims)))
print("Charset:\t%s"%tokenizer.charset)

"""
Dimensions:	(58, 336)
Charset:	 %0#%()*+-./0123456789:=@ABCFGHIKLMNOPRSTVWZ[\]acdeghilnorstu
"""

# 13. Tokenize and visualize a batch of product molecules
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
print(torch.__version__)

product_tokens = tokenizer.tokenize(reaction_df.product_ROMol[0:20])
print([len(v) for v in product_tokens])

from torch.nn.utils.rnn import pad_sequence

product_padded = pad_sequence(product_tokens)
plt.matshow(product_padded.numpy().T)
plt.show()


# 14. Split data into train/validation sets for model input
X_train = reaction_df.reactant_ROMol[reaction_df.set == "train"]
y_train = reaction_df.product_ROMol[reaction_df.set == "train"]
X_val = reaction_df.reactant_ROMol[reaction_df.set == "val"]
y_val = reaction_df.product_ROMol[reaction_df.set == "val"]
X_train.shape, y_train.shape, X_val.shape, y_val.shape
# ((109800,), (109800,), (27450,), (27450,))

y_val

print(reaction_df['set'].unique())

print(f"Number of validation samples with 'val': {len(reaction_df[reaction_df['set'] == 'val'])}")

X_train.head()


# 15. Set device for PyTorch (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 16. Define PyTorch Dataset class for reactant/product pairs
class MolDataset(Dataset):
    def __init__(self, reactant, product, tokenizer, augment):
        self.reactant = reactant
        self.product = product
        self.tokenizer = tokenizer
        self.augment = augment
    def __len__(self):
        return len(self.reactant)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        reactant = self.reactant.iloc[idx]
        product = self.product.iloc[idx]
        reactant_tokens = self.tokenizer.tokenize([reactant], augment=self.augment)[0]
        product_tokens = self.tokenizer.tokenize([product], augment=self.augment)[0]
         
        return reactant_tokens, product_tokens

train_dataset = MolDataset(X_train, y_train, tokenizer, augment=False)
val_dataset = MolDataset(X_val, y_val, tokenizer, augment=False)
print(train_dataset, val_dataset)

reactant_tokens, product_tokens = val_dataset[0]
reactant_tokens, product_tokens


# 17. Define collate function and DataLoader for batching
batch_size=120
def collate_fn(r_and_p_list):
    r, p = zip(*r_and_p_list)
    return pad_sequence(r), pad_sequence(p)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          collate_fn=collate_fn,
                                          num_workers=2,
                                          drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=500,
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          num_workers=2,
                                          drop_last=True)
train_loader

for reactant, product in train_loader:
    break
reactant.shape


# 18. Define the LSTM-based sequence-to-sequence model for reaction prediction
import torch.nn.functional as F
class MolBrain(nn.Module):
    def __init__(self, num_tokens, hidden_size, embedding_size, dropout_rate):
        super(MolBrain, self).__init__() 
         
        self.embedding = nn.Embedding(num_tokens, embedding_size) 

        self.lstm_encoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size//2, num_layers=1,
                                    batch_first=False, bidirectional=True)
         
        #Second layer of the encoder
        self.lstm_encoder_2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2, num_layers=1,
                                    batch_first=False, bidirectional=True)
         
        #Transform the output states into a larger size for non-linear transformation
        self.latent_encode = nn.Linear(hidden_size, hidden_size*2)
         
        #Decode the latent code into the start states for the decoder
        self.h0_decode = nn.Linear(hidden_size*2, hidden_size)
        self.c0_decode = nn.Linear(hidden_size*2, hidden_size)
        self.h0_decode_2 = nn.Linear(hidden_size*2, hidden_size)
        self.c0_decode_2 = nn.Linear(hidden_size*2, hidden_size)
         
        #First layer of the decoder
        self.lstm_decoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=1,
                                    batch_first=False, bidirectional=False)
         
        #Second layer of the decoder
        self.lstm_decoder_2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1,
                                    batch_first=False, bidirectional=False)
         
        #fully connected layers for transforming the LSTM output into the probability distribution
        self.fc0 = nn.Linear(hidden_size, hidden_size*2)
        self.fc1 = nn.Linear(hidden_size*2, num_tokens) # Output layer
         
        #Activation function, dropout and softmax layers
        self.activation = nn.ReLU() 
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)
    def encode_latent(self, reactant):
        #If batch_size is needed, we can get it like this
        batch_size = reactant.shape[1]
         
        #Embed the reactants tensor
        reactant = self.embedding(reactant)
   
        #Pass through the encoder
        lstm_out, (h_n, c_n) = self.lstm_encoder(reactant)
        #print(lstm_out.shape)
        lstm_out2, (h_n_2, c_n_2) = self.lstm_encoder_2(lstm_out)
        #h_n is (num_layers * num_directions, batch, hidden_size)
       
        #Sum the backward and forward direction last states of the LSTM encoders
        h_n = h_n.sum(axis=0).unsqueeze(0)
        h_n_2 = h_n_2.sum(axis=0).unsqueeze(0)
        #Alternative use internal states
        c_n = c_n.sum(axis=0).unsqueeze(0)
        c_n_2 = c_n_2.sum(axis=0).unsqueeze(0)
        #Concatenate output of both LSTM layers
        #hs = torch.cat([h_n, h_n_2], 2)
        cs = torch.cat([c_n, c_n_2], 2)
         
        #Non-linear transform of the hs into the latent code
        latent_code = self.latent_encode(cs)
        latent_code = self.dropout(self.activation(latent_code))
        return latent_code
 
    def latent_to_states(self, latent_code):
        h_0 = self.h0_decode(latent_code)
        c_0 = self.c0_decode(latent_code)
        h_0_2 = self.h0_decode_2(latent_code)
        c_0_2 = self.c0_decode_2(latent_code)
        return (h_0, c_0, h_0_2, c_0_2)
    def decode_states(self, states, product_in):
        h_0, c_0, h_0_2, c_0_2 = states
        #Embed the teachers forcing product input
        product_in = self.embedding(product_in)
         
        #Pass through the decoder
        out, (h_n, c_n) = self.lstm_decoder(product_in, (h_0, c_0))
        out_2, (h_n_2, c_n_2) = self.lstm_decoder_2(out, (h_0_2, c_0_2))
        #A final dense hidden layer and output the logits for the tokens
        out = self.fc0(out_2)
        out = self.dropout(out)
        out = self.activation(out)
        logits = self.fc1(out)
         
        return logits, (h_n, c_n, h_n_2, c_n_2)
    def forward(self, reactant, product_in):
        latent_code = self.encode_latent(reactant)
        states = self.latent_to_states(latent_code)
        logits, _ = self.decode_states(states, product_in)
        return logits 

# 19. Instantiate the model and move it to the selected device
num_tokens = tokenizer.dims[1]
hidden_size=256
embedding_size=128
dropout_rate=0.25
epochs = 75
batch_size=128
max_lr = 0.004
model = MolBrain(num_tokens, hidden_size, embedding_size, dropout_rate)
model.to(device)

# 20. Test model output shape with a sample batch
out = model(reactant.to(device), product[:-1,:].to(device))
out.shape

# 21. Set up optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
optimizer.param_groups[0]['lr']

from torch.optim.lr_scheduler import OneCycleLR
epochs*len(train_loader)

scheduler = OneCycleLR(optimizer=optimizer, max_lr = max_lr, total_steps = epochs*len(train_loader),
                      div_factor=25, final_div_factor=0.08)
optimizer.param_groups[0]['lr']


# 22. (Optional) Training loop and progress plotting function
# Model training loop
import ipywidgets
%matplotlib inline
def plot_progress():
    out.clear_output()
    with out:
        print("Epoch %i, Training loss: %0.4F, Validation loss %0.4F, lr %.2E"%(e, train_loss, val_loss, lrs[-1]))
        fig, ax1 = plt.subplots()
        ax1.plot(losses, label="Train loss")
        ax1.plot(val_losses, label="Val loss")
        ax1.set_ylabel("Loss")
         
        ax1.set_yscale('log')
        ax1.set_xlabel("Epochs")
        ax1.legend(loc=2)
        ax1.set_xlim((0,epochs))
        #Axes 2 for the lr
        ax2 = ax1.twinx()
        ax2.plot(lrs, c="r", label="Learning Rate")
        ax2.tick_params(axis='y', labelcolor="r")
        ax2.set_ylabel("Learning rate")
        ax2.set_yscale('log')
        ax2.legend(loc=0)
        plt.show()

# 23. Start the model training loop with progress tracking
# using GPU T4 x2
from tqdm import tqdm
model.train() #Ensure the network is in "train" mode with dropouts active
losses = []
val_losses = []
lrs = []
out = ipywidgets.Output()
display(out)
for e in range(epochs):
    running_loss = 0
    for reactant, product in tqdm(train_loader, mininterval=1):
        reactant_in = reactant.to(device)
        product_in = product[:-1,:].to(device) #Including starttoken, excluding last
        product_out = product[1:,:].to(device) #Not including start-token
         
        optimizer.zero_grad() # Initialize the gradients, which will be recorded during the forward pass
         
        output = model(reactant_in, product_in) #Forward pass of the mini-batch # (batch, sequence - 1, ohe)
        output_t = output.transpose(1,2)
         
        loss = nn.CrossEntropyLoss()(output_t, product_out)
         
        loss.backward()
        optimizer.step() # Optimize the weights
        scheduler.step() # Adjust the learning rate
         
        running_loss += loss.item()
    else:
        with torch.no_grad(): #Don't calculate the gradients
            model.eval() #Evaluation mode
            running_val_loss = 0
            for reactant_val, product_val in val_loader:
                reactant_in = reactant_val.to(device)
                product_in = product_val[:-1,:].to(device)
                product_out = product_val[1:,:].to(device)
                pred_val = model.forward(reactant_in, product_in)
                pred_val = pred_val.transpose(1,2)
                val_loss = nn.CrossEntropyLoss()(pred_val, product_out).item()
                running_val_loss = running_val_loss + val_loss
            val_loss = running_val_loss/len(val_loader)
            model.train() #Put back in train mode
             
        train_loss = running_loss/len(train_loader)
        losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        plot_progress()

# 24. Save the trained model and tokenizer to disk
import pickle
save_dir = "/kaggle/working/"
pickle.dump(model, open(f"{save_dir}seq2seq_molbrain_model.pickle","wb"))
pickle.dump(tokenizer, open(f"{save_dir}seq2seq_molbrain_model_tokenizer.pickle","wb"))


# 25. Prepare a batch from the validation set for inference
_ = model.eval()
for reactant, product in val_loader:
    reactant_in = reactant.to(device)
    product_in = product[:-1,:].to(device)
    product_out = product[1:,:].to(device)
    break
reactant_in.shape


# 26. Visualize the predicted output logits for a specific sample in the batch
i = 0 #Select compound i from validation batch
with torch.no_grad():
  pred = model.forward(reactant_in, product_in)
  pred_cpu = pred[:,i,:].cpu().detach().numpy()
pred_cpu.shape

plt.matshow(pred_cpu.T)

# 27. Convert logits to predicted token indices and decode to SMILES
indices = pred_cpu.argmax(axis=1)
indices.shape

smiles = tokenizer.reverse_tokenize(indices.reshape(1,-1), strip=False)
smiles[0]

target_smiles= tokenizer.reverse_tokenize(product_out.T, strip=False)
target_smiles[i]

Chem.MolFromSmiles(smiles[0].strip(" $"))


# 28. Visualize the latent vector and LSTM states for the selected sample
latent = model.encode_latent(reactant_in[:,i:i+1])
plt.plot(latent.cpu().detach().numpy().flatten())

states = model.latent_to_states(latent)
plt.plot(states[0].cpu().detach().numpy().flatten())
print(states[0].shape)

plt.plot(states[1].cpu().detach().numpy().flatten())

# 29. Define a greedy decoding function to generate SMILES from latent states
def greedy_decode(model, states):
    char = tokenizer.char_to_int["["]  # Assume start token is '['
    last_char = char
    stop_char = tokenizer.char_to_int["]"]  # Assume end token is ']'
    char = torch.tensor(char, device=device).long().reshape(1,-1) #The first input
    chars = [] #Collect the sampled characters
    for i in range(200):
        out, states = model.decode_states(states, char.reshape(1,-1))
        out = model.softmax(out)
        char = out.argmax() #Sample Greedy and update char
        last_char = char.item() 
        if last_char == stop_char:
            break
        chars.append(last_char)

    return chars

# 30. Use greedy decoding to generate and decode the predicted SMILES
smiles = greedy_decode(model, states)
result = tokenizer.reverse_tokenize(np.array([smiles]))
result


def greedy_decode(model, states):
    char = tokenizer.char_to_int["["] # 시작 토큰을 '['으로 가정 (이전 코드의 '^'와 다를 수 있음)
    last_char = char
    stop_char = tokenizer.char_to_int["]"] # 종료 토큰을 ']'으로 가정 (이전 코드의 '$'와 다를 수 있음)
    char = torch.tensor(char, device=device).long().reshape(1,-1) #The first input
    chars = [] #Collect the sampled characters
    for i in range(200):
        out, states = model.decode_states(states, char.reshape(1,-1))
        out = model.softmax(out)
        char = out.argmax() #Sample Greedy and update char
        last_char = char.item() 
        if last_char == stop_char:
            break
        chars.append(last_char)

    return chars

smiles = greedy_decode(model, states)
result = tokenizer.reverse_tokenize(np.array([smiles]))
result

Chem.MolFromSmiles(result[0], sanitize=False)

# 31. Compare the predicted SMILES with the ground truth
target_smiles= tokenizer.reverse_tokenize(product_out.T)
#target_smiles[i]
print(target_smiles[i])
Chem.MolFromSmiles(target_smiles[i].strip(" $"))


# 32. Analyze latent and state shapes for the entire validation batch
reactant_in.shape

latent = model.encode_latent(reactant_in)
latent.shape

states = model.latent_to_states(latent)
states[0].shape

states[1].shape

# 33. Batch prediction: generate SMILES predictions for the whole validation set
results = []

val_df = reaction_df[reaction_df['set'] == "val"].iloc[0:500]
reactants_smiles = val_df.reactant_ROMol.tolist()
product_smiles_gt = val_df.product_ROMol.tolist()

for i in range(len(val_df)):

    h_in = states[0][:,i:i+1,:]
    c_in = states[1][:,i:i+1,:]
    h_in_2 = states[2][:,i:i+1,:]
    c_in_2 = states[3][:,i:i+1,:]

    chars = greedy_decode(model, (h_in, c_in, h_in_2, c_in_2))
    smiles = tokenizer.reverse_tokenize(np.array([chars]))[0]

    reactant_smiles = reactants_smiles[i]
    product_smiles = product_smiles_gt[i]

    results.append({"product": product_smiles,
                    "reactants": reactant_smiles,
                    "predicted": smiles})

print(results)

# 34. Convert ROMol objects to readable SMILES strings for result visualization
from rdkit import Chem

readable_results = []
for result in results:
    product_mol = result['product']
    reactants_mol = result['reactants']
    predicted_smiles = result['predicted']

    product_smiles = Chem.MolToSmiles(product_mol) if product_mol else None
    reactant_smiles = Chem.MolToSmiles(reactants_mol) if reactants_mol else None

    readable_results.append({
        "product": product_smiles,
        "reactants": reactant_smiles,
        "predicted": predicted_smiles
    })

# 35. Print the first 10 prediction results for inspection
for item in readable_results[:10]: 
    print(f"Product:   {item['product']}")
    print(f"Reactants: {item['reactants']}")
    print(f"Predicted: {item['predicted']}")
    print("-" * 30)

