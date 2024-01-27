#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select the GPU index
import nltk
from torch.optim.lr_scheduler import _LRScheduler
import time
import math
import matplotlib.pyplot as plt
import spacy
from scipy.io import savemat
import dill as pickle
import thop


# In[2]:


# Dictionary
dicno=3
dictionaries =[['AAGCGC', 'ACGAGC', 'ATGGCG', 'AGCACG', 'CAGGTG', 'CCTACG', 'CTACGC', 'CGAACG', 'TAGGGC',
                'TCGTGC', 'TTGCCG', 'TGCTCG', 'GACGAC', 'GCAGCA', 'GTAGGC', 'GGTCGA'],
               
               ['AAGCGC', 'ACGAGC', 'ATCCGC', 'ATGGCG', 'AGCACG', 'CATCCG', 'CAGGTG', 'CCTACG', 'CCTGGT',
                'CCGTTG', 'CTACGC', 'CTGGAG', 'CGAACG', 'CGGAAC', 'TAGGGC', 'TCCGCA', 'TCGTGC', 'TCGGGA',
                'TTCGGC', 'TTGCCG', 'TGCTCG', 'GACGAC', 'GATGCG', 'GCAGCA', 'GCCTAC', 'GTAGGC', 'GGACGT',
                'GGCTAG', 'GGCTTC', 'GGTAGC', 'GGTCCT', 'GGTCGA' ],
               
               ['ACTACGACA', 'ACTGACACA', 'ACGACTACA', 'ATCAGCACA', 'AGCATCACA', 'CATCAGACA', 'CAGCATACA', 
                'TACGACACA', 'GACTACACA', 'ACTACTGAG', 'ACTACGATG', 'ACTACGTAG', 'ACTACGTGA', 'ACTGACATG',
                'ACTGACTAG', 'ACTGACTGA', 'ACTGACGAT', 'ACTGACGTA', 'ACTGATACG', 'ACTGATAGC', 'ACTGATCGA', 
                'ACTGATGAC', 'ACTGATGCA', 'ACTGCTAGA', 'ACGACTATG', 'ACGACTAGT', 'ACGACTGAT', 'ACGACTGTA', 'ACGATGACT', 
                'ACGATGATC', 'ACGATGCTA', 'ACGATGTAC', 'ACGATGTCA', 'ATCATCGAG', 'ATCAGCATG', 'ATCAGCAGT', 'ATCAGCTGA', 
                'ATCAGCGTA', 'ATCAGTACG', 'ATCAGTAGC', 'ATCAGTCAG', 'ATCAGTCGA', 'ATCAGTGCA', 'ATCGTACAG', 'ATCGTACGA', 
                'ATCGTAGAC', 'ATCGTAGCA', 'ATCGTCAGA', 'ATGACTACG', 'ATGACTAGC', 'ATGACTCGA', 'ATGACTGAC', 'ATGACTGCA', 
                'ATGACGACT', 'ATGACGATC', 'ATGACGCTA', 'ATGACGTCA', 'ATGATGCAC', 'ATGCTACAG', 'ATGCTACGA', 'ATGCTAGAC', 
                'ATGCTAGCA', 'ATGCTGACA', 'AGCATCATG', 'AGCATCAGT', 'AGCATCTAG', 'AGCATCTGA', 'AGCATCGTA', 'AGCAGTACT', 
                'AGCAGTATC', 'AGCAGTCAT', 'AGCAGTCTA', 'AGTAGCATC', 'AGTAGCTAC', 'AGTAGCTCA', 'AGTAGTCAC', 'AGTCATACG', 
                'AGTCATAGC', 'AGTCATCAG', 'AGTCATCGA', 'AGTCATGCA', 'AGTCAGATC', 'AGTCAGCAT', 'AGTCAGCTA', 'AGTCAGTAC', 
                'AGTCAGTCA', 'AGTCGTACA', 'CATCAGAGT', 'CATCAGTAG', 'CATCAGTGA', 'CATCGTAGA', 'CAGCATATG', 'CAGCATAGT', 
                'CAGCATGAT', 'CAGCATGTA', 'CAGCAGTAT', 'CAGTAGATC', 'CAGTAGCAT', 'CAGTAGCTA', 'CAGTAGTAC', 'CAGTAGTCA', 
                'CAGTCATAG', 'CAGTCATGA', 'CAGTCAGAT', 'CAGTCAGTA', 'CAGTCGATA', 'CTACTGAGA', 'CTACGATGA', 'CTACGAGAT', 
                'CTACGAGTA', 'CTGACTAGA', 'CTGACGATA', 'CTGATGACA', 'CGACTATAG', 'CGACTATGA', 'CGACTAGAT', 'CGACTAGTA',
                'CGACTGATA', 'CGACGATAT', 'CGATGACTA', 'CGATGATAC', 'CGATGCATA', 'CGTAGCATA', 'CGTCATAGA', 'CGTCAGATA', 
                'TACTACGAG', 'TACTGACAG', 'TACTGACGA', 'TACTGAGAC', 'TACTGAGCA', 'TACTGCAGA', 'TACGACAGT', 'TACGACTAG',
                'TACGACTGA', 'TACGACGAT', 'TACGATACG', 'TACGATAGC', 'TACGATCAG', 'TACGATCGA', 'TACGATGAC', 'TACGATGCA', 
                'TAGCATACG', 'TAGCATAGC', 'TAGCATCAG', 'TAGCATCGA', 'TAGCATGAC', 'TAGCATGCA', 'TAGCAGACT', 'TAGCAGCAT', 
                'TAGCAGTAC', 'TAGCAGTCA', 'TAGTAGCAC', 'TAGTCACAG', 'TAGTCACGA', 'TAGTCAGAC', 'TAGTCAGCA', 'TAGTCGACA', 
                'TCATCAGAG', 'TCATCGAGA', 'TCAGCATAG', 'TCAGCAGAT', 'TCAGCAGTA', 'TCAGTACAG', 'TCAGTACGA', 'TCAGTAGAC',
                'TCAGTAGCA', 'TCAGTCAGA', 'TCGTAGACA', 'TGACTACAG', 'TGACTACGA', 'TGACTAGAC', 'TGACTAGCA', 'TGACTGACA', 
                'TGACGACAT', 'TGACGACTA', 'TGACGATAC', 'TGATGACAC', 'TGATGCACA', 'TGCTACAGA', 'GACTACATG', 'GACTACTAG', 
                'GACTACTGA', 'GACTACGAT', 'GACTACGTA', 'GACTGACAT', 'GACTGACTA', 'GACTGATAC', 'GACTGATCA', 'GACTGCATA', 
                'GACGACTAT', 'GACGATACT', 'GACGATATC', 'GACGATCAT', 'GACGATCTA', 'GATGACACT', 'GATGACTAC', 'GATGACTCA', 
                'GATGCTACA', 'GCATCATAG', 'GCATCAGAT', 'GCATCAGTA', 'GCATCGATA', 'GCAGCATAT', 'GCAGTACAT', 'GCAGTACTA', 
                'GCAGTATAC', 'GCAGTATCA', 'GCAGTCATA', 'GCTACGATA', 'GCTGACATA', 'GCTGATACA', 'GTAGCACAT', 'GTAGCACTA', 
                'GTAGCATCA', 'GTAGTCACA', 'GTCATCAGA', 'GTCAGCATA', 'GTCAGTACA', 'ACTGATGTG', 'ACGATGTGT', 'ATCAGTGTG', 
                'ATCGTAGTG', 'ATGACTGTG', 'ATGACGTGT', 'ATGATGCTG', 'ATGATGCGT', 'ATGATGTCG', 'ATGATGTGC', 'ATGCTAGTG', 
                'ATGCTGATG', 'ATGCTGAGT', 'ATGCTGTGA', 'AGTAGCTGT', 'AGTAGTCTG', 'AGTAGTCGT', 'AGTAGTGCT', 'AGTAGTGTC', 
                'AGTCATGTG', 'AGTCAGTGT', 'AGTCGTATG', 'AGTCGTAGT', 'AGTCGTGTA', 'CAGTAGTGT', 'CTGATGATG', 'CTGATGAGT', 
                'CTGATGTAG', 'CTGATGTGA', 'CGATGATGT', 'CGTAGTATG', 'CGTAGTAGT', 'CGTAGTGAT', 'CGTAGTGTA', 'TACGATGTG', 
                'TAGCATGTG', 'TAGCAGTGT'],
               
               ['ACGTAGCAGA', 'ACTACAGACG', 'AGACGATGCA', 'AGCGACTATC', 'ATAGCTCGTG', 'CACGAGCTAT','CGTCTGTAGT',
                'CTATCAGCGA', 'GATAGTCGCT', 'GCAGACATCA', 'GTGCTCGATA', 'TATCGAGCAC', 'TCGCTGATAG', 'TCTGCTACGT',
                'TGATGTCTGC', 'TGCATCGTCT'],
               
               ['ATCGATCGATCGAT','TACGTAGCTAGCTA', 'TAGCTACGTAGCTA', 'ATGCATGCATCGAT', 'ATGCTACGATGCTA', 
                'TAGCATGCTACGAT', 'TACGATCGTACGAT', 'ATCGTAGCATGCTA', 'TACGATGCTAGCAT', 'ATCGTACGATCGTA', 
                'ATGCTAGCATCGTA', 'TAGCATCGTAGCAT', 'TAGCTAGCTACGTA', 'ATGCATCGATGCAT', 'ATCGATGCATGCAT', 
                'TACGTACGTACGTA'],
              
               ['ATCGATCG', 'ATCGATGC', 'ATCGTAGC', 'ATCGTACG', 'ATGCTAGC', 'ATGCTACG', 'ATGCATCG', 'ATGCATGC',
                'TAGCTAGC', 'TAGCTACG', 'TAGCATCG', 'TAGCATGC', 'TACGATCG', 'TACGATGC', 'TACGTAGC', 'TACGTACG']
              
               ]
dicts=dictionaries[dicno]
print(dicts)


# In[3]:


batch_size=2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr=0.001
num_epochs=1000
patience=400
train_num_samples=30000
test_num_samples=10000
num_errors=4
K=len(dicts[0])
hidden_size=32


# In[4]:


path_directory=f"substitutionmodels/BiLSTM/Dictionary_{dicno}/batch_size_{batch_size}/num_errors_{num_errors}/initial_lr_{lr}/epochs{num_epochs}_p{patience}_tr{train_num_samples}ts{test_num_samples}K{K}hs{hidden_size}/"
path_directory


# In[5]:


# Check if "models" folder exists, create it if it doesn't
if not os.path.exists(path_directory):
    os.makedirs(path_directory)


# In[6]:


class LanguageDataset(Dataset):

    def __init__(self, dictionary, num_samples, num_errors):
        self.dictionary = dictionary
        self.num_samples = num_samples 
        self.num_errors = num_errors

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        source_idx = idx % len(self.dictionary)  
        sequence = self.dictionary[source_idx]

        if self.num_errors > 0:
            sequence = self.introduce_errors(sequence, self.num_errors)

        x = torch.tensor([self.char_to_idx(c) for c in sequence], dtype=torch.long)
        y = torch.tensor([self.char_to_idx(c) for c in self.dictionary[source_idx]], dtype=torch.long)
        return x, y

    def introduce_errors(self, sequence, num_errors):
        sequence = list(sequence)
        for _ in range(num_errors):
            i = random.randint(0, len(sequence)-1)
            if sequence[i] == 'A':
                sequence[i] = random.choice(['T', 'G', 'C'])
            elif sequence[i] == 'T':
                sequence[i] = random.choice(['A', 'G', 'C'])
            elif sequence[i] == 'G':
                sequence[i] = random.choice(['A', 'T', 'C'])
            elif sequence[i] == 'C':
                sequence[i] = random.choice(['A', 'T', 'G'])

        return ''.join(sequence)

    def char_to_idx(self, char):
        return {'A': 0, 'C': 1, 'T': 2, 'G': 3}[char]


# In[7]:


random.randint(1, 2)


# In[8]:


class LanguageDataset1(Dataset):

    def __init__(self, dictionary, num_samples, num_errors):
        self.dictionary = dictionary
        self.num_samples = num_samples 
        self.num_errors = num_errors

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        source_idx = idx % len(self.dictionary)  
        sequence = self.dictionary[source_idx]

        if self.num_errors > 0:
            sequence = self.introduce_errors(sequence, self.num_errors)

        x = torch.tensor([self.char_to_idx(c) for c in sequence], dtype=torch.long)
        y = torch.tensor([self.char_to_idx(c) for c in self.dictionary[source_idx]], dtype=torch.long)
        return x, y

    def introduce_errors(self, sequence, num_errors):
        ne = random.randint(1, num_errors)
        sequence = list(sequence)
        for _ in range(ne):
            i = random.randint(0, len(sequence)-1)
            if sequence[i] == 'A':
                sequence[i] = random.choice(['T', 'G', 'C'])
            elif sequence[i] == 'T':
                sequence[i] = random.choice(['A', 'G', 'C'])
            elif sequence[i] == 'G':
                sequence[i] = random.choice(['A', 'T', 'C'])
            elif sequence[i] == 'C':
                sequence[i] = random.choice(['A', 'T', 'G'])

        return ''.join(sequence)

    def char_to_idx(self, char):
        return {'A': 0, 'C': 1, 'T': 2, 'G': 3}[char]


# In[9]:


train_data = LanguageDataset1(dicts, num_samples=train_num_samples, num_errors=num_errors)
test_data = LanguageDataset1(dicts, num_samples=test_num_samples, num_errors=num_errors)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# In[10]:


t1=iter(train_loader)
X,y=next(t1)


# In[11]:


X.shape


# In[12]:


y


# In[13]:


def one_hot_encode(sequences, n_unique):

    # Get size of batch and max sequence length
    batch_size,seq_len=sequences.shape
    #len(sequences)
    #seq_len=len(sequences[0])
    
    # Initialize encoded tensor
    oneencoding = torch.zeros(batch_size, seq_len, n_unique, device = device)
    # Set 1 at index of each character
    for i in range(batch_size):
        for j in range(seq_len):
            index = sequences[i][j]
            oneencoding[i, j, index] = 1          
    return oneencoding


# In[14]:


X.shape


# In[15]:


def one_hot_decode(encoded_seq):
    """Converts a batch of one-hot encoded sequences to indices"""
    
    # Dimensions will be (batch_size, seq_len, num_classes)
    batch_size, seq_len, num_classes = encoded_seq.shape
    
    # Flatten to treat as a list of one-hot vectors
    encoded_seq = encoded_seq.reshape(-1, num_classes)
    
    # Argmax over classes to get class index
    indices = encoded_seq.argmax(dim=1).to(device)
    
    # Reshape back to original dimensions
    indices = indices.reshape(batch_size, seq_len)
    
    return indices


# In[16]:


class BiLSTMModel(nn.Module):

    def __init__(self, hidden_size, dropout):
        super().__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=4, 
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(2*hidden_size, 4)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):

        # BiLSTM 1
        out, _ = self.lstm1(x)
        out = self.softmax(self.fc(out))
        return out


# In[17]:


model=BiLSTMModel(hidden_size,0.1)


# In[18]:


def cal_model_parameters(model):
    total_param  = []
    for p1 in model.parameters():
        total_param.append(int(p1.numel()))
    return sum(total_param)


# In[19]:


cal_model_parameters(model)


# In[20]:


model=model.to(device)


# In[21]:


train_next=iter(train_data)
Xp, Yp=next(train_next)
#Xp = torch.cat([Xp.view(1,-1), 4*torch.ones(1, num_errors, dtype=torch.int).to(device)], dim=1)
xp=one_hot_encode(Xp.view(1,-1).to(device), 4).to(device)


# In[22]:


flops, params = thop.profile(model, inputs=(xp,), verbose=False)


# In[23]:


print(f'Model Flops: {flops}')
print(f'Model Params Num: {params}\n')


# In[24]:


criterion= nn.MSELoss().to(device)


# In[25]:


optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)


# In[26]:


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]


# In[27]:


scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=num_epochs * len(train_loader),
                                            T_warmup=20 * len(train_loader),
                                            eta_min=1e-6)


# In[28]:


start_time = time.time()
num_train_batches=len(train_loader)
num_test_batches=len(test_loader)
train_losses = []
val_losses = []
patience_counter = 0
best_val_loss = float('inf')

for i in range(num_epochs):
    loss1 = 0
    epoch_time = time.time()
    model.train()
    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):  
        optimizer.zero_grad()
        y_train=y_train.to(device)
        # Apply the model
        xt=one_hot_encode(X_train.to(device), 4).to(device)
        y_pred=model(xt)
        #print(y_pred.shape)
        yt=one_hot_encode(y_train, 4).to(device)
        loss = criterion(y_pred, yt) 
        # Update parameters
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss1=loss1+loss
        #if (b+1)%50==1:
        #    print(f'epoch: {i+1}/{num_epochs} batch:{b+1}/{num_train_batches} loss: {loss.item():10.8f}')      
    train_loss=loss1/num_train_batches  
    train_losses.append(train_loss.item())
    # Run the testing batches
    model.eval()
    with torch.no_grad():
        loss1=0
        for b, (X_test, y_test) in enumerate(test_loader):
            y_test=y_test.to(device)
            # Apply the model
            xt=one_hot_encode(X_test.to(device), 4).to(device)
            y_pred=model(xt)
            
            yt=one_hot_encode(y_test, 4).to(device)
            loss = criterion(y_pred, yt) 
            loss1=loss1+loss         
        val_loss=loss1/num_test_batches  
        val_losses.append(val_loss.item())
    
    print(f'epoch:{i+1}/{num_epochs} average TL:{train_loss.item():10.8f} average VL:{val_loss.item():10.8f} epoch time:{time.time() - epoch_time:.0f} seconds, lr:{optimizer.param_groups[0]["lr"]:.2e}')               
     # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), path_directory+"model.pth")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Validation loss did not decrease for {patience} epochs. Stopping training.')
        break  
        
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed            


# In[29]:


# Plotting
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# In[30]:


dict={'train_losses':train_losses, 'val_losses':val_losses, 'flops':flops, 'parameters':params }


# In[31]:


savemat(path_directory+'losses.mat', dict)
with open(path_directory+'losses.pkl', 'wb') as file:
    
    # A new file will be created
    pickle.dump(dict, file)


# # Testing

# In[32]:


model.load_state_dict(torch.load(path_directory+"model.pth"))


# In[33]:


start_test=1
stop_test=4
num_test=200000
test_error_points=torch.arange(start_test, stop_test+1)
#nucleotide error rate
ner=torch.zeros((stop_test-start_test+1)).to(device)
#sequence error rate
ser=torch.zeros((stop_test-start_test+1)).to(device)


# In[34]:


def count_different_rows(matrix1, matrix2):
    """
    Calculates the number of rows that are not equal between two 2D PyTorch matrices.

    Args:
        matrix1 (torch.Tensor): A 2D PyTorch tensor of shape (m, n).
        matrix2 (torch.Tensor): A 2D PyTorch tensor of shape (m, n).

    Returns:
        int: The number of rows that are not equal between the two matrices.
    """
    assert matrix1.shape == matrix2.shape, "The input matrices must have the same shape."
    equal_rows = torch.all(matrix1 == matrix2, dim=1)
    return matrix1.shape[0] - torch.sum(equal_rows).item()


# In[35]:


i1=0
for i in range(start_test, stop_test+1):
    test_data1 = LanguageDataset(dicts, num_samples=num_test, num_errors=i)
    test_loader1 = DataLoader(test_data1, batch_size=batch_size, shuffle=True)
    n_tbatch=len(test_loader1)
    ne=0.0
    count=0.0
    # Run the testing batches
    model.eval()
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader1):
            y_test=y_test.to(device)
            # Apply the model
            xt=one_hot_encode(X_test.to(device), 4).to(device)
            y_pred=model(xt)
            y_pred_d=one_hot_decode(y_pred)
            ne=ne+torch.sum(y_pred_d!=y_test).item()
            count=count+count_different_rows(y_test,y_pred_d)
        ner[i1]=ne/(n_tbatch*batch_size*K)
        ser[i1]=count/(n_tbatch*batch_size)
    i1=i1+1


# In[36]:


dict1={'ner':torch.Tensor.cpu(ner).numpy(), 'ser':torch.Tensor.cpu(ser).numpy()}


# In[37]:


ner


# In[38]:


ner


# In[39]:


ser


# In[40]:


savemat(path_directory+'metric.mat', dict1)
with open(path_directory+'metric.pkl', 'wb') as file:
    
    # A new file will be created
    pickle.dump(dict1, file)


# In[41]:


plt.plot(torch.Tensor.cpu(test_error_points).numpy(),torch.Tensor.cpu(ner).numpy())
plt.yscale('log')
plt.xlabel('Number of errors')
plt.ylabel('NER')    
plt.grid(True)
plt.show()


# In[42]:


plt.plot(torch.Tensor.cpu(test_error_points).numpy(),torch.Tensor.cpu(ser).numpy())
plt.yscale('log')
plt.xlabel('Number of errors')
plt.ylabel('SER')    
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




