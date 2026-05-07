#imports
import os
import random
from string import ascii_letters

import torch
from unidecode import unidecode
from sklearn.model_selection import train_test_split

#set device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset path
data_dir = "./data/data/names"

#create language to label mapping
lang2label = {
    file_name.split(".")[0]: torch.tensor([i], dtype=torch.long)
    for i, file_name in enumerate(os.listdir(data_dir))
}

#reverse mapping
label2lang = {
    label.item(): lang
    for lang, label in lang2label.items()
}

#total number of languages
num_langs = len(lang2label)

#character vocabulary
char2idx = {
    letter: i
    for i, letter in enumerate(ascii_letters + " .,:;-'")
}

#total characters
num_letters = len(char2idx)

#convert name into one-hot tensor
def name2tensor(name):

    #shape = (sequence_length, batch_size, input_size)
    tensor = torch.zeros(len(name), 1, num_letters)

    #fill one-hot positions
    for i, char in enumerate(name):
        tensor[i][0][char2idx[char]] = 1

    return tensor

#store tensors and labels
tensor_names = []
target_langs = []

#load dataset
for file in os.listdir(data_dir):

    with open(os.path.join(data_dir, file), encoding="utf-8") as f:

        #extract language name
        lang = file.split(".")[0]

        #clean unicode characters
        names = [unidecode(line.rstrip()) for line in f]

        #convert every name into tensor
        for name in names:

            try:
                tensor_names.append(name2tensor(name))
                target_langs.append(lang2label[lang])

            except KeyError:
                pass

#split dataset
train_idx, test_idx = train_test_split(
    range(len(target_langs)),
    test_size=0.1,
    shuffle=True,
    stratify=target_langs
)

#training dataset
train_dataset = [
    (tensor_names[i], target_langs[i])
    for i in train_idx
]

#testing dataset
test_dataset = [
    (tensor_names[i], target_langs[i])
    for i in test_idx
]