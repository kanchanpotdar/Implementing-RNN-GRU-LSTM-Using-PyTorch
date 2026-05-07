#imports
import random
import torch
from torch import nn
import matplotlib.pyplot as plt

from utils import *

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

#LSTM model
class LSTMModel(nn.Module):

    def __init__(self, hidden_size, num_layers):

        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_letters,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        #final classifier
        self.fc = nn.Linear(
            hidden_size,
            num_langs
        )

    def forward(self, x):

        #initialize hidden state
        hidden_state = self.init_hidden()

        #initialize cell state
        cell_state = self.init_cell()

        #LSTM forward pass
        output, (hidden_state, cell_state) = self.lstm(
            x,
            (hidden_state, cell_state)
        )

        #take final timestep output
        output = self.fc(output[-1])

        return output

    def init_hidden(self):

        return torch.zeros(
            self.num_layers,
            1,
            self.hidden_size,
            device=device
        )

    def init_cell(self):

        return torch.zeros(
            self.num_layers,
            1,
            self.hidden_size,
            device=device
        )

#hyperparameters
hidden_size = 256
learning_rate = 0.0005
num_epochs = 15

#create model and move to GPU
model = LSTMModel(
    hidden_size=hidden_size,
    num_layers=2
).to(device)

#loss function
criterion = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

#store losses
losses = []

#training
for epoch in range(num_epochs):

    random.shuffle(train_dataset)

    model.train()

    for i, (name, label) in enumerate(train_dataset):

        #move data to GPU
        name = name.to(device)
        label = label.to(device)

        #forward pass
        output = model(name)

        #loss
        loss = criterion(output, label)

        losses.append(loss.item())

        #reset gradients
        optimizer.zero_grad()

        #backpropagation
        loss.backward()

        #gradient clipping
        nn.utils.clip_grad_norm_(
            model.parameters(),
            1
        )

        #update weights
        optimizer.step()

        #print progress
        if (i + 1) % 3000 == 0:

            print(
                f"Epoch {epoch+1} | "
                f"Step {i+1} | "
                f"Loss {loss.item():.4f}"
            )

#evaluation
num_correct = 0

model.eval()

with torch.no_grad():

    for name, label in test_dataset:

        #move data to GPU
        name = name.to(device)
        label = label.to(device)

        output = model(name)

        _, pred = torch.max(output, dim=1)

        num_correct += bool(pred == label)

#accuracy
accuracy = num_correct / len(test_dataset)

print(f"Accuracy: {accuracy*100:.2f}%")

#prediction function
def predict(name):

    model.eval()

    #move tensor to GPU
    tensor_name = name2tensor(name).to(device)

    with torch.no_grad():

        output = model(tensor_name)

        _, pred = torch.max(output, dim=1)

    return label2lang[pred.item()]

print(predict("Mike"))
print(predict("Qin"))
print(predict("Fernando"))

#save model
torch.save(model.state_dict(), "lstm_model.pth")

#save plot
plt.figure(figsize=(10,6))
plt.plot(losses)
plt.title("LSTM Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")

plt.savefig("lstm_loss.png")

plt.show()