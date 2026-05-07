#imports
import random
import torch
from torch import nn
import matplotlib.pyplot as plt

from utils import *

#GRU model
class GRUModel(nn.Module):

    def __init__(self, hidden_size, num_layers):

        super(GRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #GRU layer
        self.gru = nn.GRU(
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

        #GRU forward pass
        output, hidden_state = self.gru(
            x,
            hidden_state
        )

        #take last timestep output
        output = self.fc(output[-1])

        return output

    def init_hidden(self):

        return torch.zeros(
            self.num_layers,
            1,
            self.hidden_size
        )

#hyperparameters
hidden_size = 256
learning_rate = 0.0005
num_epochs = 15

#create model
model = GRUModel(
    hidden_size=hidden_size,
    num_layers=2
)

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

    for i, (name, label) in enumerate(train_dataset):

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

        output = model(name)

        _, pred = torch.max(output, dim=1)

        num_correct += bool(pred == label)

#accuracy
accuracy = num_correct / len(test_dataset)

print(f"Accuracy: {accuracy*100:.2f}%")

#prediction function
def predict(name):

    model.eval()

    tensor_name = name2tensor(name)

    with torch.no_grad():

        output = model(tensor_name)

        _, pred = torch.max(output, dim=1)

    return label2lang[pred.item()]

print(predict("Mike"))
print(predict("Qin"))
print(predict("Fernando"))

#save model
torch.save(model.state_dict(), "gru_model.pth")

#save plot
plt.figure(figsize=(10,6))
plt.plot(losses)
plt.title("GRU Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")

plt.savefig("gru_loss.png")

plt.show()