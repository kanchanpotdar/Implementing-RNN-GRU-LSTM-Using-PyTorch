#imports
import random
import torch
from torch import nn
import matplotlib.pyplot as plt

from utils import *

#manual vanilla RNN
class MyRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(MyRNN, self).__init__()

        self.hidden_size = hidden_size

        #current input + previous memory -> new memory
        self.in2hidden = nn.Linear(
            input_size + hidden_size,
            hidden_size
        )

        #current input + previous memory -> output
        self.in2output = nn.Linear(
            input_size + hidden_size,
            output_size
        )

    def forward(self, x, hidden_state):

        #combine current character and previous memory
        combined = torch.cat((x, hidden_state), 1)

        #new hidden state
        hidden = torch.tanh(self.in2hidden(combined))

        #prediction output
        output = self.in2output(combined)

        return output, hidden

    #initial memory
    def init_hidden(self):

        return torch.zeros(1, self.hidden_size)

#hyperparameters
hidden_size = 256
learning_rate = 0.0005
num_epochs = 15

#create model
model = MyRNN(
    input_size=num_letters,
    hidden_size=hidden_size,
    output_size=num_langs
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

#training loop
for epoch in range(num_epochs):

    random.shuffle(train_dataset)

    for i, (name, label) in enumerate(train_dataset):

        #reset memory for every new word
        hidden_state = model.init_hidden()

        #process characters one-by-one
        for char in name:

            output, hidden_state = model(
                char,
                hidden_state
            )

        #calculate loss
        loss = criterion(output, label)

        losses.append(loss.item())

        #reset gradients
        optimizer.zero_grad()

        #backpropagation through time
        loss.backward()

        #prevent exploding gradients
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

        hidden_state = model.init_hidden()

        for char in name:

            output, hidden_state = model(
                char,
                hidden_state
            )

        _, pred = torch.max(output, dim=1)

        num_correct += bool(pred == label)

#calculate accuracy
accuracy = num_correct / len(test_dataset)

print(f"Accuracy: {accuracy*100:.2f}%")

#prediction function
def predict(name):

    model.eval()

    tensor_name = name2tensor(name)

    with torch.no_grad():

        hidden_state = model.init_hidden()

        for char in tensor_name:

            output, hidden_state = model(
                char,
                hidden_state
            )

        _, pred = torch.max(output, dim=1)

    return label2lang[pred.item()]

#test predictions
print(predict("Mike"))
print(predict("Qin"))
print(predict("Fernando"))

#save model
torch.save(model.state_dict(), "rnn_model.pth")

#save predictions
with open("rnn_predictions.txt", "w", encoding="utf-8") as f:

    f.write(f"Accuracy: {accuracy*100:.2f}%\n\n")

    with torch.no_grad():

        for name, label in test_dataset:

            hidden_state = model.init_hidden()

            for char in name:

                output, hidden_state = model(
                    char,
                    hidden_state
                )

            _, pred = torch.max(output, dim=1)

            predicted_lang = label2lang[pred.item()]
            actual_lang = label2lang[label.item()]

            f.write(
                f"Predicted: {predicted_lang} | "
                f"Actual: {actual_lang}\n"
            )

#plot loss
plt.figure(figsize=(10,6))
plt.plot(losses)
plt.title("RNN Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")

#save plot
plt.savefig("rnn_loss.png")

plt.show()