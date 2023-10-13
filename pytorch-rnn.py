# imports
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hiddden_size = 256 
num_classes = 10
learning_rate = 0.001 
batch_size = 64 
num_epochs = 2

# create RNN 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes * 2)
        self.fc2 = nn.Linear(num_classes * 2, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out 

# load data 
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network 
model = RNN(input_size, hiddden_size, num_layers, num_classes).to(device)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network 
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data to cuda if possible 
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward 
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step 
        optimizer.step()

# check accuracy on triaing & testing to see how good our model 
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accucracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0 
    num_samples = 0 
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            # print(x.shape)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)