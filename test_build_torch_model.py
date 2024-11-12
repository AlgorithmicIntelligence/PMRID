import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the input, hidden, and output sizes
input_size = 28 * 28  # Example for an image with size 28x28
hidden_size = 128
output_size = 10  # Example for 10 classes

# Create an instance of the network
model = SimpleNet(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data
X = torch.randn(100, input_size)  # 100 samples of 28x28 images
y = torch.randint(0, 10, (100,))  # 100 labels for 10 classes

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model, 'simple_net_model2.pth')
scripted_model = torch.jit.script(model)
scripted_model.save('simple_net.pt')