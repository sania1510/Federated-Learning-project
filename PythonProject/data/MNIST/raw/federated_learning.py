import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import flwr as fl  # Flower framework
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Define the model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.log_softmax(self.fc5(x), dim=1)
        return x

# Prepare the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Split the dataset into client-specific subsets
def get_data_loaders(client_id: int, num_clients: int, batch_size: int):
    client_dataset_size = len(dataset) // num_clients
    indices = range(client_id * client_dataset_size, (client_id + 1) * client_dataset_size)
    client_dataset = torch.utils.data.Subset(dataset, indices)
    train_size = int(0.8 * len(client_dataset))
    val_size = len(client_dataset) - train_size
    train_dataset, val_dataset = random_split(client_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.train_accuracies = []

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            correct, total = 0, 0
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total
            self.train_accuracies.append(accuracy)
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {accuracy:.4f}")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(num_epochs=5)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                output = self.model(images)
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        return float(accuracy), len(self.val_loader.dataset), {}

# Start Flower server
def start_server():
 from flwr.server.app import ServerConfig
 # Configure the strategy
 strategy = fl.server.strategy.FedAvg(
 min_available_clients=2, # Minimum number of clients required to proceed
 )
 # Start the server with ServerConfig
 server_config = ServerConfig(num_rounds=3)
 fl.server.start_server(server_address="localhost:8080", strategy=strategy, config=server_config)
# Start Flower clients
def start_client(client_id, num_clients):
    train_loader, val_loader = get_data_loaders(client_id, num_clients, batch_size=32)
    model = Classifier()
    client = FlowerClient(model, train_loader, val_loader)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

# Main execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py [server|client] [client_id|num_clients]")
        sys.exit(1)

    role = sys.argv[1]
    if role == "server":
        start_server()
    elif role == "client":
        if len(sys.argv) != 3:
            print("Usage: python script.py client <client_id>")
            sys.exit(1)
        client_id = int(sys.argv[2])
        num_clients = 10
        start_client(client_id, num_clients)
    else:
        print("Invalid role. Use 'server' or 'client'.")
