import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Convert embeddings from string to numpy.ndarray type
def string_to_array(s):
    s = s.strip("[]")
    return np.fromstring(s, sep=' ')


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)  # Adjust the hidden layer size as needed
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 2)  # Output size is 2 for binary classification

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


df = pd.read_csv('task1_embeddings.csv')
#print(type(df.loc[0,'embed1']))
df['embed1'] = df['embed1'].apply(string_to_array)
df['embed2'] = df['embed2'].apply(string_to_array)
df['embed3'] = df['embed3'].apply(string_to_array)
#print(type(df.loc[0,'embed1']))
# Create a features array with embeddings only
X = np.array([np.concatenate([row['embed1'], row['embed2'], row['embed3']]) for index, row in df.iterrows()])
y = df['label'].values

city_names = df[['city1', 'city2', 'city3']]
indices = df.index.to_numpy()

# Split data and indices
X_train, X_test, y_train, y_test, indices_train, indices_test = \
      train_test_split(X, y, indices, test_size=0.2, random_state=42, stratify=y)
_, city_names_test = train_test_split(city_names, test_size=0.2, random_state=42, stratify=y)

# y is classification (0/1)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long) 
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

# Initialize the model
input_size = X_train.shape[1]
model = SimpleNN(input_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20 
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
all_preds = []
all_labels = []
# Evaluate Model
model.eval()  # Evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        # For evalution metrics
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total} %')



# Writing output to file
with open('nn_prediction.txt', 'w') as file:
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader.dataset):
            outputs = model(inputs.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            actual = labels.item()
            pred = predicted.item()

            original_index = indices_test[i]
            city1, city2, city3 = city_names_test.iloc[i]
            predicted_city = city3 if pred == 1 else city2
            actual_city = city3 if actual == 1 else city2

            output_str = f"{original_index} {pred == actual}\nWhich is closer to {city1}? Predicted: {predicted_city}, Actual: {actual_city}\n"
            file.write(output_str)

print(classification_report(all_labels, all_preds, target_names=['City2', 'City3']))
