import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Convert embeddings from string to numpy.ndarray type
def string_to_array(s):
    s = s.strip("[]")
    return np.fromstring(s, sep=' ')

df = pd.read_csv('gpt2_embeddings.csv')
print(type(df.loc[0,'City1']))
df['City1'] = df['City1'].apply(string_to_array)
df['City2'] = df['City2'].apply(string_to_array)
df['City3'] = df['City3'].apply(string_to_array)
# Create a features array with embeddings only
X = np.array([np.concatenate([row['City1'], row['City2'], row['City3']]) for index, row in df.iterrows()])
y = df['y'].values

city_names = df[['City1', 'City2', 'City3']]
indices = df.index.to_numpy()

# Split data and indices
X_train, X_test, y_train, y_test, indices_train, indices_test = \
      train_test_split(X, y, indices, test_size=0.2, random_state=42, stratify=y)


# Also split the city names to align them with the test set for later analysis
_, city_names_test = train_test_split(city_names, test_size=0.2, random_state=42, stratify=y)

# Train model with embeddings only
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

with open('lr_prediction.txt', 'w') as file:
# Iterate over the predictions, actual labels, and corresponding city names
    for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
        original_index = indices_test[i]
        city1, city2, city3 = city_names_test.iloc[i]
        predicted_city = city3 if pred == 1 else city2
        actual_city = city3 if actual == 1 else city2

        output_str = f"{original_index} {pred == actual}\nWhich is closer to {city1}? Predicted: {predicted_city}, Actual: {actual_city}\n"

        file.write(output_str)

print(classification_report(y_test, y_pred))