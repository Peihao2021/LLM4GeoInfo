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

df = pd.read_csv('task1_embeddings.csv')
print(type(df.loc[0,'embed1']))
df['embed1'] = df['embed1'].apply(string_to_array)
df['embed2'] = df['embed2'].apply(string_to_array)
df['embed3'] = df['embed3'].apply(string_to_array)
print(type(df.loc[0,'embed1']))
# Create a features array with embeddings only
X = np.array([np.concatenate([row['embed1'], row['embed2'], row['embed3']]) for index, row in df.iterrows()])
y = df['label'].values

city_names = df[['city1', 'city2', 'city3']]
indices = df.index.to_numpy()

# Split data and indices
X_train, X_test, y_train, y_test, indices_train, indices_test = \
      train_test_split(X, y, indices, test_size=0.2, random_state=42, stratify=y)

#print("Indices of rows in the test set:")
# for index in indices_test:
#     print(index)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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