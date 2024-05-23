# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt
# import io
# import base64

# app = Flask(__name__)

# # Load data
# df = pd.read_csv('canteen_transactions.csv')

# # Load item names
# items_df = pd.read_csv('items.csv')
# item_id_to_name = dict(zip(items_df['Item ID'], items_df['Item Name']))

# # Split data into features and target
# X = df[['User ID', 'Item ID', 'Quantity']]
# y = df['Rating']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a decision tree classifier
# clf = DecisionTreeClassifier()

# # Train the model
# clf.fit(X_train, y_train)

# # Make predictions
# y_pred = clf.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# # Initialize the KMeans clustering model
# kmeans = KMeans(n_clusters=5, random_state=42)

# # Fit the model to the data
# kmeans.fit(X)

# # Get the cluster labels for each data point
# labels = kmeans.labels_
# df['Cluster'] = labels

# # Define a function to recommend items for a given cluster
# def recommend_items(cluster_id, n_items=5):
#     cluster_data = df[df['Cluster'] == cluster_id]
#     top_items = cluster_data['Item ID'].value_counts().head(n_items).index.tolist()
#     top_item_names = [item_id_to_name[item_id] for item_id in top_items]
#     return top_item_names

# @app.route('/')
# def index():
#     # Generate clustering plot
#     plt.figure()
#     plt.scatter(df['User ID'], df['Item ID'], c=labels, cmap='viridis')
#     plt.xlabel('User ID')
#     plt.ylabel('Item ID')
#     plt.title('Clustering of Canteen Transactions')
#     plt.colorbar(label='Cluster')
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plot_url = base64.b64encode(img.getvalue()).decode()

#     return render_template('index.html', accuracy=accuracy, classification_rep=classification_rep, plot_url=plot_url)

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     cluster_id = int(request.form['cluster_id'])
#     items = recommend_items(cluster_id)
#     return render_template('recommend.html', cluster_id=cluster_id, items=items)

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request
import pandas as pd
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load data
df = pd.read_csv('canteen_transactions.csv')
items_df = pd.read_csv('items.csv')
item_id_to_name = pd.Series(items_df['Item Name'].values, index=items_df['Item ID']).to_dict()
# item_id_to_name = dict(zip(items_df['Item ID'], items_df['Item Name']))

# Split data into features and target
X = df[['User ID', 'Item ID', 'Quantity']]
y = df['Rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Initialize the KMeans clustering model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

def plot_clusters():
    plt.figure(figsize=(8, 6))
    plt.scatter(df['User ID'], df['Item ID'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('User ID')
    plt.ylabel('Item ID')
    plt.title('Clustering of Canteen Transactions')
    plt.colorbar(label='Cluster')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    plot_url = plot_clusters()
    return render_template('index.html', accuracy=accuracy, classification_rep=classification_rep, plot_url=plot_url)

def recommend_items(cluster_id, n_items=5):
    cluster_data = df[df['Cluster'] == cluster_id]
    top_items = cluster_data['Item ID'].value_counts().head(n_items).index.tolist()
    top_item_names = [item_id_to_name[item_id] for item_id in top_items]
    return top_item_names

@app.route('/recommend', methods=['POST'])
def recommend():
    cluster_id = int(request.form['cluster_id'])
    items = recommend_items(cluster_id)
    return render_template('recommend.html', cluster_id=cluster_id, items=items)

if __name__ == '__main__':
    app.run(debug=True)
