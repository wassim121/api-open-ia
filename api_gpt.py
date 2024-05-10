import numpy as np  # Import NumPy library for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib library for visualization
from collections import Counter  # Import Counter class from collections module

# Define a dictionary of points with two categories: 'blue' and 'orange'
points = {'blue': [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
          'orange': [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]]}

new_point = [3, 3]  # Define a new point for prediction


# Define a function to calculate Euclidean distance between two points
def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


# Define a class for k-nearest neighbors classifier
class KNearestNeighbors:

    # Constructor method to initialize the number of neighbors (k)
    def __init__(self, k=3):
        self.k = k
        self.points = None  # Initialize points to be None

    # Method to fit the training data
    def fit(self, points):
        self.points = points  # Store the training data

    # Method to predict the class of a new point
    def predict(self, new_point):
        distances = []  # Initialize an empty list to store distances

        # Calculate distances between the new point and all points in training data
        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category])

        # Select k-nearest neighbors and predict the most common class among them
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result


# Instantiate the KNearestNeighbors classifier with k=3
clf = KNearestNeighbors(k=3)
clf.fit(points)  # Fit the training data
print(clf.predict(new_point))  # Predict the class of the new point

# Visualize KNN Distances in 2D

# Set up the plot
ax = plt.subplot()
ax.grid(False, color='#000000')  # Turn off grid lines

# Set background color
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')

# Set axis colors
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Scatter plot the points for each category
for point in points['blue']:
    ax.scatter(point[0], point[1], color='#104DCA', s=60)

for point in points['orange']:
    ax.scatter(point[0], point[1], color='#EF6C35', s=60)

# Scatter plot the new point with appropriate color based on prediction
new_class = clf.predict(new_point)
color = '#EF6C35' if new_class == 'orange' else '#104DCA'
ax.scatter(new_point[0], new_point[1], color=color, marker='*', s=200, zorder=100)

# Plot dashed lines connecting the new point to k-nearest neighbors
for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color='#104DCA', linestyle='--', linewidth=1)

for point in points['orange']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color='#EF6C35', linestyle='--', linewidth=1)

plt.show()  # Display the plot

# 3D Example

# Define a dictionary of points in 3D space
points = {'blue': [[2, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 3], [2, 1, 6]],
          'orange': [[5, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6], [10, 10, 4]]}

new_point = [3, 3, 4]  # Define a new point in 3D space

clf = KNearestNeighbors(k=3)  # Instantiate the KNearestNeighbors classifier with k=3
clf.fit(points)  # Fit the training data
print(clf.predict(new_point))  # Predict the class of the new point

# Set up the 3D plot
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.grid(True, color='#323232')  # Turn on grid lines

# Set background color
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')

# Set axis colors
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Scatter plot the points for each category
for point in points['blue']:
    ax.scatter(point[0], point[1], point[2], color='#104DCA', s=60)

for point in points['orange']:
    ax.scatter(point[0], point[1], point[2], color='#EF6C35', s=60)

# Scatter plot the new point with appropriate color based on prediction
new_class = clf.predict(new_point)
color = '#EF6C35' if new_class == 'orange' else '#104DCA'
ax.scatter(new_point[0], new_point[1], new_point[2], color=color, marker='*', s=200, zorder=100)

# Plot dashed lines connecting the new point to k-nearest neighbors
for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color='#104DCA',
            linestyle='--', linewidth=1)

for point in points['orange']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color='#EF6C35',
            linestyle='--', linewidth=1)

plt.show()  # Display the plot
