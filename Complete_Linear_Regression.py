import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

print(tf.__version__)

# THIS IS THE FIRST DATASETS
# # Create features
# X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# # Create labels 
# y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# # Visualize it 
# # plt.scatter(X,y)
# # plt.show()

# # Figure out the relationship
# print(y == X + 10)

# input_shape = X.shape
# output_shape = y.shape
# print(input_shape, output_shape)

# # Converting numpy to tensors
# X = tf.constant(X)
# y = tf.constant(y)

# print(X, y)


# CREATE NEW DATASETS WITH MORE DATA
# Steps in modelling in tensorflow
# Set the random seed
tf.random.set_seed(42)

# Evaluating the model
# Create new datasets
X = tf.range(-100, 100, 4)
# Make label for the datasets
y = X + 10

# Split the data into test and train dataset
X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

# Visualize data
# plt.figure(figsize=(10,7))
# plt.scatter(X_train, y_train, c="b", label="Training data")
# plt.scatter(X_test, y_test, c="g", label="Testing data")
# plt.legend()
# plt.show()

# Build model 3
tf.random.set_seed(42)
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_3.compile(
    loss=tf.keras.losses.mae,
    # optimizer=tf.keras.optimizers.Adam(lr=0.01),
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=["mae"]
)

model_3.fit(X_train, y_train, epochs=500)

y_preds_3 = model_3.predict(X_test)


def plot_predictions(predictions,train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show()

print(model_3.predict([20]))
plot_predictions(predictions=y_preds_3)








