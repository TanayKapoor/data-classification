import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def calculate_accuracy(clf, X_train, y_train, X_test, y_test):
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"Train accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")

    plt.plot(clf.estimators_[0].feature_importances_)
    plt.show()

    confusion_matrix = tf.math.confusion_matrix(y_test, test_predictions)
    plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
    plt.show()

    return train_accuracy, test_accuracy