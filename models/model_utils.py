from sklearn.metrics import f1_score
import numpy as np

def dice_score(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def train_model(model, X_train, y_train, X_val, y_val, epochs=50):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
    return model, history
