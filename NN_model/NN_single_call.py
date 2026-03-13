import os 
import numpy as np
import keras
from auxiliary_functions_NN import create_NN_features

# Ορισμός μονοπατιών
path_to_nn_model = os.path.join('NN_model')

# Φόρτωση δεδομένων
x_vals_import = np.genfromtxt(os.path.join(path_to_nn_model, 'x_vals.csv'), delimiter=",")
features_import = np.genfromtxt(os.path.join(path_to_nn_model, 'features.csv'), delimiter=",")
# --- minimal fix for single-row CSVs ---
x_vals_import = np.atleast_2d(x_vals_import)       # (N,) -> (1,N)
features_import = np.atleast_2d(features_import)   # (17,) -> (1,17)

batch_size = 1  # single curve always
means_import = np.array([60, 60, 0.15, 0.15, 1, 1, 1.25, 1.25, 0.392699081698724, 0.475, 0.475, 200, 0.3, -1, 25.5, 0.05, 4])
halfranges_import = np.array([40, 40, 0.15, 0.15, 0.05, 0.05, 0.1, 0.1, 0.0436332312998582, 0.025, 0.025, 100, 0.05, 1, 24.5, 0.05, 3])

# Κανονικοποίηση χαρακτηριστικών
features_scaled = (features_import - means_import) / halfranges_import

# Δημιουργία χαρακτηριστικών εισόδου για το NN
ncurve = x_vals_import.shape[0]    
flags_flip = np.full((ncurve,), True)

use_flipped_features = False
ifeat1 = [0, 2, 4, 6, 9]
ifeat2 = [1, 3, 5, 7, 10]

use_periodic_features = True
add_feat = 2 if use_periodic_features else 1

NN_input_all = create_NN_features(
    x_vals_import, features_scaled, use_periodic_features,
    use_flipped_features, ifeat1, ifeat2, flags_flip
)

# Φόρτωση του προεκπαιδευμένου μοντέλου
model_path = os.path.join(path_to_nn_model, '_nn_model.h5')
weights_path = os.path.join(path_to_nn_model, '_nn_model.weights.h5')

NN_model = keras.models.load_model(model_path, compile=False)
#NN_model.load_weights(weights_path)
NN_model.load_weights(model_path)

# Υπολογισμός προβλέψεων
y_pred_vector = NN_model.predict(NN_input_all, batch_size=batch_size, verbose=2)
y_pred_matrix = y_pred_vector.reshape(x_vals_import.shape)

np.savetxt(os.path.join(path_to_nn_model, 'y_vals.csv'), y_pred_matrix, fmt='%.15g', delimiter=',')
