import numpy as np
import pickle
import copy

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, weight_reg_l1=0, weight_reg_l2=0, bias_reg_l1=0, bias_reg_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_reg_l1 = weight_reg_l1
        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l1 = bias_reg_l1
        self.bias_reg_l2 = bias_reg_l2

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on params
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_reg_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_reg_l1 * dL1
        # L2 on weights
        if self.weight_reg_l2 > 0:
            self.dweights += 2 * self.weight_reg_l2 * self.weights

        # L1 on biases
        if self.bias_reg_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_reg_l1 * dL1
        # L2 on biases
        if self.bias_reg_l2 > 0:
            self.dbiases += 2 * self.bias_reg_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Retrieve layer params
    def get_params(self):
        return self.weights, self.biases

    # Set weights and biases in a layer instance
    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Dropout layer
class Layer_Dropout:
    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

# Input layer
class Layer_Input:
    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

# Convolutional Layer
class Layer_Conv2D:
    def __init__(self, n_filters, filter_size, stride=1, padding=0, input_channels=None):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        
        # Weights will be initialized when we know input shape
        self.weights = None
        self.biases = np.zeros((n_filters, 1))
        
    def forward(self, inputs, training):
        # Save inputs
        self.inputs = inputs
        
        # Initialize weights if not done yet
        if self.weights is None:
            n_channels = inputs.shape[-1]
            self.weights = 0.01 * np.random.randn(
                self.n_filters, self.filter_size, self.filter_size, n_channels
            )
        
        # Add padding if needed
        if self.padding > 0:
            inputs = np.pad(inputs, 
                          ((0, 0), (self.padding, self.padding), 
                           (self.padding, self.padding), (0, 0)),
                          mode='constant')
        
        batch_size, input_height, input_width, _ = inputs.shape
        output_height = (input_height - self.filter_size) // self.stride + 1
        output_width = (input_width - self.filter_size) // self.stride + 1
        
        # Initialize output
        self.output = np.zeros((batch_size, output_height, output_width, self.n_filters))
        
        # Convolution operation
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                receptive_field = inputs[:, h_start:h_end, w_start:w_end, :]
                for f in range(self.n_filters):
                    self.output[:, i, j, f] = np.sum(
                        receptive_field * self.weights[f] + self.biases[f],
                        axis=(1, 2, 3)
                    )
    
    def backward(self, dvalues):
        batch_size, input_height, input_width, _ = self.inputs.shape
        _, output_height, output_width, _ = dvalues.shape
        
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.sum(dvalues, axis=(0, 1, 2)).reshape(-1, 1)
        self.dinputs = np.zeros_like(self.inputs)
        
        # Add padding to dinputs if needed
        if self.padding > 0:
            padded_dinputs = np.pad(self.dinputs,
                                  ((0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding), (0, 0)),
                                  mode='constant')
        else:
            padded_dinputs = self.dinputs
        
        # Calculate gradients
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                receptive_field = self.inputs[:, h_start:h_end, w_start:w_end, :]
                dout = dvalues[:, i, j, :].reshape(batch_size, 1, 1, 1, -1)
                
                # Gradient w.r.t weights
                self.dweights += np.sum(receptive_field[:, None] * dout, axis=0)
                
                # Gradient w.r.t inputs
                padded_dinputs[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.weights[None] * dout, axis=-1
                )
        
        if self.padding > 0:
            self.dinputs = padded_dinputs[:, self.padding:-self.padding,
                                        self.padding:-self.padding, :]
        else:
            self.dinputs = padded_dinputs

# Simple RNN Layer
class Layer_RNN:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        # Weight matrices
        self.Wx = 0.01 * np.random.randn(n_inputs, n_neurons)  # Input weights
        self.Wh = 0.01 * np.random.randn(n_neurons, n_neurons)  # Hidden weights
        self.b = np.zeros((1, n_neurons))  # Bias
        
        self.states = []  # To store hidden states
    
    def forward(self, inputs, training):
        # inputs shape: (batch_size, time_steps, n_inputs)
        self.inputs = inputs
        batch_size, time_steps, _ = inputs.shape
        
        # Initialize hidden state
        self.states = [np.zeros((batch_size, self.n_neurons))]
        self.output = np.zeros((batch_size, time_steps, self.n_neurons))
        
        # Process sequence
        for t in range(time_steps):
            xt = inputs[:, t, :]
            ht_prev = self.states[-1]
            
            # RNN equation: ht = tanh(Wx * xt + Wh * ht-1 + b)
            ht = np.tanh(
                np.dot(xt, self.Wx) + np.dot(ht_prev, self.Wh) + self.b
            )
            self.states.append(ht)
            self.output[:, t, :] = ht
    
    def backward(self, dvalues):
        # dvalues shape: (batch_size, time_steps, n_neurons)
        batch_size, time_steps, _ = dvalues.shape
        
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)
        self.dinputs = np.zeros_like(self.inputs)
        
        dh_next = np.zeros((batch_size, self.n_neurons))
        
        # Backward pass through time
        for t in reversed(range(time_steps)):
            ht = self.states[t + 1]
            ht_prev = self.states[t]
            xt = self.inputs[:, t, :]
            
            # Gradient of tanh
            dt = dvalues[:, t, :] + dh_next
            dt_raw = dt * (1 - ht * ht)
            
            # Gradients
            self.db += np.sum(dt_raw, axis=0, keepdims=True)
            self.dWx += np.dot(xt.T, dt_raw)
            self.dWh += np.dot(ht_prev.T, dt_raw)
            dh_next = np.dot(dt_raw, self.Wh.T)
            self.dinputs[:, t, :] = np.dot(dt_raw, self.Wx.T)

    def get_params(self):
        return [self.Wx, self.Wh, self.b]

    def set_params(self, params):
        self.Wx, self.Wh, self.b = params

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable, let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and calculate sample-wise gradient
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Sigmoid activation
class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs, training):
        # Save input and calculate/save output of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

# Linear activation
class Activation_Linear:
    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# SGD optimizer
class Optimizer_SGD:
    # Initialize optimizer - set settings, learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update params
    def update_params(self, layer):
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous updates multiplied by retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # Vanilla SGD updates (as before momentum update)
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any param updates
    def post_update_params(self):
        self.iterations += 1

# Adagrad optimizer
class Optimizer_Adagrad:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update params
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Vanilla SGD param update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any param updates
    def post_update_params(self):
        self.iterations += 1

# RMSprop optimizer
class Optimizer_RMSprop:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update params
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        # Vanilla SGD param update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any param updates
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer
class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0.001, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update params
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum layer.weight with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD param update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any param updates
    def post_update_params(self):
        self.iterations += 1

# Common loss class
class Loss:
    # Regularization loss calculation
    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0
        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:
            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_reg_l1 > 0:
                regularization_loss += layer.weight_reg_l1 * np.sum(np.abs(layer.weights))
            # L2 regularization - weights
            if layer.weight_reg_l2 > 0:
                regularization_loss += layer.weight_reg_l2 * np.sum(layer.weights * layer.weights)
            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_reg_l1 > 0:
                regularization_loss += layer.bias_reg_l1 * np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_reg_l2 > 0:
                regularization_loss += layer.bias_reg_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y, include_regularization=False):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated(self, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses, axis=-1)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = - (y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Squared Error loss
class Loss_MeanSquaredError(Loss):  # L2 loss
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):  # L1 loss
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Common accuracy class
class Accuracy:
    # Calculates an accuracy given predictions and ground truth values
    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)
        # Calculate an accuracy
        accuracy = np.mean(comparisons)
        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculates precision value based on passed in ground truth values
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

# Model class
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss, optimizer and accuracy
    def set(self, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()
        # Count all the objects
        layer_count = len(self.layers)
        # Initialize a list containing trainable layers:
        self.trainable_layers = []
        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights" it's a trainable layer - add it to the list of trainable layers
            # We don't need to check for biases - checking for weights is enough
            if hasattr(self.layers[i], 'weights') or isinstance(self.layers[i], Layer_Conv2D) or isinstance(self.layers[i], Layer_RNN):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross-Entropy
        # create an object of combined activation and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X, y, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)
        # Default value if batch size is not being set
        train_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
        # Dividing rounds down. If there are some remaining data but not a full batch, this won't include it
        if train_steps * batch_size < len(X):
            train_steps += 1

        # History dictionary to store loss, accuracy, and learning rate
        self.history = {
            'loss': [],
            'accuracy': [],
            'learning_rate': []
        }
        self.epoch_history = {
            'loss': [],
            'accuracy': []
        }

        # Main training loop
        for epoch in range(1, epochs + 1):
            # Print epoch number
            print(f'epoch: {epoch} ')
            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            # Iterate over steps
            for step in range(train_steps):
                # If batch size is not set - train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                # Otherwise slice a batch
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)
                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                # Perform backward pass
                self.backward(output, batch_y)
                # Optimize (update params)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Store step-wise loss, accuracy, and learning rate
                self.history['loss'].append(loss)
                self.history['accuracy'].append(accuracy)
                self.history['learning_rate'].append(self.optimizer.current_learning_rate)

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate} ')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            self.epoch_history['loss'].append(epoch_loss)
            self.epoch_history['accuracy'].append(epoch_accuracy)
            print(f'training, acc: {epoch_accuracy:.3f}, loss: {epoch_loss:.3f} (data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate} ')

            # If there is the validation data
            if validation_data is not None:
                # Evaluate the model:
                self.evaluate(*validation_data, batch_size=batch_size)

    # Evaluates the model using passed in dataset
    def evaluate(self, X_val, y_val, batch_size=None):
        # Default value if batch size is not being set
        validation_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            # Otherwise slice a batch
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)
            # Calculate the loss
            self.loss.calculate(output, batch_y)
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(f'validation, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f} ')

    # Predicts on the samples
    def predict(self, X, batch_size=None):
        # Default value if batch size is not being set
        prediction_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
        # Dividing rounds down. If there are some remaining data but not a full batch, this won't include it
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []
        # Iterate over steps
        for step in range(prediction_steps):
            # If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X
            # Otherwise slice a batch
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)

    # Performs forward pass
    def forward(self, X, training):
        # Call forward method on the input layer
        # this will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)
        # Call forward method of every object in a chain
        # Pass output of the previous object as a param
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        # "layer" is now the last object from the list,
        # return its output
        return layer.output
    
    # Performs backward pass
    def backward ( self , output , y ):
        # If softmax classifier
        if self.softmax_classifier_output is not None :
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[ - 1 ].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a param
            for layer in reversed (self.layers[: - 1 ]):
                layer.backward(layer.next.dinputs)
            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)
        # Call backward method going through all the objects
        # in reversed order passing dinputs as a param
        for layer in reversed (self.layers):
            layer.backward(layer.next.dinputs)
    
    # Retrieves and returns params of trainable layers
    def get_params ( self ):
        # Create a list for params
        params = []
        #Iterable trainable layers and get their params
        for layer in self.trainable_layers:
            params.append(layer.get_params())
        # Return a list
        return params
    
    # Updates the model with new params
    def set_params ( self , params ):
        # Iterate over the params and layers
        # and update each layers with each set of the params
        for param_set, layer in zip (params,self.trainable_layers):
            layer.set_params(*param_set)

    # Saves the params to a file
    def save_params ( self , path ):
        # Open a file in the binary-write mode
        # and save params into it
        with open (path,'wb' ) as f:
            pickle.dump(self.get_params(), f)

    # Loads the weights and updates a model instance with them
    def load_params ( self , path ):
        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open (path,'rb' ) as f:
            self.set_params(pickle.load(f))

    # Saves the model
    def save ( self , path ):
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)
        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()
        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop( 'output',None)
        model.loss.__dict__.pop( 'dinputs', None )
        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in [ 'inputs','output','dinputs','dweights','dbiases']:
                layer.__dict__.pop( property , None )
        # Open a file in the binary-write mode and save the model
        with open (path,'wb' ) as f:
            pickle.dump(model, f)

    # Loads and returns a model
    @ staticmethod
    def load ( path ):
        # Open file in the binary-read mode, load a model
        with open (path,'rb' ) as f:
            model = pickle.load(f)
        # Return a model
        return model
    
    def summary(self):
        print("\nModel Summary:")
        print("=" * 50)
        total_params = 0

        for i, layer in enumerate(self.layers):
            layer_type = type(layer).__name__
            
            # Check if layer has weights
            if hasattr(layer, 'weights'):
                weight_shape = layer.weights.shape
                bias_shape = layer.biases.shape
                num_params = np.prod(weight_shape) + np.prod(bias_shape)
            else:
                weight_shape = "N/A"
                bias_shape = "N/A"
                num_params = 0

            total_params += num_params
            
            print(f"Layer {i + 1}: {layer_type}")
            print(f"    Weights Shape: {weight_shape}")
            print(f"    Biases Shape: {bias_shape}")
            print(f"    Trainable Params: {num_params}")
            print("-" * 50)
        
        print(f"Total params: {total_params}")
        print("=" * 50)
