import os
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as k

class AC_Model(tf.keras.Model):
    def __init__(self, input_s, num_actions, is_training=True):
        super(AC_Model, self).__init__()
        self.value_s = None
        self.action_s = None
        self.num_actions = num_actions
        self.training = is_training

        # Dicounting hyperparams for loss functions
        self.entropy_coef = 0.01
        self.value_function_coeff = 0.50
        self.max_grad_norm = .50
        self.learning_rate = 7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5
    
        self.hiddenLayer1 = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_initializer=tf.initializers.lecun_uniform(),
            # kernel_regularizer=keras.regularizers.l2(l=0.01),
            name="hidden_layer1", 
            use_bias=True,
            dtype="float64",
            trainable=is_training )
        
        self.hiddenLayer2 = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_initializer=tf.initializers.lecun_uniform(),
            # kernel_regularizer=keras.regularizers.l2(l=0.01),
            name="hidden_layer2", 
            use_bias=True,
            dtype="float64",
            trainable=is_training )
        
        self.hiddenLayer3 = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_initializer=tf.initializers.lecun_uniform(),
            # kernel_regularizer=keras.regularizers.l2(l=0.01),
            name="hidden_layer3", 
            use_bias=True,
            dtype="float64",
            trainable=is_training )
        
        self.hiddenLayer4 = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_initializer=tf.initializers.lecun_uniform(),
            # kernel_regularizer=keras.regularizers.l2(l=0.01),
            name="hidden_layer4", 
            use_bias=True,
            dtype="float64",
            trainable=is_training )

        self.dropout1 = tf.keras.layers.Dropout(.5)
        self.dropout2 = tf.keras.layers.Dropout(.5)
        self.dropout3 = tf.keras.layers.Dropout(.5)
        self.dropout4 = tf.keras.layers.Dropout(.5)

        # self.LN1 = tf.keras.layers.LayerNormalization()
        # self.LN2 = tf.keras.layers.LayerNormalization()
        # self.LN3 = tf.keras.layers.LayerNormalization()
        # self.LN4 = tf.keras.layers.LayerNormalization()

        self.lstm = tf.keras.layers.SimpleRNN(128, trainable=is_training, dtype=tf.float64)

        # Output Layer consisting of an Actor and a Critic
        self._value = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.orthogonal(1.0),
            # kernel_regularizer=keras.regularizers.l2(l=0.01),
            activation='linear',
            name="value_layer",
            use_bias=False,
            trainable=is_training )
   
        self._policy = tf.keras.layers.Dense(
            self.num_actions,
            activation='linear',
            kernel_initializer=tf.keras.initializers.orthogonal(0.1),
            # kernel_regularizer=keras.regularizers.l2(l=0.01),
            use_bias=False,
            name="policy_layer", trainable=is_training )

    def call(self, input_s, keep_p=1.0):

        # NN layers
        hidden1_out = self.hiddenLayer1(input_s)
        # hidden1_out = self.LN1(hidden1_out)
        hidden1_out = self.dropout1(hidden1_out)
        
        hidden2_out = self.hiddenLayer2(hidden1_out)
        # hidden2_out = self.LN2(hidden2_out)
        hidden2_out = self.dropout2(hidden2_out)
        # expanded = tf.expand_dims(hidden2_out, axis=1)
        # expanded = tf.dtypes.cast(expanded, dtype="float32")
        # lstm_out = self.lstm(expanded)
        
        hidden3_out = self.hiddenLayer3(hidden2_out)
        # hidden3_out = self.LN3(hidden3_out)
        hidden3_out = self.dropout3(hidden3_out)
        
        hidden4_out = self.hiddenLayer4(hidden3_out)
        # hidden4_out = self.LN4(hidden4_out)
        hidden4_out = self.dropout4(hidden4_out)

        # Actor and the Critic outputs
        value = self._value(hidden4_out)
        logits = self._policy(hidden4_out)
        action_dist = tf.nn.softmax(logits)

        return logits, action_dist, value
    
    # Makes a step in the environment
    def step(self, observation, keep_per):
        softmax, logits, value = self.call(observation, keep_per)
        return logits.numpy(), softmax.numpy(), tf.squeeze(value).numpy()

    # Returns the critic estimation of the current state value
    def value_function(self, observation):
        action_dist, softmax, value = self.call(observation, 1.0)
        return tf.squeeze(value).numpy()[0]

    def save_model_weights(self): 
        try:
            current_dir = os.getcwd()   
            model_save_path = current_dir + '\Model\checkpoint.tf'
            self.save_weights(model_save_path, save_format='tf')
        except:
            print("ERROR: There was an issue saving the model weights.")
            pass

    def load_model_weights(self):
        current_dir = os.getcwd()
        model_save_path = current_dir + '\Model\checkpoint.tf'
        self.load_weights(filepath=model_save_path)

    # Turn logits to softmax and calculate entropy
    def logits_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)

    # standard entropy
    def softmax_entropy(self, p0):
        return - tf.reduce_sum(p0 * tf.math.log(p0 + 1e-20), axis=1)