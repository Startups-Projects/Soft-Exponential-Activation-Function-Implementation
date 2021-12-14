from os import confstr, write
from re import S
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Dense_layer(tf.keras.layers.Layer):
    def __init__(self, output_neurons):
        super(Dense_layer, self).__init__()
        self.units = output_neurons

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)

        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        self.trainable_weight = [self.w, self.b]

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            })
        return config

    def call(self, inputs):
        dot_product = tf.matmul(inputs, self.w)+self.b
        return dot_product


class ParametricSoftExp(tf.keras.layers.Layer):
    def __init__(self, alpha_init=-1, **kwargs):
        self.alpha_init = tf.keras.backend.cast_to_floatx(alpha_init)
        super(ParametricSoftExp, self).__init__()

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.alphas = tf.Variable(self.alpha_init*np.ones(input_shape), trainable=True,dtype=tf.float32)
        self.trainable_weight = [self.alphas]

    def call_alpha_gt0(self, x, alpha):
        return alpha + (tf.math.exp(alpha * x) - 1.) / alpha

    def call_alpha_lt0(self, x, alpha):
        # base_alpha = (1 - alpha**2)/alpha
        return -(1/alpha) * tf.math.asinh(1 - alpha * (x + alpha))
        # safe_log = tf.where(x > 0 and x > base_alpha, tf.math.log(1 - alpha*(x + alpha)), tf.ones_like(x))
        # return tf.where(x > 0 and x > base_alpha, - (1/alpha) * safe_log, tf.ones_like(x))

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha_init': self.alpha_init})
        return config

    def call(self, x):
        return tf.keras.backend.switch(self.alphas > 0, self.call_alpha_gt0(x, self.alphas), tf.keras.backend.switch(self.alphas < 0, self.call_alpha_lt0(x, self.alphas), x))


class neural_network:
    def __init__(self, train_data, train_label, test_data, test_label, epochs=50, batch_size=128):
        self.train_data = tf.Variable(train_data/255, dtype=tf.float32)
        self.train_label = tf.one_hot(train_label, depth=10)
        self.test_data = tf.Variable(test_data/255, dtype=tf.float32)
        self.test_label = tf.one_hot(test_label, depth=10)
        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs', histogram_freq=1, write_graph=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        ]

    # @tf.function
    def model(self):
        inputs = tf.keras.layers.Input(shape=self.train_data.shape[1:])
        flat = tf.keras.layers.Flatten()(inputs)

        self.dense1 = Dense_layer(128)
        dense1_dense = self.dense1(flat)
        self.activ1 = ParametricSoftExp()
        dense1_act = self.activ1(dense1_dense)

        self.dense2 = Dense_layer(128)
        dense2_dense = self.dense2(dense1_act)
        self.activ2 = ParametricSoftExp()
        dense2_act = self.activ2(dense2_dense)

        output = tf.keras.layers.Dense(
            10, activation='softmax', kernel_initializer='glorot_uniform')(dense2_act)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        loss = tf.keras.losses.CategoricalCrossentropy()
        model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
        return model
    
    def model_with_relu(self):
        inputs = tf.keras.layers.Input(shape=self.train_data.shape[1:])
        flat = tf.keras.layers.Flatten()(inputs)
        self.hidden1 = tf.keras.layers.Dense(128, activation='relu')(flat)
        self.hidden2 = tf.keras.layers.Dense(128, activation='relu')(self.hidden1)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(self.hidden2)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        loss = tf.keras.losses.CategoricalCrossentropy()
        model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
        return model

    def train(self, plot=True):
        model = self.model()
        history = model.fit(np.array(self.train_data), np.array(self.train_label),
                            epochs=self.epochs, batch_size=self.batch_size, validation_data=(np.array(self.test_data), np.array(self.test_label)), callbacks=self.callbacks)
        model.summary()
        if plot:
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
        return history.history


if __name__ == '__main__':
    (train_data, train_label), (test_data,
                                test_label) = tf.keras.datasets.mnist.load_data()

    classifier = neural_network(train_data, train_label, test_data, test_label)
    classifier.train()
