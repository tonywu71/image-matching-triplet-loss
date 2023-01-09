import tensorflow as tf


class FFBlock(tf.keras.layers.Layer):
    def _build_layer(self):
        list_layers = [
            tf.keras.layers.Dense(units=self.units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(self.dropout)
        ]
        return tf.keras.Sequential(list_layers)
        
    
    def __init__(self, units: int, dropout: float=0.):
        super().__init__()
        
        self.units = units
        self.dropout = dropout
        
        self.layer = self._build_layer()
        

    def call(self, inputs):
        return self.layer(inputs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "dropout": self.dropout
        }
