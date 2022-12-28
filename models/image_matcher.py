import tensorflow as tf

class ImageMatcher():
    def _create_e2e_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input()
        
        x_1 = self.feature_model(inputs[0])
        x_2 = self.feature_model(inputs[1])
        
        feature_1 = self.model(x_1)
        feature_2 = self.model(x_2)
        
        outputs = tf.keras.layers.Lambda(lambda x_1, x_2: tf.norm(x_1 - x_2, ord='euclidean'))(feature_1, feature_2)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="image_matching")

        return model
        
    
    def __init__(self, model_filepath: str) -> None:
        self.feature_model = None
        self.model = self._create_e2e_model()
        pass
    
    
    def predict(self, im_1, im_2) -> float:
        """Given two images, perform image pre-processing and returns the probability that
        these 2 images are similar.

        Returns:
            float
        """
        # TODO: Image preprocessing (img size, ...)
        
        return self.model(im_1, im_2)
    
    
    def evaluate():
        tf.keras.metrics.AUC(from_logits=False)