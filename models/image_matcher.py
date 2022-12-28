from typing import Optional
import logging

import tensorflow as tf

from dataloader.data_generator import preprocess_inputs
from models.feature_model import IMAGE_SIZE_EFFICIENTNET


logger = logging.getLogger(__name__)


class ImageMatcher():
    def _create_e2e_model(self, feature_model: tf.keras.Model) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(2, *IMAGE_SIZE_EFFICIENTNET, 3))
        
        x_1 = feature_model(inputs[:, 0])  # type: ignore
        x_2 = feature_model(inputs[:, 1])  # type: ignore
        
        outputs = tf.keras.layers.Lambda(lambda x: - tf.norm(x[0] - x[1], ord='euclidean', axis=-1))([x_1, x_2])
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="image_matching")

        return model
        
    
    def __init__(self, model_filepath: str) -> None:
        self.feature_model = tf.keras.models.load_model(model_filepath, compile=False)
        self.model = self._create_e2e_model(feature_model=self.feature_model)  # type: ignore
        
        self.model.compile(metrics=[tf.keras.metrics.AUC(from_logits=True)])
        
        logger.info("Successfully created E2E model.")
    
    
    def __call__(self, dataset: tf.data.Dataset) -> tf.Tensor:
        return self.model(dataset)  # type: ignore
    
    
    def get_auc(self, dataset: tf.data.Dataset, steps: Optional[int]=None) -> float:
        _, auc = self.model.evaluate(dataset, steps=steps) # first element is the undefined loss
        return auc
    
    
    def predict(self, im_1, im_2) -> float:
        """Given two images, perform image pre-processing and returns the probability that
        these 2 images are similar.

        Returns:
            float
        """
        im_1 = preprocess_inputs(im_1)
        im_2 = preprocess_inputs(im_2)
        
        inputs = tf.stack([im_1, im_2], axis=0)
        
        inputs = tf.expand_dims(inputs, axis=0)
        
        return self.model(inputs)   # type: ignore
    