from typing import List, Optional, Union
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

from models.preprocessing import get_image_augmentation_layer
from models.ff_block import FFBlock


TF_HUB_MODELS = {
    "resnet50": "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
    "efficientnet": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2"
}
IMAGE_SIZE_EFFICIENTNET = (224, 224)


def get_feature_extractor(model_name: str) -> tf.keras.Model:
    if model_name in TF_HUB_MODELS:
        feature_extractor = hub.KerasLayer(TF_HUB_MODELS[model_name], trainable=False)
    else:
        raise ValueError('Only "efficientnet" and "resnet50" are supported.')
    return feature_extractor


def load_and_compile_model(model_name: str,
                           embedding_dim: int=256,
                           intermediate_linear_units: Optional[Union[int, List[int]]]=None,
                           dropout: float=0.,
                           image_augmentation: bool=False) -> tf.keras.Model:
    
    if intermediate_linear_units is None:
        intermediate_linear_units = []
    else:
        if type(intermediate_linear_units) is int:
            intermediate_linear_units = [intermediate_linear_units] # convert to a list of 1 element
        elif type(intermediate_linear_units) is list:
            intermediate_linear_units = intermediate_linear_units
        else:
            raise TypeError("`intermediate_linear_units` should be an integer or a list of integer.")
    
    feature_extractor = get_feature_extractor(model_name)
    
    list_layers = []
    
    if image_augmentation:
        list_layers.append(get_image_augmentation_layer())
    
    list_layers.extend([
        feature_extractor,
        tf.keras.layers.Flatten()
    ])
    
    for units in intermediate_linear_units:
        list_layers.append(FFBlock(units=units, dropout=dropout))

    list_layers.extend([
        tf.keras.layers.Dense(units=embedding_dim),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings for triplet loss
    ])
    
    model_name = tf.keras.Sequential(list_layers, name="feature_model")

    model_name.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tfa.losses.TripletSemiHardLoss()
    )
    
    return model_name
