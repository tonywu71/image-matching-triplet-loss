import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

from models.preprocessing import get_image_augmentation_layer

TF_HUB_MODELS = {
    "resnet50": "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
    "efficientnet": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2"
}
IMAGE_SIZE = (224, 224)


def get_feature_extractor(model_name: str) -> tf.keras.Model:
    if model_name in TF_HUB_MODELS:
        feature_extractor = hub.KerasLayer(TF_HUB_MODELS[model_name], trainable=False)
    else:
        raise ValueError('Only "efficientnet" and "resnet50" are supported.')
    return feature_extractor


def load_and_compile_model(model_name: str, dropout: float=0., image_augmentation: bool=False) -> tf.keras.Model:
    feature_extractor = get_feature_extractor(model_name)
    
    list_layers = []
    
    if image_augmentation:
        list_layers.append(get_image_augmentation_layer())
    
    list_layers.extend([
        feature_extractor,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings for triplet loss
    ])
    
    model_name = tf.keras.Sequential(list_layers)

    model_name.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tfa.losses.TripletSemiHardLoss()
    )
    
    return model_name
