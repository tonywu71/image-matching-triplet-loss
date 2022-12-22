import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

TF_HUB_MODELS = {
    "resnet50": "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
    "efficientnet": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2"
}
IMAGE_SIZE = (224, 224)


def get_feature_extractor(model: str) -> tf.keras.Model:
    if model in TF_HUB_MODELS:
        feature_extractor = hub.KerasLayer(TF_HUB_MODELS[model], trainable=False)
    else:
        raise ValueError('Only "efficientnet" and "resnet50" are supported.')
    return feature_extractor


def load_and_compile_model(model: str) -> tf.keras.Model:
    feature_extractor = get_feature_extractor(model)
    
    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings for triplet loss
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tfa.losses.TripletSemiHardLoss()
    )
    
    return model
