import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa


TF_HUB_MODEL_URL = {
    "efficientnet": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2"
}


def load_and_compile_model() -> tf.keras.Model:
    feature_layer = hub.KerasLayer(TF_HUB_MODEL_URL["efficientnet"])

    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())
    
    return model
