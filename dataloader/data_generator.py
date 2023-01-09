import logging
from pathlib import Path
from typing import Tuple
import tensorflow as tf

from dataloader.utils import plot_classes, plot_from_one_class
from models.feature_model import IMAGE_SIZE_EFFICIENTNET

BUFFER_SIZE = 1000

logger = logging.getLogger(__name__)


def preprocess_inputs(x: tf.Tensor) -> tf.Tensor:
    """Resize the image and applies the specific preprocesing related to the
    model_feature pre-trained layer.

    Args:
        x (tf.Tensor)

    Returns:
        tf.Tensor
    """
    x = tf.image.resize(x, size=IMAGE_SIZE_EFFICIENTNET)
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    return x


def _preprocessing_function(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Preprocessing function to use with map for an unbatched tf.data.Dataset object.

    Args:
        x (tf.Tensor)
        y (tf.Tensor)

    Returns:
        Tuple[tf.Tensor, tf.Tensor]
    """
    
    x = preprocess_inputs(x)
    
    return x, y


class DataGenerator():
    def __init__(self,
                 directory: str,
                 batch_size: int,
                 image_size: Tuple[int, int],
                 shuffle: bool=True,
                 seed: int=0,
                 val_split: float=0.2,
                 test_split: float=0.1
                 ) -> None:
        
        assert Path(directory).is_dir(), f"`{directory}` is not a directory."
        
        
        # --- Storing basic attributes ---
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.val_split = val_split
        self.test_split = test_split
        self.val_and_test_split = self.val_split + self.test_split
        self.seed = seed


        # --- Generate datasets ---
        self.train_raw, self.val_and_test_raw = tf.keras.utils.image_dataset_from_directory(
            directory=self.directory,
            batch_size=None, # type: ignore
            image_size=self.image_size,
            shuffle=self.shuffle,
            seed=self.seed,
            validation_split=self.val_and_test_split,
            subset="both"
        )
        
        self.train_size = tf.data.experimental.cardinality(self.train_raw)
        self.val_and_test_size = tf.data.experimental.cardinality(self.val_and_test_raw).numpy()
        self.val_raw = self.val_and_test_raw.take(int(self.val_and_test_size * (self.val_split / self.val_and_test_split)))
        self.test_raw = self.val_and_test_raw.skip(int(self.val_and_test_size * (self.val_split / self.val_and_test_split)))\
                                             .take(int(self.val_and_test_size * (self.test_split / self.val_and_test_split)))
        
        
        # --- Preprocessing ---
        self.train_unbatched = self.train_raw.map(_preprocessing_function)
        self.val_unbatched = self.val_raw.map(_preprocessing_function)
        self.test_unbatched = self.test_raw.map(_preprocessing_function)
        
        
        # --- Optimize pipeline ---
        self.train = self.train_unbatched.shuffle(buffer_size=BUFFER_SIZE).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.val = self.val_unbatched.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.test = self.test_unbatched.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        logger.info(f"Successfully generated a DataGenerator object from `{directory}`")
        
        return
    
    
    def get_val_one_class(self, class_id: int):
        return self.val.filter(lambda x, y: y==class_id)
    
    
    def plot_classes(self):
        plot_classes(ds=self.train)
    
    
    def plot_from_one_class(self, class_id: int):
        plot_from_one_class(ds=self.train, class_id=class_id)
