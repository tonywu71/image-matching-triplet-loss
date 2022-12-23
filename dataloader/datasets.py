import logging
from pathlib import Path
from typing import Tuple
import tensorflow as tf
import tensorflow_addons as tfa

from dataloader.utils import plot_classes, plot_from_one_class
from models.feature_model import IMAGE_SIZE


logger = logging.getLogger(__name__)


def preprocessing_function(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Preprocessing function to use with map for an unbatched tf.data.Dataset object.

    Args:
        x (tf.Tensor)
        y (tf.Tensor)

    Returns:
        Tuple[tf.Tensor, tf.Tensor]
    """
    x = tf.image.resize(x, size=IMAGE_SIZE)
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    
    return x, y


class DataGenerator():
    def __init__(self,
                 directory: str,
                 batch_size: int,
                 image_size: Tuple[int, int],
                 shuffle: bool=True,
                 seed: int=0,
                 validation_split: float=0.2
                 ) -> None:
        
        assert Path(directory).is_dir(), f"`{self.directory}` is not a directory."
        
        # --- Storing basic attributes ---
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.seed = seed

        # --- Generate datasets ---
        self.train_unbatched, self.val_unbatched = tf.keras.utils.image_dataset_from_directory(
            directory=self.directory,
            batch_size=None, # type: ignore
            image_size=self.image_size,
            shuffle=self.shuffle,
            seed=self.seed,
            validation_split=self.validation_split,
            subset="both"
        )
        
        # --- Preprocessing ---
        self.train = self.train_unbatched.map(preprocessing_function)
        self.val = self.val_unbatched.map(preprocessing_function)
        
        # --- Optimize pipeline ---
        self.train = self.train.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.val = self.val.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        logger.info(f"Successfully generated a DataGenerator object from `{directory}`")
        
        return
    
    
    def get_val_one_class(self, class_id: int):
        return self.val.filter(lambda x, y: y==class_id)
    
    
    def plot_classes(self):
        plot_classes(ds=self.train)
    
    
    def plot_from_one_class(self, class_id: int):
        plot_from_one_class(ds=self.train, class_id=class_id)
