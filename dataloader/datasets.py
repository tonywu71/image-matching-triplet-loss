from typing import Tuple
import tensorflow as tf

from dataloader.utils import plot_classes, plot_from_one_class


class DataGenerator():
    def __init__(self,
                 directory: str,
                 batch_size: int,
                 image_size: Tuple[int, int],
                 shuffle: bool=True,
                 seed: int=0,
                 validation_split: float=0.2,
                 ) -> None:
        
        # --- Storing basic attributes ---
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.seed = seed

        # --- Generate datasets ---
        self.train, self.val = tf.keras.utils.image_dataset_from_directory(
            directory=directory,
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=self.shuffle,
            seed=self.seed,
            validation_split=self.validation_split,
            subset="both"
        )
    
    
    def plot_classes(self):
        plot_classes(ds=self.train)
    
    
    def plot_from_one_class(self, class_id: int):
        plot_from_one_class(ds=self.train, class_id=class_id)
