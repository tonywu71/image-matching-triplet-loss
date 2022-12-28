from typing import Tuple, Callable
import tensorflow as tf
from dataloader.data_generator import DataGenerator


def get_generator_2_combination_from_datasets(dataset: tf.data.Dataset) -> Callable:
    ds_1 = dataset
    ds_2 = dataset
    
    def gen_2_combination_from_datasets():
        for x_1 in ds_1:
            for x_2 in ds_2:
                yield x_1, x_2
    
    return gen_2_combination_from_datasets


def match_mapping(x, y) -> Tuple[tf.Tensor, tf.Tensor]:
    (x_1, cls_1), (x_2, cls_2) = x, y
    return (tf.stack([x_1, x_2], axis=0), tf.cast(cls_1 == cls_2, dtype=tf.uint8))  # type: ignore


def get_pairwise_dataset(data_generator: DataGenerator, image_size: Tuple[int, int]) -> tf.data.Dataset:
    """Get the pairwise dataset for our end-to-end model. Note that the output Dataset is not batched.

    Args:
        data_generator (DataGenerator)
        image_size (Tuple[int, int])

    Returns:
        tf.data.Dataset
    """
    ds_pairs = tf.data.Dataset.from_generator(get_generator_2_combination_from_datasets(
        dataset=data_generator.val_unbatched),
        output_signature=(
              (
                  tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),  # type: ignore
                  tf.TensorSpec(shape=(), dtype=tf.uint8)  # type: ignore
              ),
              (
                  tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),  # type: ignore
                  tf.TensorSpec(shape=(), dtype=tf.uint8)  # type: ignore
              )
        ))
    
    ds_pairs = ds_pairs.map(match_mapping)
    
    return ds_pairs
