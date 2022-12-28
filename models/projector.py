import logging
from pathlib import Path

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

from train import IMAGE_SIZE_DATASET


logger = logging.getLogger(__name__)


EMBEDDINGS_DIRPATH = Path("embeddings")


def save_embeddings_for_tf_projector(model: tf.keras.Model,
                                     ds_val: tf.data.Dataset,
                                     experiment_name: str):
    savedir = EMBEDDINGS_DIRPATH / f"{experiment_name}"
    savedir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate the network:
    embeddings = model.predict(ds_val)

    # Save test embeddings for visualization in projector:
    vecs_filepath = savedir / "vecs.tsv"
    np.savetxt(vecs_filepath, embeddings, delimiter='\t')
    logger.info(f"Embeddings successfully saved at `{vecs_filepath}`.")

    # Save metadata for projector:
    meta_filepath = savedir / "meta.tsv"
    with open(meta_filepath, 'w', encoding='utf-8') as f:
        for img, labels in tfds.as_numpy(ds_val): # type: ignore
            [f.write(str(x) + "\n") for x in labels]

    logger.info(f"Metadata successfully saved at `{meta_filepath}`.")
    
    return


def generate_sprite(ds_val: tf.data.Dataset):
    images_pil = []

    for x, y in ds_val: 
        img_pil = Image.fromarray(x.numpy().astype(np.uint8))
        images_pil.append(img_pil)

    one_square_size = int(np.ceil(np.sqrt(len(images_pil))))
    master_width = IMAGE_SIZE_DATASET[0] * one_square_size
    master_height = IMAGE_SIZE_DATASET[1] * one_square_size
    spriteimage = Image.new(
        mode="RGBA",
        size=(master_width, master_height),
        color=(0,0,0,0) # fully transparent
    )

    for count, image in enumerate(images_pil):
        div, mod = divmod(count, one_square_size)
        w_loc = IMAGE_SIZE_DATASET[0] * mod
        h_loc = IMAGE_SIZE_DATASET[1] * div
        spriteimage.paste(image, (w_loc, h_loc))

    spriteimage.convert("RGB").save("sprite.jpg", transparency=0)
