import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


logger = logging.getLogger(__name__)
EMBEDDINGS_DIRPATH = Path("embeddings")


def save_embeddings_for_tf_projector(model: tf.keras.Model, ds_val: tf.data.Dataset):
    # Evaluate the network:
    embeddings = model.predict(ds_val)

    # Save test embeddings for visualization in projector:
    vecs_filepath = EMBEDDINGS_DIRPATH / "vecs.tsv"
    np.savetxt(vecs_filepath, embeddings, delimiter='\t')
    logger.info(f"Embeddings successfully saved at `{vecs_filepath}`.")

    # Save metadata for projector:
    meta_filepath = EMBEDDINGS_DIRPATH / "meta.tsv"
    with open(meta_filepath, 'w', encoding='utf-8') as f:
        for img, labels in tfds.as_numpy(ds_val): # type: ignore
            [f.write(str(x) + "\n") for x in labels]

    logger.info(f"Metadata successfully saved at `{meta_filepath}`.")
    
    return
