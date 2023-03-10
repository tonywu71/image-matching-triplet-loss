from dataloader.data_generator import DataGenerator

def test_DataGenerator():
    try:
        data_generator = DataGenerator(
            directory="tiny-imagenet-200/train/",
            batch_size=128,
            image_size=(64, 64),
            shuffle=True,
            seed=0,
            val_split=0.2,
            test_split=0.1
        )
    except:
        raise
