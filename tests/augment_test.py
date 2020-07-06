import unittest
import pathlib
from rc_car.datastore.donkeycar_dataloader import DonkeyCarDataset

class TestAugmentImage(unittest.TestCase):

    aug_dataset = DonkeyCarDataset([pathlib.Path('tests/data/reverse_drive'),
                                pathlib.Path('tests/data/usual_drive')],
                                augment=True)

    def test_augment(self):
        # For now only 3 augmentation per image
        self.assertIsNot(0, len(self.aug_dataset))

if __name__=="__main__":
    unittest.main()