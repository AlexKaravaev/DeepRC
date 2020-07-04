import unittest
import pathlib
from rc_car.datastore.donkeycar_dataloader import DonkeyCarDataset

class TestDonkeyDataset(unittest.TestCase):

    dataset = DonkeyCarDataset([pathlib.Path('tests/data/reverse_drive'),
                                pathlib.Path('tests/data/usual_drive')])

    def test_loading_len(self):
        print("Running tests for loading training data")
        self.assertEqual(len(self.dataset), 6)

    def test_loading_single_data(self):
        print("Running tests for loading image and goal vector")
        self.assertEqual(self.dataset[2]['throttle'],0)
        self.assertEqual(self.dataset[2]['angle'],0)

if __name__ == "__main__":
    unittest.main()    