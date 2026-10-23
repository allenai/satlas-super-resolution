import unittest
from unittest.mock import patch

import numpy as np

from ssr.utils.infer_utils import format_s2naip_data


class FormatS2NaipDataTest(unittest.TestCase):
    @patch("ssr.utils.infer_utils.random.sample", side_effect=lambda population, count: list(population)[:count])
    def test_zero_channel_alone_does_not_make_an_image_invalid(self, _sample):
        frames = np.full((2, 32, 32, 3), 100, dtype=np.uint8)
        frames[0, 0, 0] = [0, 20, 30]  # Not a fully black pixel.
        frames[1] = 200

        tensor, _ = format_s2naip_data(frames.reshape(64, 32, 3), 1, "cpu")

        self.assertAlmostEqual(tensor[0, 0, 1, 1].item(), 100 / 255, places=6)


if __name__ == "__main__":
    unittest.main()
