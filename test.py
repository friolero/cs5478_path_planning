import sys

import ipdb

from map import ImageMap2D
from utils import vis_map


def test_image_map(image_fn):
    map = ImageMap2D(image_fn)
    vis_map(map)


if __name__ == "__main__":

    test_image_map(sys.argv[1])
    ipdb.set_trace()
