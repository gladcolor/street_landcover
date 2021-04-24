import unittest
import measure_width as mw

import shapely
import cv2
import numpy as np
import math
import imutils
import fiona
import sys
import pandas as pd
sys.path.append(r'D:\Code\StreetView\gsv_pano')
sys.path.append(r'E:\USC_OneDrive\OneDrive - University of South Carolina\StreetView\gsv_pano')

from pano import GSV_pano
from PIL import Image
import time
import os
import glob
import multiprocessing as mp
# from label_centerlines import get_centerline
from shapely.geometry import Point, Polygon, mapping, LineString, MultiLineString
from shapely import speedups
speedups.disable()
# stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer


class MyTestCase(unittest.TestCase):


    def test_check_touched(self):
        # img_file = r'D:\Research\sidewalk_wheelchair\DC_DOMs\XzB9K8BHqMpZVKZR-E9MBw_DOM_0.05.tif'
        img_file = r'D:\Research\sidewalk_wheelchair\DC_DOMs\zhJf6faax0FFmR67jKzoKA_DOM_0.05.tif'
        img_pil = Image.open(img_file)
        img_np = np.array(np.array(img_pil))
        img_cv = cv2.merge([img_np])
        col = 49
        row = 203
        # col = 192   # False
        # row = 137
        category_list = [58, 255]  # car: 58, other: 255
        mask_np = mw.create_mask(img_np, category_list)
        is_touched = mw.check_touched(col, row, mask_np)
        self.assertEqual(is_touched, True)

    # def test_create_mask(self):
    #     img_file = r'D:\Research\sidewalk_wheelchair\DC_DOMs\XzB9K8BHqMpZVKZR-E9MBw_DOM_0.05.tif'
    #     img_pil = Image.open(img_file)
    #     img_np = np.array(np.array(img_pil))
    #     img_cv = cv2.merge([img_np])
    #     category_list = [58, 255]  # car: 58, other: 255
    #     img_file_catetory_sum = 96284
    #     mask_np = mw.create_mask(img_np, category_list)
    #     img_mask = mask_np.astype(np.uint8)
    #     img_mask = np.where(img_mask > 0, 255, 0)
    #     # cv2.imshow("mask", cv2.merge([img_mask.astype(np.uint8)]))
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     self.assertEqual(mask_np.sum(), img_file_catetory_sum)


if __name__ == '__main__':
    unittest.main()
