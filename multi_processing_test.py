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
import geopandas as gpd
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
# stackoverflow.com/ques


def multi_process(func, args, process_cnt = 6):



    print("Done")


if __name__ == "__main__":

    multi_process()