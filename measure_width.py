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
# stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer

LINE_COUNT = 160


def cv_img_rotate_bound(image, angle, flags=cv2.INTER_NEAREST):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=flags)

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    '''
    rows, cols = np.where(x == 1) # .T sets Fortran order down-then-right
    run_lengths = []
    run_rows = []
    prev = -2
    for idx, b in enumerate(cols):
        if (b != prev+1):
            run_lengths.extend((b, 0))
            run_rows.extend((rows[idx],))
        # else:  #(b < prev),  new line
        #     pass
        run_lengths[-1] += 1
        prev = b
    return run_lengths, run_rows



def cal_witdh_from_list(img_list, crs_local=6847):

    # img_path = r'ZXyk9lKhL5siKJglQPqfMA_DOM_0.05.tif'
    # img_list = [img_path]
    total_cnt = len(img_list)
    start_time = time.perf_counter()
    # print("total_cnt: ", os.getpid(),  total_cnt)
    cnt = 0

    yaw_csv_file = r'H:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\near_compassA.csv'
    # yaw_csv_file = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\near_compassA.csv'
    df_yaw = pd.read_csv(yaw_csv_file)
    df_yaw = df_yaw.set_index("panoId")



    while len(img_list) > 0:
        try:
            img_path = img_list.pop()
            cnt = total_cnt - len(img_list)
            cnt += 1
            cal_witdh(img_path, df_yaw, crs_local=crs_local)
            if cnt % 100 == 0:
                print(cnt, img_path)
        except Exception as e:
            print("Error in cal_witdh_from_list():", e)
            continue

def read_worldfile(file_path):
    try:
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        lines = [line[:-1] for line in lines]
        resolution = float(lines[0])
        upper_left_x = float(lines[4])
        upper_left_y = float(lines[5])
        return resolution, upper_left_x, upper_left_y
    except Exception as e:
        print("Error in read_worldfile(), return Nones:", e)
        return None, None, None


def cal_witdh(img_path, df_yaw, crs_local=6847):


    basename = os.path.basename(img_path)
    dirname = os.path.dirname(img_path)
    panoId = basename[:22]
    json_file = os.path.join(dirname, panoId + '.json')
    pano1 = GSV_pano(json_file=json_file, crs_local=6847)
    # print(pano1.jdata)


    pano_yaw_deg = pano1.jdata['Projection']['pano_yaw_deg']
    street_yaw = df_yaw.loc[panoId]['CompassA']
    if abs(street_yaw - pano_yaw_deg) > 150:
        pano_yaw_deg = street_yaw + 180
    else:
        pano_yaw_deg = street_yaw

    img_pil = Image.open(img_path)
    img_np = np.array(img_pil)
    # im_cv = cv2.imread(img_path)
    target_ids = [8, 12]

    class_idx = img_np
    target_np = np.zeros(img_np.shape)
    for i in target_ids:
        target_np = np.logical_or(target_np, class_idx == i)



    # AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)

    morph_kernel_open  = (5, 5)
    morph_kernel_close = (10, 10)
    g_close = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_close)
    g_open  = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_open)

    target_np = target_np.astype(np.uint8)



    yaw_deg =  -pano_yaw_deg
    # print("yaw_deg:", yaw_deg)

    cv2_closed = cv2.morphologyEx(target_np, cv2.MORPH_CLOSE, g_close) # fill small gaps
    cv2_opened = cv2.morphologyEx(cv2_closed, cv2.MORPH_OPEN, g_open)

    cv2_opened = np.where(cv2_opened == 0, 0, 255).astype(np.uint8)

    opened_color = cv2.merge((cv2_opened, cv2_opened, cv2_opened))
    img_rotated = cv_img_rotate_bound(cv2_opened, yaw_deg)

    # cv2.imshow("img_rotated", np.where(img_rotated == 0, 0, 255).astype(np.uint8))


    # draw lines
    line_cnt = LINE_COUNT
    img_h, img_w = img_rotated.shape
    start_x = 0
    end_x = img_w -1
    interval = int(img_h / line_cnt)
    line_ys = range(interval, img_h, interval)
    line_thickness = 1


    to_RLE = img_rotated[line_ys]
    run_lengths, run_rows = rle_encoding(to_RLE)
    # print("rung_lengths, rows:\n", run_lengths[::2], "\n", run_rows, "\n", run_lengths[1::2])


    # cv2.imshow("Raw image", AOI.astype(np.uint8))

    # cv2_closed = np.where(cv2_closed == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("cv2_closed", cv2_closed.astype(np.uint8))

    # for y in line_ys:
    #     cv2.line(opened_color, (start_x, y), (end_x, y), (0, 255, 0), thickness=line_thickness)

    angle_deg = 0
    angle_rad = math.radians(angle_deg)
    lengths = run_lengths[1::2]
    lengths = np.array(lengths)
    new_lengths = lengths.copy()
    new_run_cols = run_lengths[::2].copy()
    pen_lengths = lengths * math.cos(angle_rad)
    to_x = (pen_lengths * math.cos(angle_rad)).astype(int)
    to_y = (pen_lengths * math.sin(angle_rad)).astype(int)
    max_width_meter = 30
    pix_resolution = 0.05
    max_width_pix = int(max_width_meter / pix_resolution)
    for idx, col in enumerate(run_lengths):
        if idx % 2 == 0:
            idx2 = int(idx / 2)
            row = run_rows[idx2] * interval + interval - 1
            radius = 5
            # print(row, col)
            length = run_lengths[idx + 1]
            new_run_cols[idx2] = col
            if (length > max_width_pix) and (idx2 > 0):
                length = new_lengths[idx2 - 1]
                new_run_cols[idx2] = new_run_cols[idx2 - 1]
                print("long length!")
                # print("length, new_lengths[idx2], max_width_pix:", length, new_lengths[idx2], max_width_pix)

            new_lengths[idx2] = length
            new_run_cols[idx2] = new_run_cols[idx2]

            # print("col, new_run_cols[idx2]:", col, new_run_cols[idx2])
            col = new_run_cols[idx2]
            to_x[idx2] = new_lengths[idx2]
            # to_y[idx2] = row

            end_x = col + to_x[idx2]
            end_y = row + to_y[idx2]

            # cv2.line(opened_color, (col, row), (end_x, end_y), (0, 0, 255), thickness=line_thickness)
            # cv2.circle(opened_color, (col, row), radius, (0, 255, 0), line_thickness)


    # cv2.imshow("cv2_opened", opened_color)

    # find contour
    raw_contours, hierarchy = cv2.findContours(img_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # contours = [np.squeeze(cont) for cont in raw_contours]
    #
    # start_time = time.perf_counter()
    # centerlines = get_polygon_centline(contours[0:])
    # print("Time used to get centerline: ", time.perf_counter() - start_time)
    # p = Polygon([[0, 0], [2, 2], [2, 0]])
    # pts = centerlines.reshape((-1,1,2))
    # all_pair_list = seg_contours(raw_contours[5:6], opened_color)
    img_rotated_color = cv2.merge((img_rotated, img_rotated, img_rotated))

    category_list = [58, 255]
    raw_img_rotated = cv_img_rotate_bound(img_np, yaw_deg)
    # cv2.imshow("raw_img_rotated", raw_img_rotated)

    car_mask_np = create_mask(raw_img_rotated, category_list=category_list)

    # for l in centerlines:
    #     pts = l.coords.xy
    #     pts = np.array(pts).T.reshape((-1, 1, 2)).astype(np.int32)
    #     # pts = np.array(pts).T.
    #     cv2.polylines(img_rotated_color, [pts], False, (255), 2)

    # all_pair_list = seg_contours(raw_contours[:], img_rotated_color)
    all_pair_list = seg_contours(raw_contours[:], car_mask_np, img_rotated)
    # print(all_pair_list)
    end_points = np.zeros((len(all_pair_list) * 2, 2))

    for idx, pair in enumerate(all_pair_list):
        x = int(pair[1])
        y = int(pair[2])
        end_x = int(pair[3])
        end_y = int(pair[4])
        cv2.line(img_rotated_color, (x, y), (end_x, end_y), (255, 0, 0), thickness=line_thickness)
        end_points[idx * 2] = np.array([x, y])
        end_points[idx * 2 + 1] = np.array([end_x, end_y])

    # cv2.imshow("img_rotated added pairs", img_rotated.astype(np.uint8))

    end_points = np.hstack((end_points, np.ones((end_points.shape[0], 1))))
    tx = img_rotated.shape[0] / 2
    ty = img_rotated.shape[1] / 2
    # print("tx, ty:", tx, ty)
    end_points_transed = points_2D_translation(end_points, tx, ty)
    # print("end_points_transed:", end_points_transed[0])

    tx = target_np.shape[0] / 2
    ty = target_np.shape[1] / 2
    #
    # print("tx, ty:", tx, ty)

    end_points_rotated = points_2D_rotated(end_points_transed, yaw_deg)
    # print("end_points_rotated:", end_points_rotated[0])
    end_points_transed = points_2D_translation(end_points_rotated, -tx, ty)

    # print("final end_points_transed:", end_points_transed.astype(int))
    line_thickness = 1
    radius = 2
    raw_AOI_color = np.where(target_np == 0, 0, 255).astype(np.uint8)
    raw_AOI_color = cv2.merge((raw_AOI_color, raw_AOI_color, raw_AOI_color))
    line_cnt = len(end_points_transed)
    line_cnt = int(line_cnt)
    end_points_transed = end_points_transed.astype(int)

    dom_path = r'AZK1jDGIZC1zmuooSZCzEg.tif'
    dom_path = r'-ft2bZI1Ial4C6N_iwmmvw_DOM_0.05.tif'
    # im_dom = cv2.imread(dom_path)
    for idx in range(0, line_cnt, 2):
        col = end_points_transed[idx][0]
        row = end_points_transed[idx][1]
        # to_y[idx2] = row

        end_x = end_points_transed[idx + 1][0]
        end_y = end_points_transed[idx + 1][1]
        # cv2.line(opened_color, (col, row), (end_x , end_y), (0, 0, 255), thickness=line_thickness)
        # cv2.line(im_dom,       (col, row), (end_x , end_y), (0, 0, 255), thickness=line_thickness)
        # cv2.circle(raw_AOI_color, (end_x, end_y), radius, (0, 255, 0), line_thickness)
        # cv2.circle(raw_AOI_color, (col, row), radius, (0, 255, 0), line_thickness)

    # write txt
    saved_path = r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # saved_path = r'H:\Research\sidewalk_wheelchair\DC_DOMs_measuremens'
    new_name = os.path.join(saved_path, f'{panoId}_widths.csv')
    worldfile_ext = img_path[-3] + img_path[-1] + 'w'
    worldfile_path = img_path[:-3] + worldfile_ext
    wf_resolution, wf_x, wf_y = read_worldfile(worldfile_path)
    # print("wf_resolution, wf_x, wf_y:", wf_resolution, wf_x, wf_y)
    f = open(new_name, 'w')
    f.writelines('panoId,contour_num,center_x,center_y,length,start_x,start_y,end_x,end_y,cover_ratio,is_touched\n')
    # print("new_name:", new_name)
    for idx in range(0, line_cnt, 2):
        col = end_points_transed[idx][0] * wf_resolution + wf_x
        row = wf_y - end_points_transed[idx][1] * wf_resolution
        # to_y[idx2] = row

        end_x = end_points_transed[idx + 1][0] * wf_resolution + wf_x
        end_y = wf_y - end_points_transed[idx + 1][1] * wf_resolution

        center_x = (end_x + col) / 2
        center_y = (end_y + row) / 2

        idx2 = int(idx/2)
        length = all_pair_list[idx2][3] - all_pair_list[idx2][1]
        contour_num = all_pair_list[idx2][0]
        length = length * wf_resolution
        cover_ratio = all_pair_list[idx2][5]
        is_touched = all_pair_list[idx2][6]

        f.writelines(f'{panoId},{contour_num},{center_x:.3f},{center_y:.3f},{length:.3f},{col:.3f},{row:.3f},{end_x:.3f},{end_y:.3f},{cover_ratio:.3f},{int(is_touched)}\n')
        # print("center_x, center_x:", f'{center_x},{center_y},{length},{col},{row},{end_x},{end_y}\n')
        # f.writelines(f'{center_x},{center_y},{length},{col},{row},{end_x},{end_y}\n')
    f.close()

    cv2.imshow("opened_color", opened_color)
    # cv2.imshow("im_dom", im_dom)
    # cv2.imshow("img_rotated_color", img_rotated_color)

    # end_points_transed = end_points_transed[:, 0:2]


    # rotated = imutils.rotate_bound(opened_color, -45)....


    # to_RLE = np.where(to_RLE == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("to_RLE", to_RLE.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cal_witdh0(img_path):
    # img_path = r'AZK1jDGIZC1zmuooSZCzEg_DOM.tif'
    # # img_path = r'Ld-CMATy8ZxKap6VAtZTEg_DOM.tif'
    # # img_path = r'v-VR9FB7kCxU1eDLEtFiJQ_DOM.tif'
    # img_path = r'-0D29S37SnmRq9Dju9hkqQ_DOM.tif'
    img_path = r'ZXyk9lKhL5siKJglQPqfMA_DOM_0.05.tif'

    # img_path = r'H:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_DOMs\-ft2bZI1Ial4C6N_iwmmvw_DOM_0.05.tif'


    panoId_2019 = r'ZXyk9lKhL5siKJglQPqfMA'
    pano1 = GSV_pano(panoId=panoId_2019, crs_local=6847, saved_path=os.getcwd())
    # print(pano1.jdata)
    pano_yaw_deg = pano1.jdata['Projection']['pano_yaw_deg']

    target_ids = [244]
    im_cv = cv2.imread(img_path)

    red_channel = im_cv[:, :, 2]
    class_idx = red_channel
    AOI = np.zeros((len(class_idx), len(class_idx)))
    for i in target_ids:
        AOI = np.logical_or(AOI, class_idx == i)
        # print(AOI)
    # AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)
    # print(AOI)

    morph_kernel_open  = (5, 5)
    morph_kernel_close = (10, 10)
    g_close = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_close)
    g_open  = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_open)

    raw_AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)

    # cv2.imshow("Raw image", np.where(raw_AOI == 0, 0, 255).astype(np.uint8) )

    # yaw_deg =  226.4377593994141 - 90
    # yaw_deg =  92.53645324707031
    yaw_deg =  -pano_yaw_deg
    # print("yaw_deg:", yaw_deg)
    #
    cv2_closed = cv2.morphologyEx(raw_AOI, cv2.MORPH_CLOSE, g_close) # fill small gaps
    cv2_opened = cv2.morphologyEx(cv2_closed, cv2.MORPH_OPEN, g_open)

    cv2_opened = np.where(cv2_opened == 0, 0, 255).astype(np.uint8)

    opened_color = cv2.merge((cv2_opened, cv2_opened, cv2_opened))
    img_rotated = imutils.rotate_bound(cv2_opened, yaw_deg)

    # draw lines
    line_cnt = 160
    img_h, img_w = img_rotated.shape
    start_x = 0
    end_x = img_w -1
    interval = int(img_h / line_cnt)
    line_ys = range(interval, img_h, interval)
    line_thickness = 1


    to_RLE = img_rotated[line_ys]
    run_lengths, run_rows = rle_encoding(to_RLE)
    # print("rung_lengths, rows:\n", run_lengths[::2], "\n", run_rows, "\n", run_lengths[1::2])

    AOI = np.where(AOI == 0, 0, 255).astype(np.uint8)

    # cv2.imshow("Raw image", AOI.astype(np.uint8))

    # cv2_closed = np.where(cv2_closed == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("cv2_closed", cv2_closed.astype(np.uint8))

    for y in line_ys:
        cv2.line(opened_color, (start_x, y), (end_x, y), (0, 255, 0), thickness=line_thickness)

    angle_deg = 0
    angle_rad = math.radians(angle_deg)
    lengths = run_lengths[1::2]
    lengths = np.array(lengths)
    new_lengths = lengths.copy()
    new_run_cols = run_lengths[::2].copy()
    pen_lengths = lengths * math.cos(angle_rad)
    to_x = (pen_lengths * math.cos(angle_rad)).astype(int)
    to_y = (pen_lengths * math.sin(angle_rad)).astype(int)
    max_width_meter = 30
    pix_resolution = 0.05
    max_width_pix = int(max_width_meter / pix_resolution)
    for idx, col in enumerate(run_lengths):
        if idx % 2 == 0:
            idx2 = int(idx / 2)
            row = run_rows[idx2] * interval + interval - 1
            radius = 5
            # print(row, col)
            length = run_lengths[idx + 1]
            new_run_cols[idx2] = col
            if (length > max_width_pix) and (idx2 > 0):
                length = new_lengths[idx2 - 1]
                new_run_cols[idx2] = new_run_cols[idx2 - 1]
                print("long length!")
                # print("length, new_lengths[idx2], max_width_pix:", length, new_lengths[idx2], max_width_pix)

            new_lengths[idx2] = length
            new_run_cols[idx2] = new_run_cols[idx2]

            # print("col, new_run_cols[idx2]:", col, new_run_cols[idx2])
            col = new_run_cols[idx2]
            to_x[idx2] = new_lengths[idx2]
            # to_y[idx2] = row

            end_x = col + to_x[idx2]
            end_y = row + to_y[idx2]

            # cv2.line(opened_color, (col, row), (end_x, end_y), (0, 0, 255), thickness=line_thickness)
            # cv2.circle(opened_color, (col, row), radius, (0, 255, 0), line_thickness)


    # cv2.imshow("cv2_opened", opened_color)

    # find contour
    raw_contours, hierarchy = cv2.findContours(img_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # contours = [np.squeeze(cont) for cont in raw_contours]
    #
    # start_time = time.perf_counter()
    # centerlines = get_polygon_centline(contours[0:])
    # print("Time used to get centerline: ", time.perf_counter() - start_time)
    # p = Polygon([[0, 0], [2, 2], [2, 0]])
    # pts = centerlines.reshape((-1,1,2))
    # all_pair_list = seg_contours(raw_contours[5:6], opened_color)
    img_rotated_color = cv2.merge((img_rotated, img_rotated, img_rotated))

    # for l in centerlines:
    #     pts = l.coords.xy
    #     pts = np.array(pts).T.reshape((-1, 1, 2)).astype(np.int32)
    #     # pts = np.array(pts).T.
    #     cv2.polylines(img_rotated_color, [pts], False, (255), 2)

    all_pair_list = seg_contours(raw_contours[:], img_rotated_color)
    end_points = np.zeros((len(all_pair_list) * 2, 2))
    for idx, pair in enumerate(all_pair_list):
        x = int(pair[1])
        y = int(pair[2])
        end_x = int(pair[3])
        end_y = int(pair[4])
        cv2.line(img_rotated_color, (x, y), (end_x, end_y), (255, 0, 0), thickness=line_thickness)
        end_points[idx * 2] = np.array([x, y])
        end_points[idx * 2 + 1] = np.array([end_x, end_y])

    # cv2.imshow("img_rotated added pairs", img_rotated.astype(np.uint8))

    end_points = np.hstack((end_points, np.ones((end_points.shape[0], 1))))
    tx = img_rotated.shape[0] / 2
    ty = img_rotated.shape[1] / 2
    # print("tx, ty:", tx, ty)
    end_points_transed = points_2D_translation(end_points, tx, ty)
    # print("end_points_transed:", end_points_transed[0])

    tx = raw_AOI.shape[0] / 2
    ty = raw_AOI.shape[1] / 2
    #
    # print("tx, ty:", tx, ty)

    end_points_rotated = points_2D_rotated(end_points_transed, yaw_deg)
    # print("end_points_rotated:", end_points_rotated[0])
    end_points_transed = points_2D_translation(end_points_rotated, -tx, ty)

    # print("final end_points_transed:", end_points_transed.astype(int))
    line_thickness = 1
    radius = 2
    raw_AOI_color = np.where(raw_AOI == 0, 0, 255).astype(np.uint8)
    raw_AOI_color = cv2.merge((raw_AOI_color, raw_AOI_color, raw_AOI_color))
    line_cnt = len(end_points_transed)
    line_cnt = int(line_cnt)
    end_points_transed = end_points_transed.astype(int)

    # dom_path = r'AZK1jDGIZC1zmuooSZCzEg.tif'
    # dom_path = r'-ft2bZI1Ial4C6N_iwmmvw_DOM_0.05.tif'
    # im_dom = cv2.imread(dom_path)
    for idx in range(0, line_cnt, 2):
        col = end_points_transed[idx][0]
        row = end_points_transed[idx][1]
        # to_y[idx2] = row

        end_x = end_points_transed[idx + 1][0]
        end_y = end_points_transed[idx + 1][1]
        cv2.line(opened_color, (col, row), (end_x , end_y), (0, 0, 255), thickness=line_thickness)
        # cv2.line(im_dom,       (col, row), (end_x , end_y), (0, 0, 255), thickness=line_thickness)
        # cv2.circle(raw_AOI_color, (end_x, end_y), radius, (0, 255, 0), line_thickness)
        # cv2.circle(raw_AOI_color, (col, row), radius, (0, 255, 0), line_thickness)
    cv2.imshow("opened_color", opened_color)
    # cv2.imshow("im_dom", im_dom)
    cv2.imshow("img_rotated_color", img_rotated_color)

    # end_points_transed = end_points_transed[:, 0:2]


    # rotated = imutils.rotate_bound(opened_color, -45)....


    # cv2.imshow("rotated", rotated)
    # to_RLE = np.where(to_RLE == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("to_RLE", to_RLE.astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def points_2D_translation(points, tx, ty):
    t_mat = np.array([[1, 0, -tx],
                      [0, -1, ty],
                      [0, 0, 1]])
    results = points.dot(t_mat.T)
    return results

def points_2D_rotated(points, angle_deg):
    angle = math.radians(angle_deg)
    r_mat = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]])
    results = points.dot(r_mat.T)
    return results



def seg_contours(raw_contours, mask_np,  img_rotated, interval_pix=10, max_width_pix=50):
    contours = [np.squeeze(cont) for cont in raw_contours]

    # con = cv2.drawContours(opened_color_img, raw_contours, -1, (0, 255, 0), 2)
    # cv2.imshow("raw_contours", opened_color_img)

    centeral_col = mask_np.shape[1] / 2
    centeral_col = int(centeral_col)

    all_pair_list = []

    for idx, contour in enumerate(contours):
        Xs = contour[:, 0]
        Ys = contour[:, 1]
        y_min = Ys.min()
        y_max = Ys.max()
        x_min = Xs.min()
        x_max = Xs.max()
        h = y_max - y_min
        cut_row_cnt = np.ceil(h / interval_pix).astype(int) + 1
        cut_rows = np.array((range(cut_row_cnt))) * interval_pix +  y_min
        cut_rows[-1] = y_max
        pairs = np.zeros((cut_row_cnt, 2))
        for idx2, row in enumerate(cut_rows):
            cols = contour[contour[:, 1] == row][:, 0]
            cols = np.sort(cols)

            pair = get_pair_col(cols, centeral_col)

            length = pair[1] - pair[0]
            # Huan  !!
            # if (length > max_width_pix) and (idx2 > 0):
            #     pair = pairs[idx2 - 1]

            pairs[idx2] = pair

            is_touched = line_ends_touched(pair[0], row, pair[1], row, mask_np, width=13, height=1, threshold=6)

            cover_ratio = -1

            target_mask_np = np.logical_or(img_rotated, mask_np)

            cover_ratio = get_cover_ratio(int(np.mean(pair[0:2])), int(row), target_mask_np, width=length, height=length)
            # cover_ratio = 0
            cover_ratio_threshold = 0.5

            all_pair_list.append((idx, pair[0], row, pair[1], row, cover_ratio, is_touched))
            # pairs.append(pair)
        # all_pair_list
    # print("all_pair_list:", all_pair_list)

    return all_pair_list

def get_cover_ratio(col, row, mask_np, width, height):

    try:
        cover_ratio = -1
        if width * height == 0:
            return cover_ratio
        mask_w, mask_h = mask_np.shape
        col_start = max(0, int(col - width/2))
        col_end = min(mask_w - 1, int(col + width/2))

        row_start = max(0, int(row - height/2))
        row_end = min(mask_h - 1, int(row + height/2))

        samples = mask_np[row_start:row_end, col_start:col_end]
        # samples = samples / 255

        cover_ratio = samples.sum() / float(width * height)

    #  show image
    # cv_mask_color = cv2.merge([np.where(mask_np > 0, 255, 0).astype(np.uint8)])
    # cv_mask_color = cv2.merge([cv_mask_color, cv_mask_color, cv_mask_color])
    # top_left = (col_start, row_start)
    # bottom_right = (col_end, row_end)
    # cv2.rectangle(cv_mask_color, top_left, bottom_right, color=(0, 255, 0), thickness=2)
    # cv2.imshow(f"cover_ratio: {cover_ratio:.3f}", cv_mask_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

        return cover_ratio
    except Exception as e:
        print("Error in get_cover_ratio():", e)
        return cover_ratio

def line_ends_touched(x1, y1, x2, y2, mask_np, width=13, height=1, threshold=6):
    is_touched1 = check_touched(x1, y1, mask_np, width=width, height=height, threshold=threshold)
    is_touched2 = check_touched(x2, y2, mask_np, width=width, height=height, threshold=threshold)
    #
    # cv2.imshow("mask_np", cv2.merge([np.where(mask_np > 0, 255, 0).astype(np.uint8)]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # re = (is_touched1 or is_touched2)

    return (is_touched1 or is_touched2)

def create_mask(img_np, category_list):
    '''
    create a mask, where True indicates the taget categories.
    :param img_np:
    :param category_list:
    :return:
    '''
    mask_np = np.zeros(shape=img_np.shape, dtype=bool)
    class_idx = img_np
    for category in category_list:
        mask_np = np.logical_or(mask_np, img_np == category)

    return mask_np

def check_touched(col, row, mask_np, width=6, height=1, threshold=6):
    '''
    check whethe a sidewalk touchs cars. car: 58, other: 255
    :param col:
    :param row:
    :param img_np:
    :param threshold:
    :return:
    '''
    mask_w, mask_h = mask_np.shape
    col_start = max(0, int(col - width/1))
    col_end = min(mask_w - 1, int(col + width/1))

    row_start = max(0, int(row - height/1))
    row_end = min(mask_h - 1, int(row + height/1))

    samples = mask_np[row_start:row_end, col_start:col_end]

    is_touched = samples.sum() > threshold

    return is_touched

def get_pair_col(cols, central_col):
    '''
    Find out the start and end col from cols intersecting the horizontal row.
    :param cols: numpy 1-D array
    :return:
    '''

    left = cols.min()
    right = cols.max()
    length = right - left

    # Case 1: one line in the top/bottom
    if length == (len(cols) - 1):
        pass

    # Case 2: two cols only
    if (len(cols) == 2) and (length > 2):
        pass

    # Case 3: several parts, like intersecting with two fingers.
    # step a: find segments
    starts = []
    prev = -2
    for col in cols:
        if col > (prev+1):
            starts.extend((col,))
        prev = col
    if len(starts) == 1:
        starts.append(cols[-1])

    if starts[0] > central_col:
        left = starts[0]
        right = starts[1]
    else:
        left = starts[-2]
        right = starts[-1]

    return left, right


def get_centerline_from_img(img_path):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil)


    im_cv = cv2.imread(img_path)
    target_ids = [12]

    class_idx = img_np
    target_np = np.zeros(img_np.shape)
    for i in target_ids:
        target_np = np.logical_or(target_np, class_idx == i)

    # img_cv = cv2.cvtColor(np.asarray(img_pil),cv2.IMREAD_GRAYSCALE)
    target_np = target_np.astype(np.uint8)
    target_cv = cv2.merge([target_np])
    raw_contours, hierarchy = cv2.findContours(target_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [np.squeeze(cont) for cont in raw_contours]
    centerlines = get_polygon_centline(contours)

    print("OK")

    # centerline1 = get_centerline(polygon, segmentize_maxlen=8, max_points=3000, simplification=0.05, smooth_sigma=5)

def get_polygon_centline(contours, world_coords=[], segmentize_maxlen=1, max_points=3000, simplification=0.5, smooth_sigma=5):
    if not isinstance(contours, list):
        contours = [contours]
    results = []
    for contour in contours:
        try:
            polygon = Polygon(contour)
            print(polygon)
            centerline1 = get_centerline(polygon, segmentize_maxlen=segmentize_maxlen, max_points=max_points, simplification=simplification, smooth_sigma=smooth_sigma)
            print(centerline1)
            results.append(centerline1)
        except Exception as e:
            print("Error in get_polygon_centline():", e)
            results.append(0)
            continue

    if len(world_coords) > 0:
        # coords =
        # centerlines = [np.c_[cont, np.ones((len(cont),))] for cont in contours]
        results = [shapely.affinity.affine_transform(cont, world_coords) for cont in results]
        print(results)

    return results

def test1():
    # path
    path = r'ZXyk9lKhL5siKJglQPqfMA_DOM_0.05.tif'

    # Reading an image in default
    # mode
    image = cv2.imread(path)

    # Window name in which image is
    # displayed
    window_name = 'Image'

    # Polygon corner points coordinates
    pts = np.array([[25, 70], [25, 160],
                    [110, 200], [200, 160],
                    [200, 70], [110, 20]],
                   np.int32)

    pts = pts.reshape((-1, 1, 2))

    isClosed = True

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    image = cv2.polylines(image, [pts],
                          isClosed, color, thickness)

def get_all_widths():
    DOM_dir = r'D:\Research\sidewalk_wheelchair\DC_DOMs'
    # DOM_dir = r'H:\Research\sidewalk_wheelchair\DC_DOMs'
    # img_list = glob.glob(os.path.join(DOM_dir, '*DOM*.tif'))
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\XzB9K8BHqMpZVKZR-E9MBw_DOM_0.05.tif']
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\Jk7bEuZo5fzWeax42a0bSw_DOM_0.05.tif']
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\Jk091JHY-Fnhv_ho_1Qqmg_DOM_0.05.tif']
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\JKABMj3d30_i7hL9T00JjA_DOM_0.05.tif']
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\2vaa4vNnHbgayTqpJZvKrg_DOM_0.05.tif']
    img_list = [r'H:\Research\sidewalk_wheelchair\DC_DOMs\2UzX00nEfTL8jXMfnhoTWw_DOM_0.05.tif']

    skip = 0

    img_list_mp = mp.Manager().list()
    for img in img_list[skip:]:
        img_list_mp.append(img)

    process_cnt = 1

    if (process_cnt == 1) or (len(img_list) < process_cnt * 3):

        cal_witdh_from_list(img_list)

    if process_cnt > 1:

        pool = mp.Pool(processes=process_cnt)

        for i in range(process_cnt):
            print(i)
            pool.apply_async(cal_witdh_from_list, args=(img_list_mp,))
        pool.close()
        pool.join()


def multi_process(func, args, process_cnt=6):
    print("Done")


def get_line(series):
    p = Point(series['col'], series['row'])
    p1 = Point(series['end_x'], series['end_y'])
    line = LineString([p1, p])
    return line

def measurements_to_shapefile(widths_files='', saved_path=''):


    # for idx, f in enumerate(widths_files):
    total_cnt = len(widths_files)
    while len(widths_files) > 0:
        f = widths_files.pop()
        processed_cnt = total_cnt -len(widths_files)
        try:
            if processed_cnt % 1000 == 0:
                print("Processing: ", processed_cnt, f)
            df = pd.read_csv(f)

            lines = df.apply(get_line, axis=1)

            gdf = gpd.GeoDataFrame(df, geometry=lines)
            basename = os.path.basename(f).replace(".csv", ".shp")
            new_name = os.path.join(saved_path, basename)
            gdf.to_file(new_name)
        except Exception as e:
            print("Error in measurements_to_shapefile():", e, f)
            continue


def measurements_to_shapefile_mp(width_dir='', saved_path=''):
    width_dir = r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2'
    saved_path = r'D:\Research\sidewalk_wheelchair\DC_DOMs_width_shapes'

    widths_files = glob.glob(os.path.join(width_dir, r'*.csv'))

    widths_files_mp = mp.Manager().list()

    for f in widths_files[10000:]:
        widths_files_mp.append(f)

    # measurements_to_shapefile(widths_files_mp, saved_path)

    process_cnt = 10
    pool = mp.Pool(processes=process_cnt)
    for i in range(process_cnt):
        pool.apply_async(measurements_to_shapefile, args=(widths_files_mp, saved_path))
    pool.close()
    pool.join()



if __name__ == "__main__":
    # test1()
    # cal_witdh()
    # get_all_widths()
    measurements_to_shapefile_mp()
    # img_path = r'ZXyk9lKhL5siKJglQPqfMA_DOM_0.05.tif'

    # cal_witdh_from_list([])

    # get_centerline_from_img(img_path)
