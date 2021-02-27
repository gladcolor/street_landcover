import cv2
import numpy as np
import math
import imutils

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

def cal_witdh():
    # img_path = r'AZK1jDGIZC1zmuooSZCzEg_DOM.tif'
    img_path = r'Ld-CMATy8ZxKap6VAtZTEg_DOM.tif'
    # img_path = r'v-VR9FB7kCxU1eDLEtFiJQ_DOM.tif'
    target_ids = [244]
    im_cv = cv2.imread(img_path)

    red_channel = im_cv[:, :, 2]
    class_idx = red_channel
    AOI = np.zeros((len(class_idx), len(class_idx)))
    for i in target_ids:
        AOI = np.logical_or(AOI, class_idx == i)
        print(AOI)
    # AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)
    print(AOI)

    morph_kernel_open  = (5, 5)
    morph_kernel_close = (15, 15)
    g_close = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_close)
    g_open  = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_open)

    AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)

    yaw_deg =  226.4377593994141 - 90
    AOI = imutils.rotate_bound(AOI, yaw_deg)

    cv2_closed = cv2.morphologyEx(AOI, cv2.MORPH_CLOSE, g_close) # fill small gaps
    cv2_opened = cv2.morphologyEx(cv2_closed, cv2.MORPH_OPEN, g_open)

    # draw lines
    line_cnt = 160
    img_h, img_w = cv2_opened.shape
    start_x = 0
    end_x = img_w -1
    interval = int(img_h / line_cnt)
    line_ys = range(interval, img_h, interval)
    line_thickness = 1


    to_RLE = cv2_opened[line_ys]
    run_lengths, run_rows = rle_encoding(to_RLE)
    print("rung_lengths, rows:\n", run_lengths[::2], "\n", run_rows, "\n", run_lengths[1::2])

    AOI = np.where(AOI == 0, 0, 255).astype(np.uint8)

    # cv2.imshow("Raw image", AOI.astype(np.uint8))

    # cv2_closed = np.where(cv2_closed == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("cv2_closed", cv2_closed.astype(np.uint8))

    cv2_opened = np.where(cv2_opened == 0, 0, 255).astype(np.uint8)
    opened_color = cv2.merge((cv2_opened, cv2_opened, cv2_opened))

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
    max_width_meter = 3
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
                # print("long length!")
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
    raw_contours, hierarchy = cv2.findContours(cv2_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_pair_list = seg_contours(raw_contours[5:6], opened_color)
    # all_pair_list = seg_contours(raw_contours[:], opened_color)
    for idx, pair in enumerate(all_pair_list):
        x = int(pair[1])
        y = int(pair[2])
        end_x = int(pair[3])
        end_y = int(pair[4])
        cv2.line(opened_color, (x, y), (end_x, end_y), (255, 0, 0), thickness=line_thickness)

    cv2.imshow("opened_color added pairs", opened_color.astype(np.uint8))



    # rotated = imutils.rotate_bound(opened_color, -45)
    # cv2.imshow("rotated", rotated)
    # to_RLE = np.where(to_RLE == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("to_RLE", to_RLE.astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def seg_contours(raw_contours, opened_color_img, interval_pix=10, max_width_pix=50):
    contours = [np.squeeze(cont) for cont in raw_contours]
    con = cv2.drawContours(opened_color_img, raw_contours, -1, (0, 255, 0), 2)
    # cv2.imshow("raw_contours", opened_color_img)

    centeral_col = opened_color_img.shape[1] / 2
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
            if (length > max_width_pix) and (idx2 > 0):
                pair = pairs[idx2 - 1]

            pairs[idx2] = pair
            all_pair_list.append((idx, pair[0], row, pair[1], row))
            # pairs.append(pair)
        # all_pair_list
    print(all_pair_list)

    return all_pair_list

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





if __name__ == "__main__":
    cal_witdh()
