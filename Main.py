# zeinab Taghavi
#
# its better image be threshed once before usage
# 1 - correct rotation to more accurately find lines
# 2 - find the high compression vertical area
# 3 - in vertical high compression areas, make all horizontal high compression areas
# 4 - detect segments with CNNs and Denses
# 5 - result is rate of each font in image

import cv2
from scipy import ndimage
import pytesseract
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import glob
import re
from Load_data import Train_Test_set
from Recognition import Font_Recognition, Describing_Result
from Image_segmentation import Find_Line_Based_on_Pixel_Density, Segment




if __name__ == '__main__':
    
    n1 = 1
    n2 = 2
    avg_time = []

    
    # Detecting lines
    for i in range(n1, n2):
        e1 = cv2.getTickCount()

        # per any type of documents set this things:
        erod_itter = 0
        dilate_itter = 0  # same for most of them
        ker_num = 0  # same for most of them
        th = 1
        bias = 150
        img_file = str(i) + '.jpg'

        min_rect_size = [0.01, 0.01]  # percents of height and width of image
        min_border_percent = [0.05, 0.05]  # percents of image's height and width is border
        Find_Line_Based_on_Pixel_Density(img_file, 5, 5, min_rect_size, min_border_percent)

    # Word segmentation in lines

    for n in range(n1,n2):

        if not os.path.isdir('./segments'):
            print ('----- Segment file created')
            os.mkdir('segments')

        all_detected_segments_direction = []
        for i in glob.glob('lines_images_for_'+str(n)+'.jpg'+'/*.jpg'):
            img_file = i

            print('Segmenting for:  '+i)
            detected_segments_direction = Segment(img_file)
            all_detected_segments_direction.append(detected_segments_direction)
    
    results = Font_Recognition(all_detected_segments_direction)
    Describing_Result(results)



