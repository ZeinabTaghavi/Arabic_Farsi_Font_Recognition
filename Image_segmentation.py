import cv2
from scipy import ndimage
import pytesseract
import numpy as np
from PIL import Image
import os



def Find_Line_Based_on_Pixel_Density(img_file,vertical_percent , horizontal_percent ,min_rect_size,min_border_percent):


    img = cv2.imread(img_file)

    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    ret, threshed_img = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    gray_env = cv2.bitwise_not(threshed_img)
    gray_corrected_rotation = threshed_img

    # 2 - find the high compression vertical area

    vertical_hist = [sum(gray_env[i,:]) for i in range(img.shape[0])]

    vertical_temp = gray_corrected_rotation.copy()
    vertical_limit = gray_env.shape[1] * 255 * vertical_percent *.01
    for i in range(len(vertical_hist)):
        if vertical_hist[i] > vertical_limit:
            vertical_temp[i,:] = 255
        else:
            vertical_temp[i,:] = 0

    cv2.imwrite('sequence_of_segmentation/'+img_file + '_find_segment_area_by_x_y_projection_1_vertical_line_detected.jpg' , vertical_temp)
    contour , _ = cv2.findContours(vertical_temp , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(corrected_rotation , contour , -1 , 100 , 3)

    # 3 - in vertical high compression areas, make all horizontal high compression areas

    vertical_lines_positions = []  # they are vertical high compression areas
    for cnt in contour:
        x , y , w , h = cv2.boundingRect(cnt)
        vertical_lines_positions.append([y,y+h])
    # corrected_rotation = cv2.rectangle(corrected_rotation , (x,y) , (x+w , y+h) , (0,0,200) ,-1)


    gray_corrected_rotation_env = cv2.bitwise_not(threshed_img)
    line_location_image = np.zeros((threshed_img.shape[0],threshed_img.shape[1]),np.uint8)
    line_location_image.fill(255)

    for y1,y2 in vertical_lines_positions:
        temp_img_env = gray_corrected_rotation_env[y1:y2,:]
        horizontal_limit = (y2-y1) * 255 * horizontal_percent * .01
        for j in range(temp_img_env.shape[1]):
            if sum(temp_img_env[:,j]) > horizontal_limit:
                line_location_image[y1:y2, j] = 0


    kernel_h = int(img.shape[1]*.01)
    kernel_v = int(img.shape[0] * .004)

    # //////////////////////////////////////////
    # print (kernel_v)
    dilate_kernel = np.ones((kernel_v, kernel_h), np.uint8)
    line_location_image = cv2.erode(line_location_image , dilate_kernel , iterations=1)
    cv2.imwrite('sequence_of_segmentation/'+img_file + '_find_segment_area_by_x_y_projection_2_just_lines.jpg', line_location_image)

    combine = cv2.bitwise_and(line_location_image , gray_corrected_rotation)
    combine = cv2.dilate(combine , np.ones((8,8),np.uint8) , iterations = 1)
    cv2.imwrite('sequence_of_segmentation/'+img_file + '_find_segment_area_by_x_y_projection_3_horizontal_line_rected.jpg',combine)


    contour ,_ = cv2.findContours(combine , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    # # gray_corrected_rotation = cv2.erode(gray_corrected_rotation , kernel=dilate_kernel , iterations=0)
    cv2.drawContours(combine , contour , -1 , 100,10)
    cv2.imwrite(img_file + '_find_segment_area_by_x_y_projection_4_contoured.jpg', combine)

    max_x = img.shape[1] * (1 - min_border_percent[1])
    min_x = img.shape[1] * min_border_percent[1]
    max_y = img.shape[0] * (1 - min_border_percent[0])
    min_y = img.shape[0] * min_border_percent[0]
    max_w = img.shape[1] * (1 - min_rect_size[1])
    min_w = img.shape[1] * min_rect_size[1]
    max_h = img.shape[0] * (1 - min_rect_size[0])
    min_h = img.shape[0] * min_rect_size[0]


    if not os.path.exists('lines_images_for_'+img_file):
        os.mkdir('lines_images_for_'+img_file)

    count = 0
    for ctn in contour:
        (x, y, w, h) = cv2.boundingRect(ctn)
        if min_w < w < max_w and min_h < h < max_h and min_x < x < max_x and min_y < y < max_y:
            count += 1
            cv2.rectangle(threshed_img, (x, y-(int(h*3/4))),(x + w, y + (int(h*7/4))), (100, 100, 100), 4)
            line = img[y-(int(h*3/4)): y + (int(h*7/4)) , x: x+w]
            cv2.imwrite('lines_images_for_' + img_file+'/'+str(count) + '_line_y1_{}_y2_{}_x1_{}_x2_{}_.jpg'.format(y-(int(h*3/4)), y + (int(h*7/4)) , x, x+w),line)


    cv2.imwrite('sequence_of_segmentation/'+img_file + '_find_segment_area_by_x_y_projection_5_bound.jpg', threshed_img)


def Segment(img_file):

    img = cv2.imread(img_file)

    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    ret, threshed_img = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    env = cv2.bitwise_not(threshed_img)
    contour ,_ = cv2.findContours(env , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    len_cnt = []
    for cnt in contour:
        if cv2.contourArea(cnt)<(img.shape[0]*img.shape[1]/3):
            len_cnt.append(cv2.contourArea(cnt))


    clustering = []
    if len(len_cnt) == 0:
        return 0

    k_pre = min(len_cnt)
    for x in range(1,11):
        k = (max(len_cnt)-min(len_cnt)) * x / 10
        clustering.append(sum([k_pre <= i < k for i in len_cnt]))
        k_pre = k

    # print(clustering)
    def_clustering = [clustering[i-1] - clustering[i] for i in range(1 , len(clustering))]
    thresh = (def_clustering.index(max(def_clustering))+1) * (max(len_cnt)-min(len_cnt))/10
    thresh_pre = (def_clustering.index(max(def_clustering))) * (max(len_cnt)-min(len_cnt))/10
    # print (thresh , thresh_pre)
    # print (len(contour),sum(clustering))

    dst_dir = img_file + 'folder' #''.join([i + '_' for i in re.split(r'.jpg', img_file)])
    if not os.path.isdir('./'+ dst_dir):
        print('made dst   : '+ dst_dir)
        os.mkdir(dst_dir)

    count = 0


    detected_segments_direction = []
    contoured_img = threshed_img.copy()
    detected_img = threshed_img.copy()
    for cnt in contour:
        if not thresh_pre< cv2.contourArea(cnt)< thresh:
            count += 1
            x , y, w, h = cv2.boundingRect(cnt)
            segment_img = threshed_img[:,x:x+w]
            detected_img[:,x:x+w] = 0
            img_name = str(count) + '_segment_y1_{}_y2_{}_x1_{}_x2_{}_.jpg'.format(y - (int(h * 3 / 4)),
                                                                                    y + (int(h * 7 / 4)), x,
                                                                                    x + w)

            cv2.imwrite(dst_dir + '/' + img_name, segment_img)

            cv2.imwrite('segments/' + img_name , segment_img)

            detected_segments_direction.append([img_name,[y - (int(h * 3 / 4)),y + (int(h * 7 / 4)), x,x + w]])
            cv2.drawContours(contoured_img, [cnt], 0, 100, 3)

    #cv2.imwrite('contoured.jpg', contoured_img)
    # cv2.imwrite('detected.jpg', detected_img)

    return detected_segments_direction


