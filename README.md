# Arabic_Farsi_Font_Recognition
#### thresh base segmentation, using deep learning for recognition
metho = contours, pixel density for segmentation

main processes:
1 - with pixel density, detect lines

<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/sequence_of_segmentation/1.jpg_find_segment_area_by_x_y_projection_1_vertical_line_detected.jpg?raw=true" width="30%" height="30%">

<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/sequence_of_segmentation/1.jpg_find_segment_area_by_x_y_projection_2_just_lines.jpg?raw=true" width="30%" height="30%">

<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/sequence_of_segmentation/1.jpg_find_segment_area_by_x_y_projection_5_bound.jpg?raw=true" width="30%" height="30%">

2 - detecting segments in lines with contoures

line:

<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/lines_images_for_1.jpg/1_line_y1_1433_y2_1493_x1_851_x2_1194_.jpg?raw=true" width="30%" height="30%">

segments:

<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/lines_images_for_1.jpg/1_line_y1_1433_y2_1493_x1_851_x2_1194_.jpgfolder/4_segment_y1_1_y2_61_x1_271_x2_339_.jpg?raw=true" width="5%" height="5%">
<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/lines_images_for_1.jpg/1_line_y1_1433_y2_1493_x1_851_x2_1194_.jpgfolder/1_segment_y1_13_y2_53_x1_246_x2_260_.jpg?raw=true" width="5%" height="5%">
<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/lines_images_for_1.jpg/1_line_y1_1433_y2_1493_x1_851_x2_1194_.jpgfolder/6_segment_y1_-2_y2_49_x1_181_x2_238_.jpg?raw=true" width="5%" height="5%">
<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/lines_images_for_1.jpg/1_line_y1_1433_y2_1493_x1_851_x2_1194_.jpgfolder/5_segment_y1_-1_y2_60_x1_136_x2_173_.jpg?raw=true" width="5%" height="5%">
<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/lines_images_for_1.jpg/1_line_y1_1433_y2_1493_x1_851_x2_1194_.jpgfolder/7_segment_y1_-2_y2_49_x1_81_x2_128_.jpg?raw=true" width="5%" height="5%">
<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/lines_images_for_1.jpg/1_line_y1_1433_y2_1493_x1_851_x2_1194_.jpgfolder/2_segment_y1_6_y2_57_x1_20_x2_69_.jpg?raw=true" width="5%" height="5%">
<img src="https://github.com/ZeinabTaghavi/Arabic_Farsi_Font_Recognition/blob/master/lines_images_for_1.jpg/1_line_y1_1433_y2_1493_x1_851_x2_1194_.jpgfolder/3_segment_y1_6_y2_57_x1_3_x2_30_.jpg?raw=true" width="5%" height="5%">

3 - 
