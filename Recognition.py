import numpy as np
from Model import FR_Model
from keras.preprocessing import image


def Font_Recognition(all_detected_segments_direction):

    FR_model = FR_Model(img_shape=(32, 32, 3))

    FR_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    FR_model.load_weights('model.h5')
    
    segments_dir = 'segments'
    results = []
    
    #print(all_detected_segments_direction)
    
    for line_direction in all_detected_segments_direction:
        for segments_direction in line_direction:
            #print('-------------------------')
            #print(segments_direction)
            img = image.load_img(segments_dir + '/' + segments_direction[0], target_size= (32,32))
            test_img = image.img_to_array(img)
            test_img = np.expand_dims(test_img , axis=0)
            result = FR_model.predict(test_img)
            results.append([result,segments_direction[1:]])
            
    return results


import math

def Describing_Result(results):
    result = [0,0,0]
    for r in results:
        result[list(r[0][0]).index(max(list(r[0][0])))] += 1
    
    #print(result)
    
    sum_segments = sum(result)
    print(sum_segments)
    print('Elham: '+ str(round((100*result[0])/sum_segments , 2)) + '% \n' +
          'Farisi: '+ str(round((100*result[1])/sum_segments , 2)) + '% \n' +
          'Vazir: '+ str(round((100*result[2])/sum_segments , 2)) + '% \n')
    return
# -*- coding: utf-8 -*-

