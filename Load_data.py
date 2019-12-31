from keras.preprocessing.image import ImageDataGenerator

def Train_Test_set(path):

    train_date_gen = ImageDataGenerator(rescale= 1./255,
                                        shear_range= .2)
    
    test_date_gen = ImageDataGenerator(rescale= 1./255)
    
    train_set= train_date_gen.flow_from_directory('data_sets/fr_datas/train',
                                                  target_size= (32,32),
                                                  batch_size= 32,
                                                  class_mode= 'categorical')
    
    test_set= test_date_gen.flow_from_directory('data_sets/fr_datas/test',
                                                  target_size= (32,32),
                                                  batch_size= 32,
                                                  class_mode= 'categorical')
    return train_set , test_set
'''
FR_model.fit_generator(train_set, 
                       samples_per_epoch = 5000,
                       epochs= 20,
                       validation_data = test_set, 
                       nb_val_samples = 1000)
'''