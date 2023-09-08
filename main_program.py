import tensorflow as tf
from EDSR import *
from Utils import PSNR
import requests
import zipfile
import os
import sys
import re
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


#Vital paths for data structure
main_path = os.getcwd()
dataset_path = os.path.join(main_path, 'dataset')
trainset_highres = os.path.join(dataset_path,'train', 'HR', 'DIV2K_train_HR')
trainset_lowres = os.path.join(dataset_path, 'train', 'LR', 'DIV2K_train_LR_bicubic', 'X4')
valset_highres = os.path.join(dataset_path, 'valid', 'HR', 'DIV2K_valid_HR')
valset_lowres = os.path.join(dataset_path, 'valid', 'LR', 'DIV2K_valid_LR_bicubic', 'X4')
my_img = os.path.join(main_path, 'my image.jpg')


#Downloading and extracting data
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

    
#name of the files that are going to be downloaded and extracted
names = ['DIV2K_train_HR.zip',
        'DIV2K_train_LR_bicubic_X4.zip',
        'DIV2K_valid_HR.zip',
        'DIV2K_valid_LR_bicubic_X4.zip']

#prefix of the download link
download_url_prefix = "http://data.vision.ee.ethz.ch/cvl/DIV2K/"

#all files that end with .zip in the dataset folder
file_names = [filename for filename in os.listdir(dataset_path) if filename.endswith('.zip')]

                                
for name in names:
    #if the zip file has already been donwloaded and process continue to the next
    if name in file_names:
        print("File already downloaded: {}".format(name))
        continue

    print("Processing {}".format(name))
    print("URL: {}".format(download_url_prefix+name))

    response = requests.get(download_url_prefix+name, stream=True)
    zip_path = os.path.join(dataset_path, name)
    if response.status_code == 200:
        with open(zip_path, 'wb') as file:
            for c in response.iter_content(chunk_size=8192):
                file.write(c)
        print("Download Sucessful")
    else:
        print(response.status_code)
        print("Download failure")
        print("Terminating program")
        sys.exit()

#repeated the loop for convinience. If the zips are available proceed with extraction
for name in names:
    if name in file_names:
        #regex pattern options that will help to put the files correctly                        
        options = ['train', 'valid', 'test']
        options2 = ['HR', 'LR']
        pattern = 'x'
        for option in options:
            temp_pattern = pattern.replace('x', option)
            if re.search(temp_pattern, name):
                folder_path = os.path.join(dataset_path, option)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                for option2 in options2:
                    temp_pattern2 = pattern.replace('x', option2)
                    if re.search(temp_pattern2, name):
                        extract_path = os.path.join(folder_path, temp_pattern2)
                        if not os.path.exists(extract_path):
                            os.makedirs(extract_path)
                        zip_path = os.path.join(dataset_path, name)   
                        if len(os.listdir(extract_path))> 0:
                            print("Already extraced: {}".format(zip_path))
                            continue                     
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            print("Extracting: {}".format(name))
                            print("At path: {}".format(extract_path))
                            zip_ref.extractall(extract_path)

print("All images has been extracted")

#The dataset is more efficiently loaded to memory by mapping where parallization can be utilized
def load(path_lr, path_hr):

    img_lr = tf.image.decode_image(tf.io.read_file(path_lr), dtype=tf.float32)
    img_hr = tf.image.decode_image(tf.io.read_file(path_hr), dtype=tf.float32)
    return img_lr, img_hr

#preprocessing
def preprocessing(img_lr, img_hr):
    scaled_img_lr = tf.divide(img_lr, 255)
    scaled_img_hr = tf.divide(img_hr, 255)   
    return scaled_img_lr, scaled_img_hr

#dataaugmenetation
"""problems:
    Since there is an issue with seeds and reproducibility in tf 2.10.
    I cant use tf.random operations and tf concat can't be used here because lr and hr 
    images have different dim. Furthermore, cond will fail in graph execution. 
    The only solution left is tf.where. 

    Rotation is even more problematic since contrib does not exists in tf 2.x and 
    apply_affine_transformation does not take batches. Because of reason mentioned earlier
    randomrotation cannot be used. Therefor the only option left is rot90.

    The seed issue was solved in TensorFlow 2.9.2, but it does match with my CUDA.
    """
@tf.function
def crop_images(img_lr, img_hr, rng):
    crop_size_lr = (48,48, tf.shape(img_lr)[-1])
    crop_size_hr = (192,192, tf.shape(img_hr)[-1])
    scale = 4
    seed = tf.get_static_value(rng.make_seeds()[0])
    if type(seed) == type(None):
        seed = 2000

    start_x_hr = tf.random.uniform((), maxval=tf.shape(img_hr)[0] - crop_size_hr[0], dtype=tf.int32, seed=seed)
    start_y_hr = tf.random.uniform((), maxval=tf.shape(img_hr)[1] - crop_size_hr[1], dtype=tf.int32, seed=seed)

    start_x_lr = tf.cast(tf.divide(start_x_hr, scale), dtype=tf.int32)
    start_y_lr = tf.cast(tf.divide(start_y_hr ,scale), dtype=tf.int32)

    cropped_img_lr = tf.image.crop_to_bounding_box(img_lr, start_x_lr, start_y_lr, crop_size_lr[0], crop_size_lr[1]) 
    cropped_img_hr = tf.image.crop_to_bounding_box(img_hr, start_x_hr, start_y_hr, crop_size_hr[0], crop_size_hr[1])
    
    # cropped_img_lr = tf.image.random_crop(img_lr, size=crop_size_lr, seed=seed) 
    # cropped_img_hr = tf.image.random_crop(img_hr, size=crop_size_hr, seed=seed)
    
    return cropped_img_lr, cropped_img_hr

@tf.function
def rotate_images(img_lr, img_hr):

    rn = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
    k = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)

    rotated_img_lr = tf.where(tf.less(rn, 0.5),
                              tf.image.rot90(img_lr, k), img_lr)
    rotated_img_hr = tf.where(tf.less(rn, 0.5), 
                              tf.image.rot90(img_hr, k), img_hr)
    
    return rotated_img_lr, rotated_img_hr

@tf.function
def flip_left_right_images(img_lr, img_hr):
    
    rn = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

    flipped_img_lr = tf.where(tf.less(rn, 0.5),
                              tf.image.flip_left_right(img_lr), img_lr)
    flipped_img_hr = tf.where(tf.less(rn, 0.5), 
                              tf.image.flip_left_right(img_hr), img_hr)

    return flipped_img_lr, flipped_img_hr

@tf.function
def flip_up_down_images(img_lr, img_hr):
    rn = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
    flipped_img_lr = tf.where(tf.less(rn, 0.5),
                              tf.image.flip_up_down(img_lr), img_lr)
    flipped_img_hr = tf.where(tf.less(rn, 0.5), 
                              tf.image.flip_up_down(img_hr), img_hr)
    
    return flipped_img_lr, flipped_img_hr

#abs paths to all training data
train_lr_paths = [os.path.join(trainset_lowres, filename) for filename in os.listdir(trainset_lowres)] 
train_hr_paths = [os.path.join(trainset_highres, filename) for filename in os.listdir(trainset_highres)]
valid_lr_paths = [os.path.join(valset_lowres, filename) for filename in os.listdir(valset_lowres)]
valid_hr_paths = [os.path.join(valset_highres, filename) for filename in os.listdir(valset_highres)]


# check how the files are orderd
# for lr, hr in zip(train_lr_paths, train_hr_paths):
#     f1 = lr.split('\\', -1)[-1]
#     f2 = lr.split('\\', -1)[-1]
#     if f1!=f2:
#         print(f1)
#         print(f2)

def main():           
    #Create a tuple of the training data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_lr_paths, train_hr_paths))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_lr_paths, valid_hr_paths))

    #load the dataset from path
    train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)

    # #preprocessing
    # train_dataset = train_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    # valid_dataset = valid_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

    #create random number generator for creating seed

    rng = tf.random.Generator.from_seed(
        tf.cast(20, dtype=tf.int64)
    )

    #check if the images are cropped at the same region
    # for lr, hr in zip(train_lr_paths[:10], train_hr_paths[:10]):
    #     load_img_lr, load_img_hr = load(lr, hr)
    #     fig, axs = plt.subplots(4,1)
    #     axs[0].imshow(load_img_lr.numpy())
    #     axs[1].imshow(load_img_hr.numpy())
    #     cropped_img_lr, cropped_img_hr = crop_images(load_img_lr ,load_img_hr, rng )
    #     axs[2].imshow(cropped_img_lr.numpy())
    #     axs[3].imshow(cropped_img_hr.numpy())
    #     cropped_img_lr, cropped_img_hr = crop_images(load_img_lr ,load_img_hr, rng )
    #     plt.show()

    #augmentations
    train_dataset = train_dataset.map(lambda lr, hr: crop_images(lr, hr, rng), num_parallel_calls=tf.data.AUTOTUNE)\
        .map(flip_left_right_images, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(flip_up_down_images, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(rotate_images, num_parallel_calls=tf.data.AUTOTUNE)

    valid_dataset = valid_dataset.map(lambda lr, hr: crop_images(lr, hr, rng), num_parallel_calls=tf.data.AUTOTUNE)\
        .map(flip_left_right_images, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(flip_up_down_images, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(rotate_images, num_parallel_calls=tf.data.AUTOTUNE)\

    #batch size
    train_dataset = train_dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

    #reduce learning rate as learning halts
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_lr = 1e-10,
        verbose=1
    )

    #create model instance, compile, train and save model to file.
    model = EDSR()

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mae', metrics=[tf.keras.metrics.PSNR()])
            
    model.fit(train_dataset, validation_data=valid_dataset, epochs=200, callbacks=[reduce_lr])

    model.save('edsr_model')



    #load model with new shape for inference. 
    # model = tf.keras.models.load_model('edsr_model', compile=False)
    # weights = model.get_weights()
    # model = None
    # pred_model = EDSR()
    # pred_model.build((None, None, None, 3))
    # pred_model.set_weights(weights)
    # model = pred_model

    # #tiling
    # tile_size=(256,256 ,3)
    # stride_size = (256, 256, 1)


    img = tf.expand_dims(tf.image.decode_image(tf.io.read_file(my_img)), axis=0) #load image to enhance
    model.use_tile() # enables tiling for very large image files

    better_img = model.predict(img) 
    #
    better_img_uint8 = tf.cast(better_img*255, dtype=tf.uint8) 
    encoded_image = tf.image.encode_jpeg(tf.squeeze(better_img_uint8))
    tf.io.write_file(os.path.join(main_path, 'better body.jpg'), encoded_image)

    #plot original and enhanced image side by side
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(tf.squeeze(img))
    axs[0].set_title('original')
    axs[0].grid(False)
    axs[1].imshow(tf.squeeze(better_img_uint8))
    axs[1].set_title('enhanced')
    axs[1].grid(False)
    plt.show()


if __name__ == "__main__":
    main()
    sys.exit()