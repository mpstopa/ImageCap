from keras.applications import VGG16
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras import Model

from models import image_preprocessing
import numpy as np
import os


def vgg_model(attn):
    """
    Generate vgg model

    :param attn: bool flag for attention model encoding
    :return: return encoder Model instance
    """
    vgg_instance = VGG16(include_top=True, weights='imagenet')
    print('vgg_instance.summary()',vgg_instance.summary())
    if attn:
        transfer_layer = vgg_instance.get_layer('block5_conv3')
    else:
        transfer_layer = vgg_instance.get_layer('fc2')
    vgg_transfer_model = Model(inputs=vgg_instance.input, outputs=transfer_layer.output)
    input_layer = vgg_instance.get_layer('input_1')
    print("input_layer.input_shape",input_layer.input_shape)
    image_size = input_layer.input_shape[1:3]
    print("image_size",image_size)
    return vgg_transfer_model, image_size

def vgg_model2(attn):
    
# this was never used successfully and is now obsolete.
# the point was to do pooling of the (14,14,512) layer to get a (1,512) layer
# saw a post saying that this was not automatic - the final output is like a
# feature map, not a class label. Needed to
# 1. set include_top to False
# 2. set pooling to 'avg' (or 'max').
# I tried to do it in parallel with the ordinary vgg_model but didn't seem to
# work. Finally just did the averaging by hand in evaluate.ipynb (search cosine).
# Note also that transfer_models.py is used by train_model as well as evaluate.

# but I wanted the image encoding vector to compare to other image encoding vectors.
# Again done in evaluate.
    
    """
    Generate vgg model

    :param attn: bool flag for attention model encoding
    :return: return encoder Model instance
    """
    vgg_instance = VGG16(include_top=False, weights='imagenet',pooling='avg')
    print('vgg_instance.summary()',vgg_instance.summary())
    if attn:
        transfer_layer = vgg_instance.get_layer('block5_conv3')
    else:
        transfer_layer = vgg_instance.get_layer('fc2')
    vgg_transfer_model = Model(inputs=vgg_instance.input, outputs=transfer_layer.output)
    input_layer = vgg_instance.get_layer('input_2')
    print("input_layer.input_shape",input_layer.input_shape)
    image_size = input_layer.input_shape[1:3]
    return vgg_transfer_model, image_size


def use_pretrained_model_for_images(filenames_with_all_captions, attn, printswitch, batch_size=64):
    """
    Uses the pretrained model without prediction layer to encode the images into the set of the features.

    :param filenames_with_all_captions: list of dictionaries containing images with the corresponding captions
    :param attn: bool flag for attention model encoding
    :param batch_size: size of the batch for CNN
    :return: np array with generated features
    """
    print("use_pretrained_model_for_image - filenames_with_all_captions", \
          list(filenames_with_all_captions.keys())[0])
    transfer_model, img_size = vgg_model(attn)
    print("transfer model",transfer_model,"img_size",img_size)
    # get the number of images in the dataset
    num_images = len(filenames_with_all_captions)
    print("num_images",num_images)
    # calculate the number of iterations
    iter_num = int(num_images / batch_size)
    # variable to print the progress each 5% of the dataset
    five_perc = int(iter_num * 0.05)
    iter_count = 0
    cur_progress = 0

    # get the paths to all images without captions
    image_paths = list(filenames_with_all_captions.keys())
    # list for the final result
    transfer_values = []

    # start and end index for each batch
    first_i = 0
    last_i = batch_size

    # loop through the images
    while first_i < num_images:
        print("filename",list(filenames_with_all_captions.keys())[first_i])
        iter_count += 1

        # progress print
        if iter_count == five_perc:
            iter_count = 0
            print(str(cur_progress) + "% of images processed")
            cur_progress += 5

        # to make sure that last batch is not beyond the number of the images
        if last_i > num_images:
            last_i = num_images

        # initialize the list for the batch
        image_batch = []

        # loop to form batches
        i00=first_i-1
        if printswitch:
            print("first_i",first_i,"last_i",last_i)

        for image in image_paths[first_i:last_i]:
            i00+=1
            if printswitch:
                print("filename, i00",i00,list(filenames_with_all_captions.keys())[i00])
            # preprocess image
            image = image_preprocessing.image_preprocessing(image, img_size)
            # append image to batch list
            image_batch.append(image)

        # run the model to encode the features
        preds = transfer_model.predict(np.array(image_batch))
#        print("preds.shape",preds.shape)

        # append predictions from the batch to the final list
        for pred in preds:
            transfer_values.append(pred)

        # update first and last indices in the batch
        first_i += batch_size
        last_i += batch_size

    reset_keras()
    del transfer_model
    return np.array(transfer_values)


def save_features(np_arr, folder, filename):
    """
    Saves encoded features into the .npy file.

    :param np_arr: the array with features which should be saved
    :param folder: path to the destination folder
    :param filename: filename of the features file
    """
    # form the full path for the file
    full_path = os.path.join(folder, filename)
    # create the folder if it does not exist
    if not os.path.exists(folder):
        os.mkdir(folder)
    # save file
    np.save(full_path, np_arr)
    print("Array was saved to {}".format(full_path))


def reset_keras():
    """
    Releases keras session
    """
    sess = get_session()
    clear_session()
    sess.close()
