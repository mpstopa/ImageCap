from configs.default import _C as config
from configs.default import update_config

from datasets import flickr8k_parse
from nltk.translate.bleu_score import corpus_bleu
from scipy import misc
from PIL import Image

from models import decoder, image_preprocessing, predict, transfer_models

import matplotlib.pyplot as plt
import math
import numpy as np
import os
import path_generation
import text_processing

config_file = "./configs/attn.yaml"
update_config(config, config_file) # config is imported from default.py _C above
# most parameters are in default.py, parameters in attn.yaml overwrite these defaults

train_vocab = text_processing.Vocabulary()
word_to_id_path = os.path.join(config.PATH.VOCABULARY_PATH, config.VOCABULARY.WORD_TO_ID)
id_to_word_path = os.path.join(config.PATH.VOCABULARY_PATH, config.VOCABULARY.ID_TO_WORD)
count_path = os.path.join(config.PATH.VOCABULARY_PATH, config.VOCABULARY.COUNT)
train_vocab.load_vocabulary(word_to_id_path, id_to_word_path, count_path)

path_gen = path_generation.PathGenerator(config.DECODER.GRU, 
                                         config.DATASET, 
                                         config.DECODER.NUM_RNN_LAYERS, 
                                         config.DECODER.BATCH_SIZE, 
                                         config.DECODER.BATCH_NORM, 
                                         config.DECODER.DROPOUT, 
                                         config.ATTENTION, 
                                         config.DECODER.ATTN_TYPE)

path_checkpoint = path_gen.get_weights_path()
model_path = path_gen.get_model_path()
captions_path = path_gen.get_captions_path()
print("path_checkpoint",path_checkpoint,"\n model_path",model_path, \
     " \n captions_path",captions_path)
	 
if config.ATTENTION:
    transfer_values = np.load(os.path.join(config.PATH.FEATURES_PATH, 'vgg16_flickr8k_train_attn.npy'))
    val_transfer_values = np.load(os.path.join(config.PATH.FEATURES_PATH, 'vgg16_flickr8k_val_attn.npy'))
else:
    transfer_values = np.load(os.path.join(config.PATH.FEATURES_PATH, 'vgg16_flickr8k_train.npy'))
    val_transfer_values = np.load(os.path.join(config.PATH.FEATURES_PATH, 'vgg16_flickr8k_val.npy'))
	
decoder_model = decoder.load_model(model_path, path_checkpoint)

VGG_transfer_model, VGG_image_size = transfer_models.vgg_model(config.ATTENTION)

### beam size might be changed here
beam_size = 1
if config.ATTENTION:
    get_weights = True
else:
    get_weights = False

path0 = 'C:/Users/MStopa/ImageCaptioning/datasets/Flickr8k/Images/240696675_7d05193aa0.jpg'
path1 = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/KM_girl1b.jpg'
# path1 = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/golf1.jpg'
# path1 = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/ChattyFeet.jpg'
# path1 = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/Picasso1.jpg'
# path1 = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/skiing1.jpg'
# path1 = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/soccer1.jpg'
# path1 = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/ToothBrushing.jpg'
# path = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/beach_shot_224_224.jpg'
# path = 'C:/Users/MStopa/ImageCaptioning/datasets/MyImages/99679241_adc853a5c0.jpg'

# x0=image_preprocessing.image_preprocessing(path0,VGG_image_size)
# y0 = np.expand_dims(x0, axis=0)
# z0 = VGG_transfer_model.predict(y0)
# x1=image_preprocessing.image_preprocessing(path,VGG_image_size)
# y1 = np.expand_dims(x1, axis=0)
# z1 = VGG_transfer_model.predict(y1)

# for i in range (4096):
#     print("i",i,"z0[0,i]",z0[0,i],"z1[0,i]",z1[0,i])

result0 = predict.generate_caption(path0, 
                                  VGG_image_size, 
                                  decoder_model, 
                                  VGG_transfer_model, 
                                  train_vocab, 
                                  beam_size=beam_size, 
                                  attn=config.ATTENTION, 
                                  get_weights=get_weights)
result1 = predict.generate_caption(path1, 
                                  VGG_image_size, 
                                  decoder_model, 
                                  VGG_transfer_model, 
                                  train_vocab, 
                                  beam_size=beam_size, 
                                  attn=config.ATTENTION, 
                                  get_weights=get_weights)

if get_weights:
    captions, probs, weights = result1
else:
    captions, probs = result1
best_caption = captions[0]

img = image_preprocessing.image_preprocessing(path0, VGG_image_size)
plt.imshow(img)
plt.show()
print('result0',result0[0])
img = image_preprocessing.image_preprocessing(path1, VGG_image_size)
plt.imshow(img)
plt.show()
print('result1',result1[0])
for i in range(len(captions)):
    print(" ".join(captions[i]),
          "{:.3f}".format(probs[i]))
		  
if get_weights:
    cols = 4
    rows = math.ceil(len(best_caption) / cols)
    plt.figure(1, figsize=(12,12))
    for word_num in range(len(best_caption)):
        weights_img = np.reshape(weights[word_num], [14,14])
        weights_img = misc.imresize(weights_img, (224, 224))
        img = image_preprocessing.image_preprocessing(path, (224,224))
        plt.subplot(rows, cols, word_num + 1)
        plt.title(best_caption[word_num], fontsize=20)
        plt.imshow(img)
        plt.imshow(weights_img, cmap='bone', alpha=0.8)
        plt.axis('off')
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig('./test.png')
    plt.show()
	
captions_file = os.path.join(config.PATH.ANNOTATIONS_PATH, "Flickr8k.token.txt")
test_txt_path = os.path.join(config.PATH.ANNOTATIONS_PATH, "Flickr_8k.testImages.txt")

filenames_with_all_captions = flickr8k_parse.generate_filenames_with_all_captions(captions_file, config.PATH.IMG_PATH)

test_filenames_with_all_captions = flickr8k_parse.generate_set(test_txt_path, 
                                                               filenames_with_all_captions, 
                                                               config.PATH.IMG_PATH)

test_captions = flickr8k_parse.make_list_of_captions(test_filenames_with_all_captions)

text_processing.preprocess_captions(test_captions)

references = []
for list_captions in test_captions:
    reference = []
    for caption in list_captions:
        reference.append(caption.split())
    references.append(reference)
	
references[0]
