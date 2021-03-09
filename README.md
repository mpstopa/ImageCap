# ImageCap
This repository contains code for a deep learning model that takes an image as an input and
generates a caption. The models used here were adapted from a the repository of https://github.com/bulatkh.

The implemented models follow the Encoder-Decoder framework and some of the decoder models also use Attention mechanisms. 
The models are based on and trained with the Flickr8k dataset. Typically, of the 8000 images in the dataset, 6000 are
used for training, 1000 are used for validation and 1000 are used for testing.

The Flickr8k Images are not included but are available in various places including at 
Kaggle here: https://www.kaggle.com/adityajn105/flickr8k/activity. These 8000 jpg files
should be placed in the directory: ./datasets/Flickr8k/Images. This repository does have 
the image captions in ./datasets/Flickr8k/annotations.

Note: references to COCO dataset are not supported and blocks with COCO processing
can be skipped/ignored.

The program uses [YACS library](https://github.com/rbgirshick/yacs) for managing configurations. The architectures 
and hyperparameters are managed using ```configs``` directory: configs/default.py``` contains a default set of 
hyperparameters. The file attn.yaml overrides the default parameters and so modifying configs>default.py itself is not 
necessary. However, it is easier to specify, for example the PATHS (see below) for the various datasets,
models etc right in the default.py file. And you can just use only default.py if you prefer (change each
time for each type of run).

Aside from the yacs library, this repository uses keras, tensorflow, json,
numpy, pandas and the Python Imaging Library (PIL).

This repository is built to work on a Jupyter notebook (JN). The principle routines for encoding, training
and evaluation are ipynb files (JN files). It is easy enough to change those into simple python
routines and run in a standard python environment.

Step 1. download and unpack the zip file for the repository.
Step 2. obtain (cf. above) the images from the Flickr8k dataset and place them
in ./datasets/Flickr8k/Images/.
Step 3. choose a configuration (Define configurations of your model and create a yaml
file in ```configs``` directory. This configuration should be then passed to the ```config_file``` 
variable in all the notebooks; can also just change default.py directly to modify the runs). 
The easiest to use is the simple gru without attention or batch_norm. 
Step 3. Run all cells of the encoder (except those rlated to COCO) in a Jupyter
notebook.
Step 4. Check that .npy files have been created in the cnn_features directory.
Step 5. Run all cells in train_model.ipynb. On a simple laptop this could take
a couple hours when attention is being used.
Step 6. Run evaluation.ipynb. You should modify this code to include your own images.
The images are preprocessed (e.g. re-sized to (3,224,224)) so most any image should work.

Note: because the caption model is trained on the Flickr8k dataset the evaluation will
not recognize actions or entities that it has not seen before. You can try to enlarge
the dataset with your own images and captions and see how that works! Fun!


build and code notes:

- paths in configs>default.py - all relative to home directory

_C.PATH.IMG_PATH = "./datasets/Flickr8k/Images/"
_C.PATH.ANNOTATIONS_PATH = "./datasets/Flickr8k/annotations/"
_C.PATH.FEATURES_PATH = "./cnn_features/"
_C.PATH.MODELS_ARCHITECTURE_PATH = "./models/"
_C.PATH.WEIGHTS_PATH = "./weights/"
_C.PATH.CALLBACKS_PATH = "./callbacks/"
_C.PATH.VOCABULARY_PATH = "./vocabulary/"

_C.VOCABULARY = CN()
_C.VOCABULARY.WORD_TO_ID = "word_to_id.pickle"
_C.VOCABULARY.ID_TO_WORD = "id_to_word.pickle"
_C.VOCABULARY.COUNT = "word_counter.pickle"

· Encoder

The encoder takes images (it reads images and captions, but does not use captions)
and runs a pre-trained model, vgg16, on them to "predict" their encoding. This
model is trained on imagenet. The predictions are stored in an .npy file in
cnn_features directory. The form of these "transfer_values" depends upon the
decoder that will be used. For ordinary GRU the transfer_value for each image 
is a 4096-dim vector which is taken from the last fully connected layer of the
vgg model. For attention models it is more complicated. The input to the decoder 
for each image is (196,512). What the encoder stores in the .npy file is a (14,14,512)
dimensional array (for each image) - this is an intermediate output of vgg16 
'block5_conv3.' In the decoder this is simply reshaped to (196,512)
and then an average is done over the first dimension (K.mean line ~155 in decoder.py),
so the input to the model is a 512-d vector (for each image).


· Train_model

Train_model takes the input parameters in default.py and attn.yaml, reads
in the appropriate .npy file containing the transfer_values (the encoding
of the images with VGG16 output by encoder.ipynb) and trains the model
to generate captions. 

Text preprocessing involves taking the Flickr8k captions (5 for eaach image),
creating a vocabulary and tokenizing the captions. Padding is done to produce
captions of a fixed (maximum) length.

The decoder model is built based on parameters for gru vs lstm, attention (lstm only),
batch normalization and dropout. The routine decoder.py has the Decoder class
and builds the specified architecture.

Train_model uses a batch generator for producing the input to the model. This
is a method for asynchronously generating an infinite number of input values
(i.e. it runs until fit_generator doesn't need any more). This makes debugging
a little more difficult than when the data is set up in advance. Evidently
fit_generator is or will be deprecated by Keras and the usual "fit" will take
a parameter for input from a generator (versus passing the input data directly).

Checkpoints are defined for the model and passed to fit_generator. In particular,
fit_generator saves weights after epochs where the val_loss improves, but not
otherwise. Weights are saved in (for example):

./model_files/weights/VGG16_LSTM_Flickr8k_2l_32b_bn_dr_attn_bahdanau.hdf5

· Evaluation

evaluation.ipynb generates captions for images using the chosen model and its
stored weights. It can use tranfer values for images that have been pre-processed
by VGG16 (for example images that are in the Flickr8k dataset - the test images)
or it can call VGG16 to create a features vector on the fly for new images.

Section 8 of the code should be modified to read and input your chosen images. 
The predict.py routing contains the generate_caption function. This function, in
addition to outputting the caption, outputs the transfer values (i.e. the encoding
of the image with VGG16) which were computed on the fly or read from an .npy
file (see above).



Some additional details:

· Decoder

The model is defined in decoder.py, class Decoder. Model gru is simplistic, lstm is more
complicated and has possibility of attention (gru does not). One case (called in build_model) is:

def _connect_transfer_values_lstm_attention(self):

The print statements are on (in this function), and show up when decoder.py is called from train_model.ipynb.
Basically shows that (6000,196,512) gets averaged with "K.mean" = "backend.mean" along
the second axis. So you are left with a single 512-d vector (for each image). It then
seems that both c and s (the two LSTM memories) are set to that feature vector.

LSTM is iterated 30 times which is the maximum length of a
caption. (max_len in call to Decoder). The model summary (which can also be printed
out in train_model) is, when attention and lstm are on, very long and complicated. 
