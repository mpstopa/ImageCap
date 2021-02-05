# ImageCap
# summary
This repository contains code for a deep learning model that takes an image as an input and
generates a caption.

The implemented models follow the Encoder-Decoder framework and some of the decoder models also use Attention mechanisms. 
The models are based on and trained with the Flickr8k dataset. Typically, of the 8000 images in the dataset, 6000 are
used for training, 1000 are used for validation and 1000 are used for testing.

The program uses [YACS library](https://github.com/rbgirshick/yacs) for managing configurations. The architectures and hyperparameters are managed using ```configs``` directory: configs/default.py``` contains a default set of hyperparameters. The file attn.yaml overrides the default parameters and so modifying default.py itself is not necessary.

The following steps are the recommendations on how to use the code to train and evaluate the models:
1. Define configurations of your model and create a yaml file in ```configs``` directory. This configuration should be then passed to the ```config_file``` variable in all the notebooks.
2. Launch ```encoder.ipynb``` and run all cells to encode image features using VGG16 encoder network. Please, note that you should specify the path to the dataset in the configurations file. Also, features for models with attention and without it are taken from the different layers of the network.
3. Run all cells in ```train_model.ipynb``` to train the model from your configurations.
4. The model might be tested in ```evaluation.ipynb```. Evaluation.ipynb can be modified to input any image desired. Generally the
image files are preprocessed to have the same size as those in the Flickr8k dataset (3,224,224) so most any image file will work.
