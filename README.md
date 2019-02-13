# aipnd-image-classifier

This is my solution to the [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) Image Classifier project.

#### Principle Objectives

1. Create a image classifier using a pretrained model in dataset of flower images
2. Convert the classifier to a command line application

#### Dependencies

* Install [Python 3.6.3](https://www.python.org/downloads)
* Install [Jupyter 5.7.0](https://jupyter.org/install)


#### Development

1. Clone the project
2. Run `jupyter notebook`
3. Edit `Image Classifier Project.ipynb`

**Note:** I suggest to train this model using GPU

#### How To Use

Train the network using train.py

```
python train.py data_directory
```

Set directory to save checkpoints

```
python train.py data_dir --save_dir save_directory
```

Choose architecture (alexnet, densenet121, vgg16)

```
python train.py data_dir --arch "vgg16"
```

Set hyperparameters

```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
```
Use GPU for training

```
python train.py data_dir --gpu
```
Predict flower name from an image with predict.py

```
Basic usage: python predict.py /path/to/image checkpoint
```

Return top KK most likely classes

```
python predict.py input checkpoint --top_k 3
```

Use a mapping of categories to real names

```
python predict.py input checkpoint --category_names cat_to_name.json
```

Use GPU for inference

```
python predict.py input checkpoint --gpu
```
