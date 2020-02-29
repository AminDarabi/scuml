# Simple cuda machine learning library

At the moment, this project could train a feed-forward neural network with a given data set, using a hybrid method based genetic algorithm and Backpropagation. it also could use trained network to guess a testset classes.

This project developed with CUDA to executing on gpu. so you need nvidia gpu, nvcc and CUDA requirements in your machine to be able tp compile and run it.

## Compile

Compile the component using :

```sh
$ make
```

## Install

You are able to install it on your machin using :

```sh
$ make install #run as root user (use sudo)
```

## Using it for training

You must have a csv file that contain all float values, with respectively, exactly one indexes-column, X features column and Y classes column that need to predict.

Output is out net file.
```sh
$ scuml train X:*:..:*:Y csvfile [GA_population_number] [generation_number] [BP_lambda] [BP_learning_rate] [BP_number_per_generation] [mutate_rate]
```

## Using it for testing

You must have a csv file that contain all float values, with respectively, exactly one indexes-column, X features column and you also need .net file (output of trainig part).

Output is CSVout file.

```sh
$ scuml test *.net csvfile
```