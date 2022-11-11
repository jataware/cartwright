---
layout: default
title: Contributing
nav_order: 8
has_toc: true
---

# Contributing to Cartwright

Contributions to Cartwright are welcome. Please read the following guidelines before contributing.

  

## General Guidelines

* Please follow the style of the code you are modifying.

* Please add tests for any new features.

* Please add documentation for any new features.

* Please add a changelog entry for any new features.

* Please add yourself to the list of contributors in the README.md file.

  
  

## Getting Started with Development

See [installation](./installation) for instructions on building repository from source for local development.

  
  

## Running Cartwright's Test Suite

Cartwright uses [pytest](https://docs.pytest.org/en/latest/) and [tox](https://tox.readthedocs.io/en/latest/) for testing. If you have installed Cartwright from source with all the dependencies, you can run pytest directly:

  

$ pytest

  

Alternatively you can use tox to run the test suite over all supported Python versions:

  

$ tox

  

Tox automatically creates a virtual environment for each Python version and runs the test suite in each environment. So tox is not reliant on your system Python version.

  
  
  

## Adding a New Category

To add a new category is a fairly simple process.

  

If you are created a new major category you can create a new file in categories dir. Once you create that file you will need to import the base class you want to use from CategoryBases file.

  

For each new category (basically like a label for training the model) you create a class that has the functions to `generate_training_data`, and `validate`.

`generate_training_data` needs to return the new category name, which needs to be the class name or for dates we use the format, and a randomly generated value that fits into that category. The `validate` function needs to take in one value and return true if it passes and nothing if it fails.

 
For for example lets add new category of fruits and vegetables

 - In cartwright/categories dir I create a new file
   fruits_vegetables.py.
  
 - Import the CategoryBase from CategoryBases.py file along with any other needed libraries.
 
 Create a class for fruits

  

    class fruits(CategoryBase):
	    def __init__(self):
		    super().__init__()
		    self.list_of_fruits=["apple","bannana", "pear", "orange"]
    
	    def generate_training_data(self):
		    return self.class_name(), np.random.sample(self.list_of_fruits)
	   
		def validate(self, value):
		    return value.lower() in self.list_of_fruits

  

We have a generate_training_data that will return our class name as the category label and a random value that is a fruit.

We also have a validate function that will return true if the value passed in is a fruit in our list.

  
  
  

## Adding a New Category Element

Adding a category element can be very simple depending on what you are adding.

  

Similar to creating a new Category you need to add a new class to the correct file in the categories/ dir. Make sure the new class has a generate_training_date and validate function and you are good to go.

  

## Training a New Model

Once you have your new category or category element we can train our model using the train_cartwright.py file.
  

    python3 -m cartwright.train_cartwright --version '0.0.2' --num_epochs 1

  
Decide on how long you want to train the model. I like to start with --num_epochs at one to make sure there are not any errors then when I am ready to fully train the model set --num_epochs to around 8. Set --version to the next model version in your models dir or you will overwright the last model.

Once the model is trained you can test the validation by running the new model. Make sure to set the model version in categorize.py file correctly before running.

  

    python3 -m cartwright.categorize path="path/to/test.csv"

  

Once that is complete and you are happy with the results you are done. You can make a PR for us to review if you think others would also find the new category or element useful.
