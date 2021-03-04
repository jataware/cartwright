# Geotime Classify

This model is a type recurrent neural network that uses LSTM to learn text classification. The goal was for a given spreadsheet where we expect some kind of geospatial and temporal columns, can we automatically infer things like:

-   Country
-   Admin levels (0 through 3)
-   Timestamp (from arbitrary formats)
-   Latitude
-   Longitude
-   Which column likely contains the "feature value"
-   Which column likely contains a modifier on the  `feature`

 The model and transformation code can be used by downloading this repo and run from a command line or jupyter notebook, or installed as a pip package.


## Retraining geotime_classify with github repo
To get started locally first download the github repo 

    git clone https://github.com/jataware/geotime_classify.git
  
  You need to have conda install on your machine. Install conda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).  
  After conda is install you can create the conda environment by running this in your terminal.
  

    ./install_geotimeClassify.sh
   This should install all the dependencies that you will need to build and run the model locally.
   Now activate the newly created conda env.
   

    conda activate geotime_classify

To run the model all you need to do is start a python3 and import the code
## Modifying training data or labels
If there is a need to retrain this model with new data or new labels (categories) it is pretty simple, but there are a few things that need to be updated.
First step is adding the new data. This happens in the FakeDate class. You will need to add the new category (string) to the array called CHOICES. Once it is added  to CHOICES add that same category (string) to self.tag2id. You can add it to the end of the dict and give it the next sequential id. Make sure the new data is not introducing any new characters that are not in the self.token_set or self.token2id dict. Lastly if the new category is not a function provided by Faker you will need to add and elif statement in the datapoint function. Note* The lab in datapoint is randomly selected from CHOICES. 
Here is a line for an example:

    elif lab == 'date_%m_%d_%Y':
                val = get_fake_date(lab)
    elif lab == 'new category':
			    val = somehowCreateNewData
Now the val will be paired and added to the new dataset

    return {"hash" : lab, "obj" : val}

Make sure to add the new category to the all_categories array in the evaluate_test_set() function. This makes sure the confusion matrix has labels for all the categories. 

Once all of those updates are done instantiate the FakeData class, create a dataframe, and split it into train, test and validation sets. You can make this any size but 400000 seems to be large enough for this model

    f=FakeData()
	f.dataframe(size=400000)
	train_split, dev_split, test_split = f.split_data()

Next we can get some model inputs and mess around to perfect our model. Learning rate, char_dim, hidden_dim, weight_decay, batch_size and num_epochs are all parameters you can change, but the defaults here work well.

    char_vocab = f.get_char_vocab()
	tag_vocab = f.get_tag2id()
	char_vocab_size = len(char_vocab)
	char_dim = 128
	hidden_dim = 32
	learning_rate = .001
	weight_decay=1e-4 
	batch_size=32
	num_epochs=9

Next we define our pytorch model

	model = LSTMClassifier(char_vocab_size, char_dim, hidden_dim, len(tag_vocab))
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	
Train the model with our parameters

    model = train_model(model, optimizer, train_split, dev_split, char_vocab, tag_vocab, batch_size, num_epochs)

Evaluate the model

    evaluate_test_set(model, test_split, char_vocab, tag_vocab)
  Confusion Marix:
  ![Alt text](images/confusionMatrix.png?raw=true "Confusion Matrix")
  
Save the model if it performed well.
torch.save(model.state_dict(), 'models/whateverModelName.pth')