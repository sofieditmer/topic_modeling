# Assignment 5: Topic Modeling

### Description of Task: Applying Unsupervised Machine Learning to Text Data
This assignment was assigned by the course instructor as “Assignment 5 – (Un)supervised Machine Learning”. The purpose of this assignment was to apply unsupervised machine learning on a self-chosen dataset containing textual data of some kind. We could choose to either train a text classifier on this text data to predict associated labels or train an LDA model to extract structured information that can provide insight into the chosen data. <br>
For this assignment I chose to train an LDA topic model on all Donald Trump tweets written between May 2009 and June 2020; a dataset available on [Kaggle](https://www.kaggle.com/austinreese/trump-tweets). The research question I was interested in was whether it would be possible to find meaningful and consistent topic within the tweets and how the content of the tweets changes over time in regard to which topics seem to be most dominating over time. Hence, the purpose of this assignment was to select an interesting dataset, formulate a research question, and perform unsupervised machine learning.  

__Research Question__ <br>
Is it possible to find meaningful and consistent topics within the tweets of Donald Trump posted between 2009 and 2020, and how does the content of the tweets change over time?

### Content and Repository Structure <br>
If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. It is important to remark that all the code that has been produced has only been tested in Linux and MacOS. Hence, for the sake of convenience, I recommend using a similar environment to avoid potential problems. <br>
The repository follows the overall structure presented below. The two scripts, ```0-preprocessing.py``` and ```1-topicmodeling.py```, are located in the ```src``` folder. The full dataset is provided in the ```data``` folder, and the outputs produced when running the scripts can be found within the ```output``` folder. The README file contains a detailed run-through of how to engage with the code and reproduce the contents.

| Folder | Description|
|--------|:-----------|
| ```data``` | A folder containing the full dataset. 
| ```src``` | A folder containing the python scripts for the particular assignment.
| ```output``` | A folder containing the outputs produced when running the python scripts within the src folder.
| ```utils```| A folder containing topic modeling utility scripts that were developed for use in class and modified for this particular assignment.
| ```requirements.txt```| A file containing the dependencies necessary to run the python script.
| ```create_lda_venv.sh```| A bash-file that creates a virtual environment in which the necessary dependencies listed in the ```requirements.txt``` are installed. This script should be run from the command line.
| ```LICENSE``` | A file declaring the license type of the repository.


### Usage and Technicalities <br>
To reproduce the results of this assignment, the user will have to create their own version of the repository by cloning it from GitHub. This is done by executing the following from the command line: 

```
$ git clone https://github.com/sofieditmer/topic_modeling.git  
```

Once the user has cloned the repository, a virtual environment must be set up in which the relevant dependencies can be installed. To set up the virtual environment and install the relevant dependencies, a bash-script is provided, which creates a virtual environment and installs the dependencies listed in the ```requirements.txt``` file when executed. To run the bash-script that sets up the virtual environment and installs the relevant dependencies, the user must first navigate to the topic modeling repository:

```
$ cd topic_modeling
$ bash create_lda_venv.sh 
```

Once the virtual environment has been set up and the relevant dependencies listed in ```requirements.txt``` have been installed within it, the user is now able to run the scripts provided in the ```src``` folder directly from the command line. In order to run the script, the user must first activate the virtual environment in which the script can be run. Activating the virtual environment is done as follows:

```
$ source lda_venv/bin/activate
```

Once the virtual environment has been activated, the user is now able to run the scripts within it:

```
(lda_venv) $ cd src

(lda_venv) $ python 0-preprocessing.py

(lda_venv) $ python 1-topicmodeling.py
````

For the ```0-preprocessing.py``` script the user is able to modify the following parameters, however, this is not compulsory:

```
-i, --input_data: str <name-of-input-data>, default = "trump.tweets.csv"
-o, --output_filename: str <name-of-output-file>, default = "clean_trump_tweets.csv"
```

For the ```1-topicmodeling.py``` script the user is able to modify the following parameters, however, once again this is not compulsory:

```
-i, --input_data: str <name-of-input-data>, default = "clean_trump.tweets.csv"
-c, --chunk_size: int <size-of-chunks>, default = 10
-p, --n_passes: int <number-of-passes>, default = 10
-m, --min_count: int <minimum-count-bigrams>, default = 2
-th, --threshold: int <threshold-for-keeping-phrases>, default = 100
-it, --iterations: int <number-of-iterations>, default = 100
-r, --rolling_mean: int <rolling-mean>, default = 50
-s, --step_size: int <size-of-steps>, default = 5
```

The abovementioned parameters allow the user to adjust the analysis of the input data, if necessary, but default parameters have been set making the script run without explicitly specifying these arguments. The user is able to modify the number of tweets to chunk together, the number of passes that model makes through the entire dataset, the number of times a bigram should occur in order to be included in the model, the threshold value that determines which phrases to keep and which to exclude, the number of iterations which specifies the number of times that the model goes through a single document, the rolling mean which specifies the number of chunk to calculate the mean from, and the step-size that is used when calculating the coherence value of different number of topics. 

### Output <br>
When running the ```0-preprocessing.py```script, the following files will be saved in the ```data``` folder: 
1. ```clean_trump_tweets.csv``` a CSV-file containing clean, chunked tweets that can be used as input for the 1-topicmodeling.py

When running the ```1-topicmodeling.py```script, the following files will be saved in the ```output``` folder: 
1. ```topics.txt``` Overview of topics generated by the LDA model
2. ```dominant_topic.csv``` Table showing the most dominant topics and their associated keywords as well as how much each topic contributes. 
3. ```topic_contributions.csv``` Dataframe showing the most contributing keywords for each topic.
4. ```topics_over_time.jpg``` Visualization of the topic contributions over time.
5. ```topic_wordclouds.png``` Topics visualized as word clouds.

### Discussion of Results <br>
The first result to remark on is the number of topics with the highest coherence score (see figure 1). Evidently, 10 topics seem to give the most coherent topics for this particular dataset. It would seem that increasing the possible number of topics to 15 would perhaps yield an even greater coherence score. However, this was tested, and the coherence score did not increase significantly, and it was therefore decided to keep the limit to 10 topics to ensure clarity.

<img src="https://github.com/sofieditmer/topic_modeling/blob/main/output/n_topics_coherence.jpg" width="500">
Figure 1: Coherence values calculated for different number of topics. <br> <br>

The model obtained a perplexity score of -7.8 and a coherence score of 0.39. The perplexity score tells us something about how well the model performed, i.e., how “surprised” it was when encountering new data. Hence, the relatively low perplexity score indicates that surprisal is low. The coherence value indicates how coherent the topics are. A high coherence value tells us that the topics actually correspond to something meaningful in the data. A coherence score of 0.39 in this case tells us that the topics are relatively interpretable, however, ideally we would want a higher coherence value to make the topics more interpretable. <br>

Figure 2 contains an overview of the 10 topics visualized as word clouds. When assessing the keywords for each topic, it seems that the tweets of Donald Trump in fact do display relatively well-defined topics. For instance, topic 4 seems to relate to the wall Donald Trump intended to build between the U.S. and Mexico, while topic 8 seems to concern fake news. However, other topics are less distinct and certain words seem to appear across different topics such as “thank” and “great” suggesting that the data could benefit from further preprocessing. 

<img src="https://github.com/sofieditmer/topic_modeling/blob/main/output/topic_wordclouds.png" width="1000">
Figure 2: Overview of topics visualized as word clouds. <br> <br>

A more extensive overview of the topics is provided by the pyLDAvis visualization, which also provides a measure of overall term frequency for each as well as a measure of term frequency for the words within each topic . 
When assessing the contribution of the 10 topics over time, it becomes evident that the content of Donald Trump’s tweet changes (see figure 3). Topic 7 which includes keywords such as “realdonaldtrump” which is Donald Trump’s twitter name, “great” and “thank” seems to be a dominating topic over time. However, at some point topic 0 seems to take over in terms of contribution. This topic includes keywords such as “great”, “people”, “country”, and “job”. Although these two dominating topics seem very similar, one could argue that Donald Trump goes from tweeting mostly about himself to tweeting more about the “people” and the “country”, which indicates a slightly shift in focus. However, while topic 7 reaches a percentage contribution of 70% over time, topic 0 only reaches around 45%.

<img src="https://github.com/sofieditmer/topic_modeling/blob/main/output/topics_over_time.jpg" width="1000">
Figure 3: Topic contribution over time. <br> <br>

To sum up and answer the initial question asked; whether it would be possible to find meaningful and consistent topics within the tweets of Donald Trump between 2009 and 2020, and whether the tweets change in content over time, it seems that relatively consistent topics can be found, however, the interpretation of the topics require contextual knowledge of Trump’s political activity. When considering the topic contributions over time, two topics are clearly dominating the content of Donald Trump’s twitter account, but these topics are very similar and possibly even too general to infer meaningful conclusions from.


### License <br>
This project is licensed under the MIT License - see the [LICENSE](https://github.com/sofieditmer/topic_modeling/blob/main/LICENSE) file for details.

### Contact Details <br>
If you have any questions feel free to contact me on [201805308@post.au.dk](201805308@post.au.dk)