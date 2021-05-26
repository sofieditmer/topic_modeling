#!/usr/bin/env python
"""
Info: This preprocessing script takes as input the Donald Trump tweets, trump_tweets.csv, and performs basic preprocessing steps to prepare them for topic modeling. 

Parameters:
    (optional) input_filename: str <name-of-input-file>, default = "trump_tweets.csv"
    (optional) output_filename: str <name-of-output-file>, default = "clean_trump_tweets.csv"

Usage:
    $ python 0-preprocessing.py
    
Output:
    - clean_trump_tweets.csv: a CSV-file containing clean, chunked tweets that can be used as input for the 1-topicmodeling.py
"""

### DEPENDENCIES ###

# core libraries
import sys
import os
sys.path.append(os.path.join(".."))

# pandas
import pandas as pd

# regex
import regex as re

# Argparse
import argparse

### MAIN FUNCTION ###

def main():
    
     ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Path to input file
    ap.add_argument("-i", "--input_filename",
                    type = str,
                    required = False, # not required argument
                    help = "Specify the name of input file",
                    default = "trump_tweets.csv") # default
    
    # Argument 2: Name of output file
    ap.add_argument("-o", "--output_filename",
                    type = str,
                    required = False, # not required argument
                    help = "Specify the name of the output file",
                    default = "clean_trump_tweets.csv") # default 

    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    input_file = os.path.join("..", "data", args["input_filename"])
    output_filename = args["output_filename"]
        
    # Start message
    print(f"\n[INFO] Initializing preprocessing of '{input_file}'...")
    
    # Instantiate the Preprocessing class
    preprocessing = Preprocessing(input_file)
    
    # Load data
    print(f"\n[INFO] Loading '{input_file}'...")
    tweets_df = preprocessing.load_data()
        
    # Clean tweets
    print("\n[INFO] Preprocessing the data...")
    tweets_df_clean = preprocessing.clean_tweets(tweets_df)
    
    # Save preprocessed file as CSV-file to data folder
    print(f"\n[INFO] Saving '{output_filename}' to 'data' folder...")
    preprocessing.save_to_csv(tweets_df_clean, output_filename)
    
    # Done
    print(f"\n[INFO] Done! You have now cleaned '{input_file}'. It has been saved as '{output_filename}' in the 'data' folder.\n")

    
### PREPROCESSING ### 
    
# Creating Preprocessing class 
class Preprocessing:
    
    # Intialize Preprocessing class
    def __init__(self, input_file):
        
        # Receive input
        self.input_file = input_file
    
    
    def load_data(self):
        """
        This method loads the data into a dataframe, selects the relevant columns, and sorts by date resulting in a dataframe
        containing the tweets in chronological order.
        """
        # Load data into dataframe
        tweets_df = pd.read_csv(self.input_file, lineterminator = "\n")
        
        # Select the relevant columns 
        tweets_df = tweets_df.loc[:, ("id", "content", "date")]
    
        # Sort by date so the tweets appear in chronological order
        tweets_df = tweets_df.sort_values(by="date")
        
        return tweets_df
        
        
    def clean_tweets(self, tweets_df):
        """
        This method cleans the tweets using the regex module and saves the cleaned tweets in a new columns in the dataframe. 
        This method was highly inspired by this thread on StackOverflow: 
        https://stackoverflow.com/questions/64719706/cleaning-twitter-data-pandas-python 
        """
        # Create empty container for the clean tweets
        clean_tweets = []

        # Loop through all tweets and perform preprocessing steps
        for tweet in tweets_df["content"]:
    
            # Lowercase
            tweet = tweet.lower()
            
            # Remove @
            tweet = re.sub("@[A-Za-z0-9]+","", tweet)
            
            # Remove hastags but keep text that follows
            tweet = tweet.replace("#", "").replace("_", " ")
    
            # Remove re-tweets (RTs) - I am only interested in his own tweets
            tweet = re.sub('RT[\s].+', '', tweet)
            
            # Remove links
            tweet = re.sub('https?:\/\/\S+', '', tweet)
    
            # Remove image urls
            tweet = re.sub('pic\.twitter\.com.[A-Za-z0-9]{10}', '', tweet)
    
            # Append to clean_tweets
            clean_tweets.append(tweet)
        
        # Create new column in dataframe and save the clean tweets
        tweets_df["clean_tweets"] = clean_tweets
        tweets_df_clean = tweets_df
        
        return tweets_df_clean

    
    def save_to_csv(self, tweets_df_clean, output_filename):
        """
        This method saves the dataframe containing the clean tweets as a csv-file in the data directory.
        """
        # Output path
        output_file = os.path.join("..", "data", output_filename)
    
        # Save dataframe as CSV in the data folder
        tweets_df_clean.to_csv(output_file,
                               index=True, 
                               encoding="utf-8")
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()