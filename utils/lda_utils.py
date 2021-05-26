#!/usr/bin/python
"""
This script stores utility functions for working with LDA using gensim. These functions were developed for use in class and modified for this particular project.
"""
 
### DEPENDENCIES ###

# NLP tools
import os 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# pandas
import pandas as pd

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# matplotlib
import matplotlib.pyplot as plt


### UTILITY FUNCTIONS ###

def process_words(texts, nlp, bigram_mod, trigram_mod, stop_words=stop_words, allowed_postags=['NOUN', "ADJ", "VERB", "ADV"]):
    """
    This method removes stopwords, forms bigrams, trigrams and performs lemmatization. 
    """
    # Use gensim simple preprocess
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    
    # Lemmatize and POS tag using spaCy
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags]) 
    
    return texts_out


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    # Calculate coherence values
    # Create empty lists that can be appended to
    coherence_values = []
    model_list = []
    
    # Compute coherence values for each number of topics specified with a particular stepsize (e.g. 5)
    for num_topics in range(start, limit, step):
        
        # Create LDA model
        model = gensim.models.LdaMulticore(corpus=corpus, 
                                           num_topics=num_topics, 
                                           id2word=dictionary)
        # Append to list
        model_list.append(model)
        
        # Compute coherence values
        coherencemodel = CoherenceModel(model=model, 
                                        texts=texts, 
                                        dictionary=dictionary, 
                                        coherence='c_v')
        
        # Append to list
        coherence_values.append(coherencemodel.get_coherence())
    
    # Create plot that visualizes the coherence values (y-axis) and the number of topics (x-axis)
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(os.path.join("..", "output", "n_topics_coherence.jpg"))
    
    return model_list, coherence_values


def format_topics_sentences(ldamodel, corpus, texts):
    """
    This method extracts the dominant topic per chunk, the contribution of the topics in percentage, 
    and the keywords for each topic.
    """
    # Create empty dataframe
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        
        # Prin
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), 
                                                                  round(prop_topic,4), 
                                                                  topic_keywords]), ignore_index=True)
            
            else:
                
                break
    
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    
    return(sent_topics_df)


if __name__=="__main__":
    pass