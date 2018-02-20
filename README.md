# Analysis of Recommendation-systems based on User Preferences
This repository contains the code for the project: "Analysis of Recommendation-systems based on User Preferences", 
which I worked on during my exchange semester at the École Polytechnique Fédérale de Lausanne.

## Abstract

This paper presents a recommender system able to understand user preferences, queries written in natural language. 
The model is able to select five different options from a large set of multi-attribute alternatives. In order to 
understand the user query, different natural language processing (NLP) models have been tested from classic approaches 
such as Latent Semantic Analysis (LSA) to word embeddings. We show first how these different models compare between 
each other and lastly that applying Tf-idf and subsequently the Jaccard coefficient performs better in this experiment. 
A new dataset is created for training and evaluating the model since a compatible one is not available. The dataset 
contains descriptions and user queries written in natural language. Further, we show that the model can be used with a 
real user by simulating it, this means that information is provided to the recommendation system gradually, instead of 
all at once.

## Dataset
The dataset is composed of hotel descriptions, hotel features, user queries and user features.

Hotel descriptions and hotel features are crawled from booking.com( crawler_booking.py)

User queries and user features are generated with the query_generator.py

## Recommender system
The folder recommender system presents the different recommendation model used and a dataset which includes the hotels 
in the region of the lac Léman (Lausanne and Geneva).

## Links
Project presentation: https://mega.nz/#!sDAWwAbD!CI2-3pkmOrOwqVZqDZCG0q-aK3bdVe_ORYlaW7LfMVQ

Project paper: https://mega.nz/#!kXYCibwQ!dNKAUjSSOogtdjzIysgmoeFPlkVMjsatyJoRTI6Tz94
### Requirements

Python 3.6.2 https://www.python.org/downloads/

For crawler_booking:

Selenium 3.6.0 http://www.seleniumhq.org/

Chromedriver 2.34 https://sites.google.com/a/chromium.org/chromedriver/downloads

For recommender_system:

Gensim https://radimrehurek.com/gensim/index.html

NLTK http://www.nltk.org

