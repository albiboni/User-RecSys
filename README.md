# User-RecSys
Code for the project: "Analysis of Recommendation-systems based on User Preferences"
Abstract

This paper presents a recommender system able to understand user pref-erences, queries written in natural language.  The model is able to select fivedifferent options from a large set of multi-attribute alternatives.  In order tounderstand the user query, different natural language processing (NLP) mod-els have been tested from classic approaches such as Latent Semantic Analysis(LSA) to word embeddings.  We show first how these different models comparebetween each other and lastly that applying Tf-idf and subsequently the Jac-card coefficient performs better in this experiment.  A new dataset is createdfor training and evaluating the model since a compatible one is not available.The dataset contains descriptions and user queries written in natural language.Further, we show that the model can be used with a real user by simulating it,this means that information is provided to the recommendation system gradu-ally, instead of all at once.

Link:

Requirements

Python 3.6.2 https://www.python.org/downloads/

For crawler_booking

Selenium 3.6.0 http://www.seleniumhq.org/
Chromedriver 2.34 https://sites.google.com/a/chromium.org/chromedriver/downloads

Dataset Generation

Hotel descriptions and hotel features: crawler_booking
User queries and user features: query_generator

