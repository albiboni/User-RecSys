from models import *
import numpy
from reccomender import *

numpy.random.seed(3)
hotel_description = "./datasets/hotel_descriptions.txt"
user_description = "./datasets/user_queries.txt"
hotel_attributes = "./datasets/hotel_attributes.txt"
user_attributes = "./datasets/user_attributes.txt"

tfidf = Tf_idf(hotel_description, user_description,hotel_attributes, user_attributes, incremental=False)
print(tfidf.results_tfidf())
#LSI = LSI(hotel_description, user_description,hotel_attributes, user_attributes)
#print(LSI.results_lsi(tf_idf='No', num_topics=305)) #better with tf-idf
#print(LSI.results_lsi(tf_idf='Yes', num_topics=300, num_best=5)) #best
#LDA = LDA(hotel_description, user_description,hotel_attributes, user_attributes)
#print(LDA.results_LDA(tf_idf='No', num_topics=250))
#D2V = Doc2Vec(hotel_description, user_description,hotel_attributes, user_attributes)
#print(D2V.d2v(pre_train='No'))
#WMDistance= WMoverD(hotel_description, user_description,hotel_attributes, user_attributes)
#print(WMDistance.WMD(5))
#FT = FastText(hotel_description, user_description,hotel_attributes, user_attributes)
#print(FT.results_fasttext(tf_idf='no'))