from nltk.corpus import stopwords
from gensim.models import Phrases
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim import similarities, matutils
from gensim.corpora import Dictionary
import numpy as np
from gensim.matutils import kullback_leibler, jaccard, hellinger

class Parent(object):

    def __init__(self, hotel_description, user_description, hotel_attributes, user_attributes, incremental=False,
                 num_incremental = 4):
        '''hotel_description = path of file
        user_description = path of file
        hotel_attributes = path of file
        user_attributes = path of file
        are list of lists, where each list corresponds to a different element that has been preprocessed and tokenized
        incremental indicates the use of the incremental user method
         num_incremental indicates the number of sentences per user query'''
        self.hotel_description = hotel_description
        self.user_description = user_description
        self.hotel_attributes = hotel_attributes
        self.user_attributes = user_attributes
        self.clean_hotel_description = self.preprocess(self.hotel_description, 'testorcone') #preprocessed
        self.clean_user_description = self.preprocess(self.user_description, 'richiestautente') #preprocessed
        self.clean_hotel_attributes = self.load_hotel_attributes() #preprocessed
        self.clean_user_attributes = self.load_user_attributes(self.user_attributes, 'richiestautente') #preprocessed
        self.dictionary = Dictionary(self.clean_hotel_description)
        self.vocab_new = dict()
        for k, v in self.dictionary.token2id.items():
            self.vocab_new[k]=v
        self.incremental= incremental
        if incremental == True:
            self.incremental_user = self.write_user_queriesXsentence(num_incremental)
            self.clean_user_description = self.preprocess("./datasets/incremental_user/queries"+str(num_incremental)+
                                                          ".txt", 'richiestautente')

        #make a switch to the user query part so that user description becomes that

    'Functions for loading and preprocessing files'

    def loader_docs(self, file, stop_word):
        # Load a text file, dividing it in different strings depending on stopword
        f = open(file, 'r')
        txt = f.readlines()
        f.close()

        l = ' '
        txt = [txt[i].strip() for i in range(len(txt))]
        txt = l.join(txt)
        txt = txt.split(stop_word)
        txt.pop()

        return txt

    def description_to_words(self, raw_review):
        # The input is a single string, and the output is a tokenized list preprocessed

        tokenizer = RegexpTokenizer(r'\w+')
        docs = raw_review.lower()  # Convert to lowercase.
        doc = tokenizer.tokenize(docs) # tokenize string

        # Remove stop words
        stops = set(stopwords.words("english"))
        doc = [w for w in doc if not w in stops]
        # Remove words of only one character
        doc = [token for token in doc if len(token) > 1]
        # Stem words
        snowball = SnowballStemmer('english')
        doc = [snowball.stem(token) for token in doc]
        return doc

    def get_bigrams(self, clean_docs):  # in place operation

        # Add bigrams to docs (only ones that appear 20 times or more).
        bigram = Phrases(clean_docs, min_count=20)
        for idx in range(len(clean_docs)):
            for token in bigram[clean_docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    clean_docs[idx].append(token)

        return clean_docs

    def preprocess(self, file, stop_word, activate_grams = 'Yes'):
        #preprocess txt file

        docs = self.loader_docs(file, stop_word)
        clean_docs = []

        for i in range(0, len(docs)):
            clean_docs.append(self.description_to_words(docs[i]))

        if activate_grams == 'Yes':
            clean_docs = self.get_bigrams(clean_docs)
        return clean_docs

    def load_hotel_attributes(self):
        #specific preprocess for hotel attributes
        clean_hotel_attributes = self.preprocess(self.hotel_attributes, 'nuovohotelinarrivo', 'NO')
        clean_hotel_attributes = [[token for token in doc if token != 'attributinuovi'] for doc in clean_hotel_attributes]
        return clean_hotel_attributes

    def load_user_attributes(self, file, stop_word):
        # specific preprocess for user attributes
        f = open(file, 'r')
        txt = f.readlines()
        f.close()
        snowball = SnowballStemmer('english')
        stops = set(stopwords.words("english"))
        #removing unnacesary parts of user_attributes
        stops.add('go')
        stops.add('play')
        stops.add('center')
        stops.add('centre')
        stops.add('nearby')
        stops.add('service')
        stops.add('do')
        stops.add('spot')
        tokenizer = RegexpTokenizer(r'\w+')

        txt = [txt[i].strip() for i in range(len(txt))]
        txt = [token for token in txt if token != '']
        txt = [token for token in txt if token != 'new_sentence']
        users = []
        user = []
        for i in range(len(txt)):
            if txt[i] == stop_word:
                users.append(user)
                user = []
            else:
                attribute = txt[i].lower()
                attribute = tokenizer.tokenize(attribute)
                attribute = [w for w in attribute if not w in stops]
                attribute = [token for token in attribute if len(token) > 1]
                attribute = [snowball.stem(token) for token in attribute]
                l = ' '
                attribute = l.join(attribute)
                user.append(attribute)
        return users

    def preprocess_special(self):
        'creates files of preprocessed hotel description and attributes used for evaluation'

        f = open('./datasets/pp/hotel_attributes.txt', 'w')
        for txt in self.clean_hotel_attributes:
            l = ' '
            txt = [txt[i].strip() for i in range(len(txt))]
            txt = l.join(txt)
            f.write(txt+'\n')
        f.close()
        f = open('./datasets/pp/hotel_descriptions.txt', 'w')
        for txt in self.clean_hotel_description:
            l = ' '
            txt = [txt[i].strip() for i in range(len(txt))]
            txt = l.join(txt)
            f.write(txt + '\n')
        f.close()
        f = open('./datasets/pp/user_queries.txt', 'w')
        for txt in self.clean_user_description:
            l = ' '
            txt = [txt[i].strip() for i in range(len(txt))]
            txt = l.join(txt)
            f.write(txt + '\n')
        f.close()

    def incremental_loader_docs(self, file):
        # Load a text file, dividing it in different strings for incremental user
        f = open(file, 'r')
        txt = f.readlines()
        f.close()
        hotels = []
        hotel = []
        for i in txt:
            line = i.strip()
            if line == 'richiestautente':
                hotels.append(hotel)
                hotel = []
            else:
                hotel.append(line)

        return hotels

    'General functions'
    def get_corpus(self):
        'get the Bag Of Words representation for the hotel_description and user_queries'
        corpus = [self.dictionary.doc2bow(doc) for doc in self.clean_hotel_description]
        BOW_user_queries = [self.dictionary.doc2bow(doc) for doc in self.clean_user_description]

        return corpus, BOW_user_queries

    def accuracy_query2hotel(self, hotel, user):
        # This function calculates how many attributes are satisfied in an hotel description
        # inputs: hotel_description index and user_description index
        # output: attributes satisfied in hotel description over total number of attributes

        tokenizer = RegexpTokenizer(r'\w+')
        count = 0
        f = open("./datasets/pp/hotel_descriptions.txt", 'r')
        clean_hotel_description = []
        for line in f:
            clean_hotel_description.append(line.split(' '))
        f.close()
        f = open("./datasets/pp/hotel_attributes.txt", 'r')
        clean_hotel_attributes = []
        for line in f:
            clean_hotel_attributes.append(line.split(' '))
        f.close()
        for user_attribute in self.clean_user_attributes[user]:
            part_count = 0
            user_attribute = tokenizer.tokenize(user_attribute)
            for part_attribute in user_attribute:

                if part_attribute in clean_hotel_description[hotel] or part_attribute in clean_hotel_attributes[hotel]:
                    part_count += 1
            if part_count == len(user_attribute):
                count += 1

        return count / len(self.clean_user_attributes[user])

    def make_accuracy_array(self, queryXhotel, num_best, bol=True):
        'for each user query it computes the accuracy of the 5 most similar hotels'
        self.preprocess_special()
        accuracy_array = np.zeros((len(self.clean_user_description), num_best))
        for i in range(np.shape(queryXhotel)[0]):
            ordered = matutils.argsort(queryXhotel[i], topn=5, reverse=bol)

            if self.incremental == False:
                for j in range(num_best):
                    accuracy = self.accuracy_query2hotel(ordered[j], i)
                    accuracy_array[i][j]= accuracy
            else:
                for j in range(num_best):
                    accuracy = self.accuracy_query2hotel(ordered[j], self.incremental_user[i])
                    accuracy_array[i][j]= accuracy
        return accuracy_array

    def get_overall_accuracy(self, accuracy_array, num_best=5):
        #It computes the average accuracy for each best hotel
        count = 0
        for i in range(num_best):
            overall_accuracy = np.sum(accuracy_array[:, i]) / accuracy_array.shape[0]
            print('recall ' + str(i + 1) + ' ' + str(overall_accuracy))
            count+= overall_accuracy
        return count

    def get_accuracy_array(self, hotel_match_X_query, num_best):
        '''for each user query it computes the accuracy of the 5 most similar hotels, works with gensim module cosine
        similarity and word's mover distance'''
        self.preprocess_special()
        accuracy_array = np.zeros((len(hotel_match_X_query), num_best))
        for i in range(len(hotel_match_X_query)):
            if self.incremental == False:
                for j in range(num_best):
                    accuracy = self.accuracy_query2hotel(hotel_match_X_query[i][j][0], i)
                    accuracy_array[i][j]= accuracy
            else:
                for j in range(num_best):
                    accuracy = self.accuracy_query2hotel(hotel_match_X_query[i][j][0], self.incremental_user[i])
                    accuracy_array[i][j]= accuracy
        return accuracy_array

    'Similarity functions'
    #A series of functions that interacts with the gensim modules

    def Jaccard_similiarity(self, corpus, corpus_model_user_description, num_best=5):
        'for each user query it computes the Jaccard coefficient with respect to each hotel'
        length = len(corpus_model_user_description)
        queryXhotel = np.zeros((length, len(corpus)))

        for i in range(length):
            for j in range(len(corpus)):
                queryXhotel[i][j] = jaccard(corpus_model_user_description[i],corpus[j])

        #np.save('jaccard_similiarity', queryXhotel)
        accuracy_array = self.make_accuracy_array(queryXhotel, num_best, bol=False)

        return accuracy_array

    def cosine_similarity(self, corpus, corpus_model_user_description, num_best=5):
        # corpus can be for example corpus_tfidf
        # corpus_model_user_description can be for example tfidf_user_queries:
        # num_best refers to the number of best hotels that will be considered
        #USES cosine similarity as implemented in Gensim
        index = similarities.MatrixSimilarity(corpus, num_best=num_best)
        hotel_match_X_query = []
        for query in corpus_model_user_description: #tfidf_user_queries:
            sims = index[query]
            hotel_match_X_query.append(sims)
        accuracy_array=self.get_accuracy_array(hotel_match_X_query, num_best)
        return accuracy_array

    def Hellinger_similiarity(self, corpus, corpus_model_user_description, num_best=5):
        'implements Hellinger similarity using gensim modules'
        length = len(corpus_model_user_description)
        queryXhotel=np.zeros((length, len(corpus)))
        print('It takes some time')
        for i in range(length):
            for j in range(len(corpus)):
                queryXhotel[i][j]=hellinger(corpus_model_user_description[i],corpus[j])
            print(i)
        #np.save('hellinger_similiarity', queryXhotel)
        accuracy_array = self.make_accuracy_array(queryXhotel, num_best, bol=False) #true?
        return accuracy_array

    def WMD_similiarity(self, corpus, w2v_model, corpus_model_user_description, num_best=5):
        'Word mover distance similarity'
        index = similarities.WmdSimilarity(corpus, w2v_model, num_best)
        hotel_match_X_query = []
        for query in corpus_model_user_description: #tfidf_user_queries::20
            sims = index[query]
            hotel_match_X_query.append(sims)
        accuracy_array = self.get_accuracy_array(hotel_match_X_query, num_best)
        return accuracy_array

    'second experiment'
    #The second experiment consists in analyzing the performance of the reccomendation system on user queries based on
    #a different amount of attributes

    def divide_query_per_num_attribute(self):
        'The queries are divided based on the number of attributes they contain'
        length_4 = []
        length_5 = []
        length_6 = []
        length_7 = []
        length_8 = []
        length_9 = []

        for i in range(len(self.clean_user_attributes)):
            length = len(self.clean_user_attributes[i])
            if length == 4:
                length_4.append(i)
            elif length == 5:
                length_5.append(i)
            elif length == 6:
                length_6.append(i)
            elif length == 7:
                length_7.append(i)
            elif length == 8:
                length_8.append(i)
            elif length == 9:
                length_9.append(i)
            else:
                print('Error,missing list of len = ' + str(len(i)))

        return length_4, length_5, length_6, length_7, length_8, length_9

    def get_accuracy_based_attributes(self,accuracy_array, length_array, num_best=1):
        'the accuracy is calculated for each user query based on the amount of attributes'

        for i in range(len(length_array)):
            numpy_length=np.array(length_array[i])
            sliced_array = accuracy_array[numpy_length]
            accuracy = 0
            for j in range(num_best):
                accuracy += np.sum(sliced_array[:,j])/sliced_array.shape[0]
            accuracy =accuracy/float(num_best)
            print('accuracy for ' + str(i + 4) + ' attributes ' + str(accuracy))

    'incremental user'
    def write_user_queriesXsentence(self, numb_sentences):
        'it divides the user query based on the amount of sentences'
        #input is the amount of sentences the query should have
        #output is a list with the index of user queries that satisfy the contraint and a text file containing all query
        f = open("./datasets/incremental_user/queries"+str(numb_sentences)+".txt", 'w')
        hotels = self.incremental_loader_docs(self.user_description)
        indexes = []
        for i in range(len(hotels)):
            if len(hotels[i]) > (numb_sentences-1):
                indexes.append(i)
                for j in range(numb_sentences):
                    sentence = hotels[i][j]
                    f.write(sentence+'\n')
                f.write('richiestautente\n')
        f.close()
        return indexes

