from reccomender import *
from gensim import models, matutils, similarities
import math as m
import random
import multiprocessing
import pickle
from time import time
from gensim.models.fasttext import FastText
'num_best always refers to the number of best hotels considered'
class Tf_idf(Parent):
    'implements Tf_idf model'
    def my_wglobal(self, doc_freq, total_docs):
        'general idf weight'
        return m.log10(total_docs / doc_freq)

    def ccc(self, doc_freq, total_docs):
        return 1.

    def my_wlocal(self, term_freq):
        'general tf weight'
        return m.log1p(term_freq)+16. #+16 probably overfit

    def init_tfidf(self):

        corpus, BOW_user_queries = self.get_corpus()
        tfidf = models.TfidfModel(corpus, wlocal=m.log1p, wglobal=self.ccc, normalize=False)
        tfidf_query = models.TfidfModel(corpus, wlocal=self.my_wlocal, wglobal=self.ccc, normalize=False)
        corpus_tfidf = tfidf[corpus]
        tfidf_user_queries = tfidf_query[BOW_user_queries]

        return corpus_tfidf, tfidf_user_queries

    def results_tfidf(self, num_best=5):
        corpus_tfidf, tfidf_user_queries = self.init_tfidf()
        'switch between cosine similarity and jaccard coefficient'
        accuracy_array = self.cosine_similarity(corpus_tfidf, tfidf_user_queries, num_best)
        #accuracy_array = self.Jaccard_similiarity(corpus_tfidf, tfidf_user_queries, num_best)
        count= self.get_overall_accuracy(accuracy_array, num_best)
        'second experiment'
        #length_array = self.divide_query_per_num_attribute()
        #self.get_accuracy_based_attributes(accuracy_array, length_array)
        return count

class jaccard(Parent):
    'simple model that uses the jaccard coefficient on the bag of words'
    def results_jack(self, num_best = 5):
        corpus, BOW_user_queries = self.get_corpus()
        accuracy_array = self.Jaccard_similiarity(corpus, BOW_user_queries, num_best)
        count = self.get_overall_accuracy(accuracy_array, num_best)
        length_array = self.divide_query_per_num_attribute()
        self.get_accuracy_based_attributes(accuracy_array, length_array)
        return count

class LSI(Tf_idf):
    'implements Latent semantic analysis'
    def init_LSI(self, tf_idf='No', num_topics=10):
        if tf_idf == 'Yes':
            corpus, BOW_user_queries = self.init_tfidf()
        else:
            corpus, BOW_user_queries = self.get_corpus()
        lsi = models.LsiModel(corpus, id2word=self.dictionary, num_topics=num_topics, onepass=False, power_iters=10)
        corpus_lsi = lsi[corpus]

        lsi_user_queries = lsi[BOW_user_queries]
        return corpus_lsi, lsi_user_queries

    def results_lsi(self, tf_idf='No', num_topics=10, num_best=5):
        corpus_lsi, lsi_user_queries = self.init_LSI(tf_idf, num_topics)
        'switch between cosine similarity and jaccard coefficient'
        accuracy_array = self.cosine_similarity(corpus_lsi, lsi_user_queries, num_best)
        #accuracy_array = self.Jaccard_similiarity(corpus_lsi, lsi_user_queries, num_best)
        count = self.get_overall_accuracy(accuracy_array, num_best)
        'second experiment'
        #length_array= self.divide_query_per_num_attribute()
        #self.get_accuracy_based_attributes(accuracy_array,length_array)
        return count

class LDA(Tf_idf):
    'implements latent dirichlet allocation '
    def init_LDA(self, tf_idf='No', num_topics=10):
        if tf_idf == 'Yes':
            corpus, BOW_user_queries = self.init_tfidf()
        else:
            corpus, BOW_user_queries = self.get_corpus()
        LDA = models.LdaModel(corpus, id2word=self.dictionary, num_topics=num_topics, #offset=1.1,
                              passes=5, iterations=400,decay=0.5,alpha='symmetric', eta=0.001, eval_every=None, random_state=3)
        #print(LDA.show_topics())
        corpus_LDA = LDA[corpus]
        LDA_user_queries = LDA[BOW_user_queries]
        return corpus_LDA, LDA_user_queries

    def results_LDA(self, tf_idf='No', num_topics=200, num_best=5):
        corpus_LDA, LDA_user_queries = self.init_LDA(tf_idf, num_topics)
        'switch between cosine similarity and Hellinger similarity'
        accuracy_array = self.Hellinger_similiarity(corpus_LDA, LDA_user_queries,num_best)
        #accuracy_array = self.cosine_similarity(corpus_LDA, LDA_user_queries, num_best)
        count = self.get_overall_accuracy(accuracy_array,num_best)
        'second experiment'
        length_array = self.divide_query_per_num_attribute()
        self.get_accuracy_based_attributes(accuracy_array, length_array)
        return count

class Doc2Vec(Parent):

    def d2v(self, pre_train='Yes', num_best=5):
        #implements doc2vec using pretrained embeddings and training a model
        if pre_train == 'Yes':
            d2v = models.doc2vec.Doc2Vec.load('./news/doc2vec.bin') #pretrained vectors found online, lower performance
            inferred_hotel = np.zeros((len(self.clean_hotel_description), 300))
            i = 0
            for hotel in self.clean_hotel_description:
                inferred_hotel[i] = matutils.unitvec(d2v.infer_vector(hotel, alpha=0.01, steps=500))
                i += 1
            d = 0
            accuracy_array = np.zeros((len(self.clean_user_description), len(self.clean_hotel_description)))
            for query in self.clean_user_description:
                inferred_query = matutils.unitvec(d2v.infer_vector(query, alpha=0.01, steps=500))
                for j in range(np.shape(inferred_hotel)[0]):
                    sim = np.dot(inferred_query, inferred_hotel[j])
                    accuracy_array[d][j] = sim
                d += 1
            accuracy_array = self.make_accuracy_array(accuracy_array, num_best)
            count = self.get_overall_accuracy(accuracy_array)
        else:
            documents = models.doc2vec.TaggedLineDocument('./datasets/pp/hotel_descriptions.txt')
            d2v = models.doc2vec.Doc2Vec(min_count=5, window=5, size=200, sample=1e-2, dm_mean=0,
                                         dm=0, negative=15, workers=1, iter=10, seed=3)

            d2v.build_vocab(documents)

            for epoch in range(10):
                shuffled = list(documents)
                random.shuffle(shuffled)
                d2v.train(shuffled, total_examples=d2v.corpus_count, epochs=d2v.iter)

            accuracy_array = []

            for query in self.clean_user_description:
                inferred_query = d2v.infer_vector(query, alpha=0.01, steps=100)

                sims = d2v.docvecs.most_similar([inferred_query], topn=num_best)
                accuracy_array.append(sims)
            accuracy_array = self.get_accuracy_array(accuracy_array, num_best)

            count = self.get_overall_accuracy(accuracy_array)
            '2 experiment'
            length_array = self.divide_query_per_num_attribute()
            self.get_accuracy_based_attributes(accuracy_array, length_array)
            d2v.save('model0')

        return count

class WMoverD(Parent):
    'implements the word mover distance'
    def doc_W2V(self, list):
        doc_W2V = []
        for txt in list:
            l = ' '
            txt = [txt[i].strip() for i in range(len(txt))]
            txt = l.join(txt)
            doc_W2V.append(txt)
        return doc_W2V

    def WMD(self,num_best=5):
        W2V_corpus = self.doc_W2V(self.clean_hotel_description)
        W2V_test = self.doc_W2V(self.clean_user_description)
        W2V = models.word2vec.Word2Vec(W2V_corpus, size=200, window=5, min_count=5,
                                       sample=1e-2, seed=3, workers=4, sg=1, hs=0, negative=15, iter=100)
        accuracy_array = self.WMD_similiarity(W2V_corpus, W2V, W2V_test, num_best)
        count = self.get_overall_accuracy(accuracy_array, num_best)
        'second experiment'
        length_array = self.divide_query_per_num_attribute()
        self.get_accuracy_based_attributes(accuracy_array, length_array)
        return count

class FastText(Tf_idf):
    'uses fasttext pretrained word_embeddings on wikipedia, read read_me in  fasttext'
    def get_doc_arrays_tfidf(self, file, train_arrays, corpus_tfidf,size):
        'computes average wordvectors with tf-idf weights'
        with open('./datasets/fasttext/embeddings/vocab_300d.pkl', 'rb') as f:
            vocab = pickle.load(f)
        embeddings = np.load('./datasets/fasttext/embeddings/embeddings_300d.npy')
        counter = 0
        f = open(file, 'r')
        for line in f:
            line_tfidf = corpus_tfidf[counter]
            weight_line_tfidf =[line_tfidf[i][1] for i in range(len(line_tfidf))]
            idx_line_tfidf = [line_tfidf[i][0] for i in range(len(line_tfidf))]
            sum_vect = 0
            len_doc = 0
            for t in line.strip().split():
                idxs_tweet = vocab.get(t, -1)
                idxs_tfidf = self.vocab_new.get(t, -1)
                if idxs_tfidf >= 0 and idxs_tweet >= 0:

                    embedding = embeddings[idxs_tweet]
                    weight_tfidf = weight_line_tfidf[idx_line_tfidf.index(idxs_tfidf)]
                    embedding = embedding*weight_tfidf
                    len_doc += 1.
                    sum_vect += embedding
                elif idxs_tfidf >= 0 and idxs_tweet < 0:
                    weight_tfidf = weight_line_tfidf[idx_line_tfidf.index(idxs_tfidf)]
                    embedding = weight_tfidf*np.ones(size)
                    len_doc += 1.
                    sum_vect += embedding
                elif idxs_tfidf < 0 and idxs_tweet >= 0:
                    embedding = embeddings[idxs_tweet]
                    len_doc += 1.
                    sum_vect += embedding
                #len_doc +=1
                #sum_vect +=embedding
            sum_vect_avg = sum_vect/len_doc
            train_arrays[counter] = sum_vect_avg
            counter += 1
        f.close()
        return train_arrays

    def get_doc_arrays(self, file, train_arrays):
        'computes average wordvectors'
        with open('./datasets/fasttext/embeddings/vocab_300d.pkl', 'rb') as f:
            vocab = pickle.load(f)
        embeddings = np.load('./datasets/fasttext/embeddings/embeddings_300d.npy')
        counter = 0
        f = open(file, 'r')
        for line in f:
            idxs_tweet = [vocab.get(t, -1) for t in line.strip().split()]  # get idx of each tweet in vocabulary
            idxs_tweet = [t for t in idxs_tweet if t >= 0]  # performs a check for words not present in vocabulary
            idxs2vects = [embeddings[i] for i in idxs_tweet]  # trasforms idx vocabulary in wordvect
            if len(idxs_tweet) > 0:
                sum_vect_avg = sum(idxs2vects)/ len(idxs_tweet)  # gives array of full tweet if any
                train_arrays[counter] = sum_vect_avg
                counter += 1

        f.close()
        return train_arrays

    def re_process(self, f):
        txt = f.readlines()
        new_docs = []

        for i in txt:
            b = i.strip().split(' ')

            new_docs.append(b)

        return new_docs

    def get_tfidf(self):
        f = open("./datasets/pp/hotel_descriptions.txt", 'r')
        g = open("./datasets/pp/user_queries.txt", 'r')
        hot_descr= self.re_process(f)
        user_descr = self.re_process(g)
        corpus = [self.dictionary.doc2bow(doc) for doc in hot_descr]
        BOW_user_queries = [self.dictionary.doc2bow(doc) for doc in user_descr]
        tfidf = models.TfidfModel(corpus, wlocal=m.log1p, wglobal=self.ccc, normalize=True)
        tfidf_query = models.TfidfModel(corpus, wlocal=self.my_wlocal, wglobal=self.ccc, normalize=False)
        corpus_tfidf = tfidf[corpus]
        tfidf_user_queries = tfidf_query[BOW_user_queries]

        return corpus_tfidf, tfidf_user_queries



    def get_accuracy_array_special(self,size, tf_idf):
        corpus_tfidf, tfidf_user_queries = self.get_tfidf()
        hotel_path ="./datasets/pp/hotel_descriptions.txt"
        hotel_zero = np.zeros((741, size))
        user_path = "./datasets/pp/user_queries.txt"
        user_zero = np.zeros((1000, size))
        if tf_idf=='yes':
            print('tfidf')
            array_hotel = self.get_doc_arrays_tfidf(hotel_path, hotel_zero, corpus_tfidf, size)
            array_user = self.get_doc_arrays_tfidf(user_path, user_zero, tfidf_user_queries, size)
        else:
            print('no tfidf')
            array_hotel = self.get_doc_arrays(hotel_path, hotel_zero)
            array_user = self.get_doc_arrays(user_path, user_zero)
        accuracy_array = np.zeros((np.shape(array_user)[0], np.shape(array_hotel)[0]))
        for i in range(np.shape(array_user)[0]):
            inferred_query = array_user[i]/np.linalg.norm(array_user[i])
            for j in range(np.shape(array_hotel)[0]):
                inferred_hotel = array_hotel[j]/np.linalg.norm(array_hotel[j])
                sim = np.dot(inferred_query, inferred_hotel)
                accuracy_array[i][j] = sim
        return accuracy_array

    def results_fasttext(self, tf_idf='yes', num_best=5):
        accuracy_array = self.get_accuracy_array_special(300, tf_idf)
        accuracy_array = self.make_accuracy_array(accuracy_array, num_best, bol=True)
        count = self.get_overall_accuracy(accuracy_array)
        'second experiment'
        length_array = self.divide_query_per_num_attribute()
        self.get_accuracy_based_attributes(accuracy_array, length_array)
        return count


