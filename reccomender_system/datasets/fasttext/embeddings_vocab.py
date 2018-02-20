import pickle
import numpy as np
folder = './embeddings/'
size_embedding = 300

with open(folder+'vocab'+str(size_embedding)+'_fast.pkl', 'rb') as f:
    vocab = pickle.load(f)


vocab_new = dict()
embeddings = np.load(folder+'embeddings_'+str(size_embedding)+'_fast.npy')
embeddings_new = np.zeros((982,size_embedding))
idx = 0
lost = 0
f = open('vocab_cut.txt', 'r')
loxt_set = set()
for line in f:
    for t in line.strip().split():
        if t in vocab:
            vocab_new[t]=idx
            embedd= vocab.get(t)

            embeddings_new[idx]=embeddings[embedd]

            idx += 1
        else:
            lost +=1
            loxt_set.add(t)
print(loxt_set)
print(lost)
print(embeddings_new)
with open(folder+'vocab_'+str(size_embedding) +'d.pkl', 'wb') as f:
    pickle.dump(vocab_new, f, pickle.HIGHEST_PROTOCOL)
np.save(folder+'embeddings_'+str(size_embedding)+'d.npy', embeddings_new)


