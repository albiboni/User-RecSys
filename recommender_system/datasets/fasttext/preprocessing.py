import pickle
import numpy as np


vocab = dict()

size_embedding = 300
embeddings = np.zeros((2519370,size_embedding))
with open('wiki.en.vec', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx != 0:


            index = line.index(' ')
            vocab[line[:index].strip()] = idx-1


            embeddings[idx-1] = np.fromstring(line[index:], sep= ' ')

            if idx%200000 == 0:
                print(idx)
        else:
            print('skip')

f.close()
folder='./embeddings/'
np.save(folder+'embeddings_'+str(size_embedding)+'_fast.npy', embeddings)
print(embeddings)

with open(folder+'vocab'+str(size_embedding)+'_fast.pkl', 'wb') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
print('vocab done')
