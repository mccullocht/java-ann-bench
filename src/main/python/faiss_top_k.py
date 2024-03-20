import numpy as np
import faiss

dimensions = 768

train = np.memmap('train.fvecs', dtype='float32').reshape(-1, dimensions)
test = np.fromfile('test.fvecs', dtype='float32').reshape(-1, dimensions)

k = 100
D, I = faiss.contrib.exhaustive_search.knn(test[0:10000], train, k)

I = np.array(I, dtype=np.int32)
I.tofile('neighbors.ivecs')
