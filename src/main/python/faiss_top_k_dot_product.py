import numpy as np
import faiss
from faiss.contrib.exhaustive_search import knn

dimensions = 3072

train = np.memmap('train.fvecs', dtype='float32').reshape(-1, dimensions)
test = np.fromfile('test.fvecs', dtype='float32').reshape(-1, dimensions)

k = 100
D, I = knn(test[0:10000], train, k, faiss.METRIC_INNER_PRODUCT)

I = np.array(I, dtype=np.int32)
I.tofile('neighbors.ivecs')
