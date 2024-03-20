import sys

import numpy as np
import random
import multiprocessing
import os

from collections import namedtuple
from datasets import load_dataset

Dataset = namedtuple('Dataset', ['name', 'size', 'dimensions'])
wiki_en_embeddings = Dataset('Cohere/wikipedia-22-12-en-embeddings', 35_167_920, 768)
simple = Dataset('Cohere/wikipedia-22-12-simple-embeddings', 485_859, 768)

dataset_info = wiki_en_embeddings
#dataset_info = simple
random.seed(0)

test_indexes = set()
while len(test_indexes) < 10000:
  test_indexes.add(random.randint(0, dataset_info.size - 1))

def process_chunk(start_idx, end_idx, chunk_id, chunk_offset, dataset_info, test_indexes):
  dataset = load_dataset(dataset_info.name, split=f'train[{start_idx}%:{end_idx}%]')
  with open(f'train_{chunk_id}.fvecs', 'wb') as train, open(f'test_{chunk_id}.fvecs', 'wb') as test:
    for i, doc in enumerate(dataset):
      idx = chunk_offset + i
      test_embedding = idx in test_indexes
      emb = doc['emb']
      emb_array = np.array(emb, dtype='<f4')
      file = test if test_embedding else train
      file.write(emb_array.tobytes())


def merge_files(file_type):
  with open(f'{file_type}.fvecs', 'wb') as outfile:
    for i in range(num_processes):
      with open(f'{file_type}_{i}.fvecs', 'rb') as infile:
        outfile.write(infile.read())
      os.remove(f'{file_type}_{i}.fvecs')


if __name__ == '__main__':
  num_processes = 50
  step = 100 / num_processes

  chunk_docs = [
    load_dataset(dataset_info.name, split=f'train[{int(i * step)}%:{int((i + 1) * step)}%]').num_rows
    for i in range(num_processes)
  ]
  chunk_offsets = [
    sum(docs for docs in chunk_docs[0:i])
    for i in range(num_processes)
  ]

  processes = []
  for i in range(num_processes):
    p = multiprocessing.Process(
        target=process_chunk,
        args=(int(i * step), int((i + 1) * step), i, chunk_offsets[i], dataset_info, test_indexes)
    )
    p.start()
    processes.append(p)

  for p in processes:
    p.join()

  merge_files('train')
  merge_files('test')