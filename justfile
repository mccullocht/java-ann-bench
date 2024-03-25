heap_size := "16g"

init:
  git submodule init
  git submodule update
  @just update-jvector
  @just update-lucene

update-jvector:
  git -C submodules/jvector pull origin ann-sandbox
  cd submodules/jvector && ./mvnw clean package
  mkdir -p libs
  cp submodules/jvector/jvector-multirelease/target/jvector-1.0.3-SNAPSHOT.jar libs/

update-lucene:
  git -C submodules/lucene fetch origin
  git -C submodules/lucene checkout binary-quantization
  git -C submodules/lucene pull
  cd submodules/lucene && ./gradlew jar
  mkdir -p libs
  cp submodules/lucene/lucene/core/build/libs/lucene-core-10.0.0-SNAPSHOT.jar libs/

build-docker:
  docker build -f Dockerfile-base -t java-ann-bench-base .
  docker build --no-cache -t java-ann-bench .

build config:
  @./gradlew run --console=plain --quiet -PminHeapSize="-Xmx{{heap_size}}" -PmaxHeapSize=-"Xms{{heap_size}}" --args="--build --config={{config}}"

query config:
  #!/usr/bin/env bash
  set -exuo pipefail

  oversample=$(yq e '.query.oversample' {{config}})
  float_hnsw=$(yq e '.query.floatHnsw' {{config}})

  export BQ_SEGMENT_RESCORE_OVERSAMPLE=${oversample}
  export BQ_FLOAT_HNSW=${float_hnsw}

  #pq_rerank=$(yq e '.query.pqRerank' {{config}})
  #mlock_graph=$(yq e '.query.mlockGraph' {{config}})
  #mmap_pq_vectors=$(yq e '.query.mmapPqVectors' {{config}})
  #mlock_pq_vectors=$(yq e '.query.mlockPqVectors' {{config}})
  #parallel_pq_vectors=$(yq e '.query.parallelPqVectors' {{config}})
  #parallel_neighborhoods=$(yq e '.query.parallelNeighborhoods' {{config}})
  #parallel_neighborhoods_beam_width=$(yq e '.query.parallelNeighborhoodsBeamWidth' {{config}})
  #parallel_rerank_threads=$(yq e '.query.parallelRerankThreads' {{config}})
  #node_cache_degree=$(yq e '.query.nodeCacheDegree' {{config}})
  #candidates=$(yq e '.query.numCandidates' {{config}})

  #export VAMANA_PQ_RERANK=${pq_rerank}
  #export VAMANA_MLOCK_GRAPH=${mlock_graph}
  #export VAMANA_MMAP_PQ_VECTORS=${mmap_pq_vectors}
  #export VAMANA_MLOCK_PQ_VECTORS=${mlock_pq_vectors}
  #export VAMANA_PARALLEL_PQ_VECTORS=${parallel_pq_vectors}
  #export VAMANA_PARALLEL_NEIGHBORHOODS=${parallel_neighborhoods}
  #export VAMANA_PARALLEL_NEIGHBORHOODS_BEAM_WIDTH=${parallel_neighborhoods_beam_width}
  #export VAMANA_PARALLEL_RERANK_THREADS=${parallel_rerank_threads}
  #export VAMANA_CACHE_DEGREE=${node_cache_degree}
  #export VAMANA_CANDIDATES=${candidates}

  ./gradlew run --console=plain --quiet -PminHeapSize="-Xmx{{heap_size}}" -PmaxHeapSize=-"Xms{{heap_size}}" --args="--query --config={{config}}"

query-docker config:
  #!/usr/bin/env bash
  set -exuo pipefail

  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  system_memory=$(yq e '.runtime.systemMemory' {{config}})
  docker run --rm \
    --privileged \
    --name query-bench-run \
    -p 20000:20000 \
    -v /usr/src:/usr/src:ro \
    -v /lib/modules:/lib/modules:ro \
    -v "$(pwd)/configs":/java-ann-bench/configs \
    -v "$(pwd)/datasets":/java-ann-bench/datasets \
    -v "$(pwd)/indexes":/java-ann-bench/indexes \
    -v "$(pwd)/reports":/java-ann-bench/reports \
    -v "$(pwd)/src/main/c/libwrappeduring.so":/java-ann-bench/src/main/c/libwrappeduring.so \
    -m ${system_memory} \
    java-ann-bench \
    bash -c "git pull && just update-lucene && just query-docker-internal {{config}}"

query-docker-internal config:
  #!/usr/bin/env bash
  set -exuo pipefail

  pq_rerank=$(yq e '.query.pqRerank' {{config}})
  mlock_graph=$(yq e '.query.mlockGraph' {{config}})
  mmap_pq_vectors=$(yq e '.query.mmapPqVectors' {{config}})
  mlock_pq_vectors=$(yq e '.query.mlockPqVectors' {{config}})
  parallel_pq_vectors=$(yq e '.query.parallelPqVectors' {{config}})
  parallel_neighborhoods=$(yq e '.query.parallelNeighborhoods' {{config}})
  parallel_neighborhoods_beam_width=$(yq e '.query.parallelNeighborhoodsBeamWidth' {{config}})
  parallel_rerank_threads=$(yq e '.query.parallelRerankThreads' {{config}})
  node_cache_degree=$(yq e '.query.nodeCacheDegree' {{config}})
  candidates=$(yq e '.query.numCandidates' {{config}})
  heap_size=$(yq e '.runtime.heapSize' {{config}})

  export VAMANA_PQ_RERANK=${pq_rerank}
  export VAMANA_MLOCK_GRAPH=${mlock_graph}
  export VAMANA_MMAP_PQ_VECTORS=${mmap_pq_vectors}
  export VAMANA_MLOCK_PQ_VECTORS=${mlock_pq_vectors}
  export VAMANA_PARALLEL_PQ_VECTORS=${parallel_pq_vectors}
  export VAMANA_PARALLEL_NEIGHBORHOODS=${parallel_neighborhoods}
  export VAMANA_PARALLEL_NEIGHBORHOODS_BEAM_WIDTH=${parallel_neighborhoods_beam_width}
  export VAMANA_PARALLEL_RERANK_THREADS=${parallel_rerank_threads}
  export VAMANA_CACHE_DEGREE=${node_cache_degree}
  export VAMANA_CANDIDATES=${candidates}

  ./gradlew run --console=plain --quiet -PminHeapSize="-Xmx${heap_size}" -PmaxHeapSize=-"Xms64m" --args="--query --config={{config}}"


iouring:
  @./gradlew runIoUring
