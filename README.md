# [SIGMOD 2025] SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search

## Notice
* SymphonyQG has been implemented in our [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library). This repo will be archived soon.

## Prerequisites
* AVX512 is required
* For details, please refer to our [technical report](https://arxiv.org/abs/2411.12229).

## Directory Structure

    ../
    ├── data/               # datasets and indices
    ├── symqglib/          
    |   ├── index/    
    |   |   ├── fastscan/   # helper function for FastScan
    |   |   └── qg/         # quantized graph
    |   ├── third/          # third party dependency
    |   └── utils/          # common utils
    ├── python/             # python bindings
    └── reproduce/          # code for reproduction


## Python Bindings (recommended)

### Bindings installation

* Install from sources in Python env (recommended version: 3.10):
```bash
apt-get install -y python-setuptools python-pip
cd python/
pip install -r requirements.txt
sh build.sh
```

### API description

* `symphonyqg.Index(index_type, metric, num_elements, dimension, degree_bound=32)` - intialize a non-constructed index
  * `index_type` defines the index type, currently only support 'QG'
  * `metric` defines the metric space, currently only support 'L2'
  * `num_elements` defines the number of elements
  * `dimension` defines the dimension of data vector
  * `degree_bound` defines the maximum out-degree of graph, must be a multiple of 32

`symphonyqg.Index` methods:
* `build_index(data, EF, num_iter=3, num_threads=ALL_THREDS)` - construct the index from `data`
    * `data` numpy array of vectors, `dtype=float32`, shape: `(num_elements, dimension)`
    * `EF` a parameter that controls the number of candidates during graph construction
    * `num_iter` number of interation for indexing, 3 by default
    * `num_threads` number of threads for indexing, use all threads in system by default
* `save(filename)` - save the `Index` to given path
* `load(filename)` - load the `Index` from given path, the loaded index must have same initialization parameters as the object
* `set_ef(EF)` - set the beam size to control time-accuracy trade-off of querying
* `search(query, k)` - search approximate `k` nearest neighbors for a given `query` 
    * `query` numpy array of a query vector, `dtype=float32`, shape: `(dimension,)` or `(1, dimension)`

### Example
For examples on real-world datasets, please refer to `./reproduce`
```python
import symphonyqg
import numpy as np

D = 64
N = 100000

# Random data
data = np.random.random((N, D)).astype('float32')

# Init index
index = symphonyqg.Index("QG", "L2", num_elements=N, dimension=D, degree_bound=32)

# Construct index
index.build_index(data, 200)

# Set beam size for querying
index.set_ef(100)

# Search query
K = 10
for i in range(10):
    query = data[i]
    knn = index.search(query, K)
    print(knn)

# Save index
index.save("./test.index")
del index

# Load index
index = symphonyqg.Index("QG", "L2", num_elements=N, dimension=D, degree_bound=32)
index.load("./test.index")
```

## Global Fixed-Center Quantization

This version stores RaBitQ codes with one global center `c_0`, chosen as the
centroid of all base vectors during index construction. Each data vector is
quantized once as:

```text
o = (o_r - c_0) / ||o_r - c_0||
```

Queries are also normalized once per search:

```text
q = (q_r - c_0) / ||q_r - c_0||
```

As a result, each graph row stores only:

```text
raw vector + one compact RaBitQ code + one factor triplet + neighbor ids
```

It no longer stores one copy of the neighbors' quantization codes under every
visited node. Old `.index` files built by the original per-node-center layout
are not compatible with this layout and must be rebuilt.

Let:

* `N` be the number of data vectors
* `d` be the original dimension
* `B = 1 << ceil_log2(d)` be the padded dimension
* `R = degree_bound`

The persistent index size is:

```text
4 * N * (d + ceil(B / 32) + 3 + R) + 4 * d + 4 * B + 4 bytes
```

The terms are raw vectors, one compact code per vector, three per-vector
RaBitQ factors, neighbor ids, the global center, the rotator, and the entry
point. Compared with the original per-node-center storage:

```text
4 * N * (d + R * (B / 32 + 4)) + 4 * B + 4 bytes
```

the new layout saves approximately:

```text
4 * N * (R - 1) * (B / 32 + 3) bytes
```


## Reproduce
* For downloading datasets and preprocessing, please refer to `./data/README.md`
* To build the new fixed-`c_0` indices and test query performance, please refer to `./reproduce/README.md` for details

## C++ examples
```bash
mkdir bin/ build/
cd build
cmake ..
make
```
* Currently, we only add an example for indexing. The APIs will be updated later.
