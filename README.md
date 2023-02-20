# Wikidata5m Graph Machine Learning

## ABSTRACT:

TODO:

## Using Graph Machine Learning for Link Prediction on the Wikidata5m dataset

Wikidata5m is a knowledge graph dataset, which integrates the Wikidata knowledge graph and Wikipedia texts.

The dataset can be found at [https://deepgraphlearning.github.io/project/wikidata5m](https://deepgraphlearning.github.io/project/wikidata5m).

I am using the transductive version, which has 4,5M+ entities, 822 relations and 20M+ edges.

## Dataset preprocessing

To work comfortably with the data, I wrote a PyG dataset, concretely `InMemoryDataset`, which downloads all the needed files and preprocesses them to finally get a PyG graph.

The dataset can then be used as follows:

```python
from torch_wikidata import Wikidata5m

dataset = Wikidata5m("datasets/")
```

## Working with large graphs

As the graph is very large, one of the main challenges is even getting the graph to memory.

I am using PyG's [LinkNeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.LinkNeighborLoader), which uses GraphSAGE under the hood to sample the edges.

We can then use traditional batching to train and test the model.

```python
for batch in data_loader:
    ...
```

Using this approach and following parameters on a T4 GPU (Rosie):
1. Without embeddings:
    - `batch_size=1024`
    - `n_neighbors=[10] * 2`
    
  - One full batch takes around 5 minutes to run.

1. With embeddings (384 features):
    - `batch_size=512`
    - `n_neighbors=[10] * 2`
    
  - One full batch takes around 8 minutes to run.

## Link prediction

Link prediction is traditionally done using pairs of nodes, we predict if there exists an edge between them.
The original nodes are infused with negative samples of edges and the model's task is to predict the true edges, while disregarding the negative edges.

My error metric is ROC AUC, metric for training is Binary Cross Entropy

## Results
--- 
## Results using graph structure

Using a 2-layer GCN, I was able to get ROC AUC Score of `0.61`

## Results using graph structure with embeddings from Wikipedia articles

I have used SentenceTransformers to get the embeddings, because of the size of the data, I was forced to use the fastest available model for inference.

```python
SentenceTransformer('paraphrase-MiniLM-L3-v2')
```

Getting the embeddings using batches took around 50 minutes (T4 GPU, Rosie)

The results with embeddings were improved up to the ROC AUC score of `0.65`.