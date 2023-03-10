import numpy as np
import polars as pl
import pandas as pd

import torch

import gzip
import shutil
import os

from constants import *

from torch_geometric.data import Data, InMemoryDataset, download_url

def make_column_map(columns: list[pl.Series]) -> dict:
    objects = pl.concat(columns, how="vertical")
    objects = objects.unique().sort()
    object_map = dict(zip(objects.to_list(), range(len(objects))))
    return object_map

def convert_df_to_int_indexes(df: pl.DataFrame) -> pl.DataFrame:
    int_columns = ["entity1", "entity2", "relation"]
    for col in df.columns:
        if col in int_columns:
            # Remove first element and convert col to int
            df = df.with_columns(pl.col(col).str.lstrip("QP").cast(pl.UInt64))
    return df

def transpose_one_dataset(dataset: pl.DataFrame, entity_map: dict, relation_map: dict) -> pl.DataFrame:
    return dataset.with_columns([
        (pl.col("entity1")).apply(lambda x: entity_map[x]).alias("entity1"),
        (pl.col("relation")).apply(lambda x: relation_map[x]).alias("relation"),
        (pl.col("entity2")).apply(lambda x: entity_map[x]).alias("entity2"),
    ])

def get_entity_map(datasets: list[pl.DataFrame], corpus_text) -> dict:
    train_data, validate_data, test_data = datasets
    entity_columns = [
            train_data["entity1"],
            train_data["entity2"],
            validate_data["entity1"],
            validate_data["entity2"],
            test_data["entity1"],
            test_data["entity2"],
            corpus_text["entity1"],
        ]
    return make_column_map(entity_columns)

def get_relation_map(datasets: list[pl.DataFrame]) -> dict: 
    relation_columns = [data["relation"] for data in datasets]
    return make_column_map(relation_columns)


class Wikidata5m(InMemoryDataset):
    # dataset = Wikidata5m("datasets/")
    # dataset[0]    
    def __init__(self, root, name="Wikidata5m", use_embeddings=False, embeddings_file="sentence_features.npy", transform=None, pre_transform=None, pre_filter=None):
        # Root is a path to a folder which contains the datasets
        self.entity_map = {}
        self.relation_map = {}
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if use_embeddings:
            if os.path.isfile(embeddings_file):
                self.data.x = self.load_embeddings(self.data.x.shape[0])
            else:
                print("Precompute embeddings")

    @property
    def raw_file_names(self):
        return [dataset.value for dataset in DatasetTarNames]

    @property
    def processed_file_names(self):
        return ["saved_dataset.pt", "processed_corpus.csv"]

            
    def download(self):
        raw_dir = self.raw_dir + "/"
        def save_gz_file(gzip_file_name: str, out_file_name: str):
            with gzip.open(gzip_file_name, 'rb') as f_in:
                with open(out_file_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        def remove_tar_files():
            for file in DatasetTarNames:
                os.remove(raw_dir + file.value)
            os.removedirs(raw_dir)
           
        def repair_utf8_corpus_file():
            corpus_columns = ["entity1", "description"]
            corpus_text = pl.read_csv(f"{self.root}/{DatasetNames.corpus.value}", sep="\t", has_header=False, new_columns=corpus_columns, encoding="utf8-lossy")
            corpus_text.write_csv(f"{self.root}/{DatasetNames.corpus.value}")


        print("Started download")
        # Download to `raw_dir_dir`.
        try:
            for url in DatasetWebPaths:
                download_url(url.value, raw_dir)
        except:
            print("Unable to download files")

        print("Unpacking")
        for tar_file in DatasetTarNames:
            tar_name = tar_file.value
            if tar_file == DatasetTarNames.corpus:
                save_gz_file(raw_dir + tar_name, f"{self.root}/{DatasetNames.corpus.value}")
            else:
                shutil.unpack_archive(raw_dir + tar_name, self.root)
        
        print("Finished download")
        repair_utf8_corpus_file()
        # remove_tar_files()

    def read_csv_files(self) -> list[pl.DataFrame]:
        entity_columns = ["entity1", "relation", "entity2"] 
        corpus_columns = ["entity1", "description"]
        def read_entity_file(file_name: str, entity_columns=entity_columns) -> pl.DataFrame:
            return pl.read_csv(f"{self.root}/{file_name}", sep="\t", has_header=False, new_columns=entity_columns)

        train_data = read_entity_file(DatasetNames.train.value)
        validate_data = read_entity_file(DatasetNames.validate.value)
        test_data = read_entity_file(DatasetNames.test.value)
        corpus_text = pl.read_csv(f"{self.root}/{DatasetNames.corpus.value}", new_columns=corpus_columns, encoding="utf8-lossy")

        return [train_data, validate_data, test_data], corpus_text


    @staticmethod
    def get_edges_from_dataset(dataset: pl.DataFrame) -> np.array:
        # PyG requires format [2, num_edges]
        edges = dataset.select(["entity1", "entity2"]).to_numpy().T
        assert(edges.shape[0] == 2)
        return edges


    def process(self):
        datasets, corpus_text = self.read_csv_files()

        datasets, self.corpus_text = self.transpose_entity_ids_to_range(datasets, corpus_text)
        train_data, validate_data, test_data = datasets

        # PyG needs data as torch tensors of longs
        train_edges = torch.tensor(self.get_edges_from_dataset(train_data), dtype=torch.long)
        # Test edges need to be in format [num_edges, *]
        validate_edges = torch.tensor(self.get_edges_from_dataset(validate_data), dtype=torch.long)
        test_edges = torch.tensor(self.get_edges_from_dataset(test_data), dtype=torch.long)
        
        # X HAS to be in float format! Pytorch gives wrong type warning        
        node_ids = torch.arange(len(self.entity_map.values()), dtype=torch.float).reshape((-1, 1))
        data_list = [Data(x=node_ids, edge_index=train_edges)] #, y=test_edges, validation_edges=validate_edges)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        self.corpus_text.write_csv(self.processed_paths[1])
        torch.save((data, slices), self.processed_paths[0])

    def transpose_entity_ids_to_range(self, datasets: list[pl.DataFrame], corpus_text: pl.DataFrame):
        # Input: train_data, validate_data, test_data, corpus_text
        datasets = [convert_df_to_int_indexes(dataset) for dataset in datasets]
        corpus_text = convert_df_to_int_indexes(corpus_text)

        self.entity_map = get_entity_map(datasets, corpus_text)
        self.relation_map = get_relation_map(datasets)

        datasets = [transpose_one_dataset(dataset, self.entity_map, self.relation_map) for dataset in datasets]
        corpus_text = corpus_text.with_columns([(pl.col("entity1")).apply(lambda x: self.entity_map[x]).alias("entity1")])
        corpus_text = corpus_text.sort("entity1")
        return datasets, corpus_text
    
    def load_embeddings(self, data_n_rows: int) -> np.array:
        print("Loading embeddings")
        features = np.load(embeddings_file)
        X = np.ones((data_n_rows, features.shape[1]))
        X[:features.shape[0], :] = features

        torch_features = torch.from_numpy(X)
        torch_features = torch_features.to(torch.float)
        return torch_features