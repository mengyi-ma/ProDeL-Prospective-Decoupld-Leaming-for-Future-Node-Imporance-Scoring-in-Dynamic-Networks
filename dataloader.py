import pandas as pd
import torch
from torch_geometric.data import Data
import torch.serialization
from torch_sparse import SparseTensor
from collections import defaultdict
from typing import Dict
from tqdm import tqdm
import os
import numpy as np
import logging
import networkx as nx
import gc
import pickle

logger = logging.getLogger("dataloader")

def calculate_laplacian_with_self_loop(matrix):
    """
    calculatetextnormalizedLaplacian matrix
    parameter:
        matrix: torch.Tensor textadjacency matrix
    return:
        torch.Tensor normalizedLaplacian matrix
    """
    if not isinstance(matrix, SparseTensor):
        raise TypeError(f"Expected SparseTensor, got {type(matrix)}")

    device = matrix.device()

    row, col, value = matrix.coo()
    size = matrix.size(0)

    self_loop_idx = torch.arange(0, size, device=device)
    self_loop_idx = torch.stack([self_loop_idx, self_loop_idx], dim=0)

    edge_index = torch.cat([torch.stack([row, col], dim=0), self_loop_idx], dim=1)
    edge_weight = torch.ones(edge_index.shape[1], device=device)

    adj_with_loop = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(size, size)
    )

    row_sum = adj_with_loop.sum(dim=1).to_dense() # type: ignore

    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    diag_indices = torch.arange(0, size, device=device)
    diag_indices = torch.stack([diag_indices, diag_indices], dim=0)
    d_mat_inv_sqrt = SparseTensor(
        row=diag_indices[0],
        col=diag_indices[1],
        value=d_inv_sqrt,
        sparse_sizes=(size, size)
    )

    temp = d_mat_inv_sqrt.matmul(adj_with_loop) # type: ignore
    normalized_laplacian = temp.matmul(d_mat_inv_sqrt)

    return normalized_laplacian.to_torch_sparse_coo_tensor()


class FeatureCalculator:
    """calculatetextcentralitymetric（degree centrality、closeness centrality、betweenness centrality、Katz centrality）"""

    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        if self.use_gpu:
            torch.cuda.empty_cache()

    def _create_temporal_graph(self, df_window: pd.DataFrame, directed: bool = False) -> nx.Graph:
        """createtext"""
        graph = nx.DiGraph() if directed else nx.Graph()

        for _, row in df_window.iterrows():
            source, target, timestamp = row['source'], row['target'], row['time_stamps/seconds']

            graph.add_edge(source, target, timestamp=timestamp)

        return graph

    def compute_4_centrality(self, df_window: pd.DataFrame, directed: bool = False) -> tuple[Dict[str, np.ndarray], list]:
        """
        calculatetextcentralitymetric（degree centrality、closeness centrality、betweenness centrality、Katz centrality）
        return: text，containstextcentralitymetrictextnumpytext
        """
        temporal_graph = self._create_temporal_graph(df_window, directed)
        nodes = list(temporal_graph.nodes())
        num_nodes = len(nodes)

        if num_nodes == 0:
            raise RuntimeError("textmissingnode")

        centralities = {
            'degree': np.zeros(num_nodes, dtype=np.float32),
            'closeness': np.zeros(num_nodes, dtype=np.float32),
            'betweenness': np.zeros(num_nodes, dtype=np.float32),
            'katz': np.zeros(num_nodes, dtype=np.float32)
        }

        if directed:
            in_degree_dict = dict(temporal_graph.in_degree())
            out_degree_dict = dict(temporal_graph.out_degree())
            total_degree_dict = {node: in_degree_dict.get(node, 0) + out_degree_dict.get(node, 0)
                                 for node in nodes}
            centralities['degree'] = np.array([total_degree_dict.get(node, 0) for node in nodes],
                                              dtype=np.float32)
        else:
            degree_dict = dict(temporal_graph.degree())
            centralities['degree'] = np.array([degree_dict[node] for node in nodes], dtype=np.float32)

        try:
            closeness_dict = nx.closeness_centrality(temporal_graph)
            centralities['closeness'] = np.array([closeness_dict[node] for node in nodes], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"closenesscentralitycalculation failed: {e}")

        try:
            betweenness_dict = nx.betweenness_centrality(temporal_graph, normalized=False)
            centralities['betweenness'] = np.array([betweenness_dict[node] for node in nodes], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"betweennesscentralitycalculation failed: {e}")

        try:
            if temporal_graph.number_of_nodes() == 0:
                raise RuntimeError("Katz centralitycalculation failed：textmissingnode")

            katz_dict = nx.katz_centrality_numpy(
                temporal_graph,
                alpha=0.05,
                beta=1.0,
                normalized=False,
                weight=None
            )
            centralities['katz'] = np.array([katz_dict[node] for node in nodes], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Katz centralitycalculation failed: {e}")

        del temporal_graph
        gc.collect()

        return centralities, nodes

    def compute_window_node_features(self, df_window: pd.DataFrame, global_nodes_dict: dict,
                                     directed: bool = False) -> np.ndarray:
        """
        calculatewindowtextnode featurestext（textcentralitymetric）
        return: text [num_window_nodes, 4]，textcontainscurrentwindowtextnode
        """
        centralities, window_nodes = self.compute_4_centrality(df_window, directed)

        num_window_nodes = len(window_nodes)
        node_features = np.zeros((num_window_nodes, 4), dtype=np.float32)

        for i, node in enumerate(window_nodes):
            node_idx = window_nodes.index(node)
            node_features[i] = [
                centralities['degree'][node_idx],
                centralities['closeness'][node_idx],
                centralities['betweenness'][node_idx],
                centralities['katz'][node_idx]
            ]

        return node_features


class TemporalGraphProcessor:
    def __init__(self, file_path, directed=False, gnn_stride=3, pred_stride=1, split_info_path=None):
        self.file_path = file_path
        self.directed = directed
        self.gnn_stride = int(gnn_stride)
        self.pred_stride = int(pred_stride)
        self.graphs = {}
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.edge_stats = defaultdict(int)
        self.split_info_path = split_info_path

        self.centrality_calculator = FeatureCalculator()

        self.split_to_windows = {}  # split_index -> [windows]

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("TemporalGraphProcessor")

    def load_and_process(self):
        """loadtextdata"""
        self.logger.info(f"textfile: {self.file_path}")

        df = pd.read_csv(self.file_path)
        df['source'] = df['source'].astype(str)
        df['target'] = df['target'].astype(str)
        required_cols = ['source', 'target', 'time_window']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"CSVfiletext: {missing}")

        if self.split_info_path and os.path.exists(self.split_info_path):
            self._load_split_info()
        else:
            self.logger.warning(f"not foundsplittextfile: {self.split_info_path}")

        all_nodes = np.unique(df[['source', 'target']].values.astype(str))
        self.node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
        self.reverse_mapping = {v: k for k, v in self.node_mapping.items()}
        self.logger.info(f"createnodetext: {len(self.node_mapping)} textnode")

        windows = sorted(df['time_window'].unique())
        self.logger.info(f"text {len(windows)} texttime window")

        df['source_id'] = df['source'].map(self.node_mapping)
        df['target_id'] = df['target'].map(self.node_mapping)
        if df[['source_id', 'target_id']].isnull().any().any():
            unmap_sources = df[df['source_id'].isnull()]['source'].unique()
            unmap_targets = df[df['target_id'].isnull()]['target'].unique()
            raise ValueError(f"textnodeID: textnode {unmap_sources}, textnode {unmap_targets}")

        for window, group in tqdm(df.groupby('time_window'),
                                  desc="texttime window",
                                  total=len(windows),
                                  dynamic_ncols=True):
            self._process_window(window, group)

        del df
        gc.collect()
        self._log_statistics()

    def _load_split_info(self):
        """loadsplittext"""
        try:
            with open(self.split_info_path, 'rb') as f:
                split_data = pickle.load(f)

            self.split_to_windows = split_data.get('split_to_windows', {})
            self.logger.info(f"loadsplittext: {len(self.split_to_windows)} textsplit")

            for split_idx, windows in self.split_to_windows.items():
                self.logger.debug(f"split {split_idx}: containswindow {windows}")

        except Exception as e:
            raise RuntimeError(f"loadsplittext: {e}")

    def get_split_windows(self, split_index):
        """textsplittextallwindow"""
        windows = self.split_to_windows[split_index]
        return windows

    def get_all_splits(self):
        """textreturn split_to_windows text（split_index）text"""
        return list(self.split_to_windows.keys())

    def _log_statistics(self):
        """text"""
        total_edges = sum(self.edge_stats.values())
        total_nodes = len(self.node_mapping)

        self.logger.info(f"text: {total_nodes} node, {total_edges} text")
        self.logger.info(f"time windowtext: {len(self.graphs)}")
        self.logger.info(f"windowsplit: {len(self.split_to_windows)} textsplit")

    def save_processed_data(self, output_dir):
        """savetextdatatextPyTorch Geometrictext"""
        os.makedirs(output_dir, exist_ok=True)

        graph_path = os.path.join(output_dir, "graph_data.pt")

        save_data = {
            'graphs': self.graphs,
            'node_mapping': self.node_mapping,
            'reverse_mapping': self.reverse_mapping,
            'split_to_windows': self.split_to_windows
        }

        torch.save(save_data, graph_path)
        self.logger.info(f"textdatatextsavetext: {graph_path}")

    @classmethod
    def load_processed_data(cls, input_dir, directed=False, gnn_stride=3, pred_stride=1):
        """loadtextdata"""
        graph_path = os.path.join(input_dir, "graph_data.pt")

        torch.serialization.add_safe_globals([Data])

        data = torch.load(graph_path, weights_only=False)

        processor = cls(
            file_path="",
            directed=directed,
            gnn_stride=gnn_stride,
            pred_stride=pred_stride
        )
        processor.graphs = data['graphs']
        processor.node_mapping = data['node_mapping']
        processor.reverse_mapping = data['reverse_mapping']
        processor.split_to_windows = data.get('split_to_windows', {})

        logging.basicConfig(level=logging.INFO)
        processor.logger = logging.getLogger("TemporalGraphProcessor")
        processor.logger.info(f"text {input_dir} loadtextdata")
        processor.logger.info(f"time windowtotal number: {len(processor.graphs)}")
        processor.logger.info(f"windowsplittext: {len(processor.split_to_windows)}")

        return processor

    def _process_window(self, window, window_df):
        """texttime window - textalltextcreatetextdata，calculateadjacency matrixtextLaplacian matrix"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        edge_nodes_global = np.unique(window_df[['source_id', 'target_id']].values.flatten())
        edge_nodes_global = edge_nodes_global[edge_nodes_global >= 0]

        if edge_nodes_global.size == 0:
            raise RuntimeError(f"window {window} textnode，text")

        self.logger.info(f"window {window}: edge_nodes_global type: {type(edge_nodes_global)}, text: {edge_nodes_global.shape}")

        edge_nodes_global_list = [int(node_id) for node_id in edge_nodes_global]

        window_local_to_global = {}
        window_global_to_local = {}

        for li, global_id in enumerate(edge_nodes_global_list):
            window_local_to_global[li] = global_id
            window_global_to_local[global_id] = li

        num_window_nodes = len(window_local_to_global)

        sources_local, targets_local = [], []
        for _, row in window_df.iterrows():
            s_g = int(row['source_id'])
            t_g = int(row['target_id'])
            if s_g in window_global_to_local and t_g in window_global_to_local:
                sources_local.append(window_global_to_local[s_g])
                targets_local.append(window_global_to_local[t_g])

        if not sources_local:
            raise RuntimeError(f"window {window} text，text")

        sources_tensor = torch.tensor(sources_local, dtype=torch.long, device=device)
        targets_tensor = torch.tensor(targets_local, dtype=torch.long, device=device)

        edge_index = torch.stack([
            torch.cat([sources_tensor, targets_tensor]),
            torch.cat([targets_tensor, sources_tensor])
        ]).to(device)

        try:
            node_features_np = self.centrality_calculator.compute_window_node_features(
                window_df[['source', 'target', 'time_stamps/seconds']],
                self.node_mapping,
                self.directed
            )

            if node_features_np.shape[0] != num_window_nodes:
                raise RuntimeError(f"window {window}: textnumber of nodes({node_features_np.shape[0]})textactualnumber of nodes({num_window_nodes})text")
            node_features = torch.from_numpy(node_features_np).float().to(device)
        except Exception as e:
            raise RuntimeError(f"window {window} textcalculation failed: {e}")

        try:
            edge_set = set()
            row_indices = []
            col_indices = []

            for s, t in zip(sources_local, targets_local):
                if not self.directed:
                    node_pair = (min(s, t), max(s, t))
                    if node_pair not in edge_set:
                        edge_set.add(node_pair)
                        row_indices.extend([node_pair[0], node_pair[1]])
                        col_indices.extend([node_pair[1], node_pair[0]])
                else:
                    if (s, t) not in edge_set:
                        edge_set.add((s, t))
                        row_indices.append(s)
                        col_indices.append(t)

            row_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
            col_tensor = torch.tensor(col_indices, dtype=torch.long, device=device)
            edge_weight = torch.ones(len(row_indices), device=device)

            adj_matrix = SparseTensor(
                row=row_tensor,
                col=col_tensor,
                value=edge_weight,
                sparse_sizes=(num_window_nodes, num_window_nodes)
            )

            laplacian_matrix = calculate_laplacian_with_self_loop(adj_matrix)


        except Exception as e:
            raise RuntimeError(f"window {window} adjacency matrixcalculation failed: {e}")

        window_info = {
            'global_to_local': window_global_to_local,
            'local_to_global': window_local_to_global,
            'global_nodes': [str(g) for g in edge_nodes_global],
            'num_nodes': num_window_nodes
        }

        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=num_window_nodes,
            window_id=torch.tensor([window], dtype=torch.long, device=device),
            is_directed=torch.tensor([self.directed], dtype=torch.bool, device=device),
            adj_matrix=adj_matrix,
            laplacian_matrix=laplacian_matrix
        )

        graph_data.window_info = window_info

        self.graphs[window] = graph_data
        self.edge_stats[window] = len(sources_local)

        del node_features_np, sources_local, targets_local, adj_matrix
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_standardized_window_mappings(self, window_id):
        """textwindowtext"""
        if window_id not in self.graphs:
            raise KeyError(f"window {window_id} text")

        window_info = self.graphs[window_id].window_info
        window_global_to_local = window_info['global_to_local']
        window_local_to_global = window_info['local_to_global']

        standardized_global_to_local = {}
        standardized_local_to_global = {}
        current_window_global_nodes = []

        for global_id, local_idx in window_global_to_local.items():
            str_global_id = str(global_id)
            int_local_idx = int(local_idx)

            if int_local_idx < 0:
                raise ValueError(f"window {window_id} text {int_local_idx} text")

            standardized_global_to_local[str_global_id] = int_local_idx
            current_window_global_nodes.append(str_global_id)

        for local_idx, global_id in window_local_to_global.items():
            int_local_idx = int(local_idx)
            if int_local_idx < 0:
                raise ValueError(f"window {window_id} text {int_local_idx} text")

            str_global_id = str(global_id)
            standardized_local_to_global[int_local_idx] = str_global_id

        current_window_global_nodes = sorted(list(set(current_window_global_nodes)))

        return {
            'global_to_local': standardized_global_to_local,
            'local_to_global': standardized_local_to_global,
            'global_nodes': current_window_global_nodes,
            'num_nodes': self.graphs[window_id].num_nodes
        }


if __name__ == "__main__":
    input_csv_path = "./data/raw/Email.csv"
    output_dir = "./data/processed_data"
    split_info_path = "./data/processed_data/Email_window_attributes.pkl"

    processor = TemporalGraphProcessor(
        file_path=input_csv_path,
        directed=False,
        gnn_stride=3,
        pred_stride=1,
        split_info_path=split_info_path
    )

    processor.load_and_process()

    processor.save_processed_data(output_dir)
