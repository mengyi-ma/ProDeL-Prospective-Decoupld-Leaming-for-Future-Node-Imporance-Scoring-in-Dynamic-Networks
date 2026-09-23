import sys
import os
import networkx as nx
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import logging
import torch
import gc
import pickle
import glob
import multiprocessing as mp
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.dataloader import TemporalGraphProcessor

logger = logging.getLogger("textdegreecalculatetext")

class CentralityCalculator:
    """calculatetextcentralitymetric（degree centrality、closeness centrality、betweenness centrality、Katz centrality）textnormalized"""

    def __init__(self):
        pass

    def compute_normalized_centralities(self, G: nx.Graph, directed: bool = False) -> Dict[str, np.ndarray]:
        """
        calculatetextnormalizedtextcentralitymetric
        return: text，containstextnormalizedcentralitymetrictextnumpytext
        """
        if G.number_of_nodes() == 0:
            raise RuntimeError("textmissingnode")

        nodes = list(G.nodes())
        num_nodes = len(nodes)

        centralities = {
            'degree': np.zeros(num_nodes, dtype=np.float32),
            'closeness': np.zeros(num_nodes, dtype=np.float32),
            'betweenness': np.zeros(num_nodes, dtype=np.float32),
            'katz': np.zeros(num_nodes, dtype=np.float32)
        }

        if directed:
            in_degree_dict = nx.in_degree_centrality(G)
            out_degree_dict = nx.out_degree_centrality(G)
            total_degree_dict = {node: in_degree_dict.get(node, 0) + out_degree_dict.get(node, 0)
                                 for node in nodes}
            centralities['degree'] = np.array([total_degree_dict.get(node, 0) for node in nodes], dtype=np.float32)

        else:
            degree_dict = nx.degree_centrality(G)
            centralities['degree'] = np.array([degree_dict[node] for node in nodes], dtype=np.float32)

        try:
            closeness_dict = nx.closeness_centrality(G)
            centralities['closeness'] = np.array([closeness_dict[node] for node in nodes], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"closeness centralitycalculation failed: {e}")

        try:
            betweenness_dict = nx.betweenness_centrality(G, normalized=True)
            centralities['betweenness'] = np.array([betweenness_dict[node] for node in nodes], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"betweenness centralitycalculation failed: {e}")

        try:
            if G.number_of_nodes() == 0:
                raise RuntimeError("Katz centralitycalculation failed：textmissingnode")

            katz_dict = nx.katz_centrality_numpy(
                G,
                alpha=0.05,
                beta=1.0,
                normalized=False,
                weight=None
            )
            centralities['katz'] = np.array([katz_dict[node] for node in nodes], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Katz centralitycalculation failed: {e}")

        return centralities


class BackgroundCalculator:

    def __init__(self, processor_dir: str, output_dir: str,
                 gnn_stride: int = 3, pred_stride: int = 1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_workers: int = 4):

        self.processor_dir = processor_dir
        self.output_dir = output_dir
        self.gnn_stride = gnn_stride
        self.pred_stride = pred_stride
        self.device = torch.device(device)
        self.num_workers = num_workers

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"createoutputdirectory: {self.output_dir}")

        self.processor = TemporalGraphProcessor.load_processed_data(processor_dir)

        self.directed = self.processor.directed
        logger.info(f"texttype: {'directed graph' if self.directed else 'undirected graph'}")
        logger.info(f"use {self.num_workers} textcalculate")

        self.centrality_calculator = CentralityCalculator()
        self.window_attributes = {}
        self.split_info = {}
        self._load_window_attributes()

        logger.info(f"textdegreecalculatetextinitialized successfully")

    def build_temporal_graph(self, window_id: int) -> nx.Graph:
        try:
            if window_id not in self.processor.graphs:
                logger.warning(f"window {window_id} datatext")
                return nx.DiGraph() if self.directed else nx.Graph()

            window_data = self.processor.graphs[window_id]

            winfo = getattr(window_data, 'window_info', {})
            local_to_global: Dict[int, int] = winfo.get('local_to_global', {})
            global_nodes = winfo.get('global_nodes', [])

            G = nx.DiGraph() if self.directed else nx.Graph()

            if window_data is None:
                raise RuntimeError(f"window {window_id} textdatatextNone")

            if not hasattr(window_data, 'edge_index') or window_data.edge_index is None:
                logger.warning(f"window {window_id} textedge_indextextNone")
                return nx.DiGraph() if self.directed else nx.Graph()

            for node_str in global_nodes:
                G.add_node(node_str)

            edge_index = window_data.edge_index.cpu().numpy()
            edge_attr = None
            if hasattr(window_data, 'edge_attr') and window_data.edge_attr is not None:
                edge_attr = window_data.edge_attr.cpu().numpy()

            edges_with_data = []
            for i in range(edge_index.shape[1]):
                u_local = int(edge_index[0, i])
                v_local = int(edge_index[1, i])
                if u_local in local_to_global and v_local in local_to_global:
                    u = str(local_to_global[u_local])
                    v = str(local_to_global[v_local])
                    if u != v:
                        t = edge_attr[i] if (edge_attr is not None and i < len(edge_attr)) else 0
                        edges_with_data.append((u, v, {'time': t}))

            G.add_edges_from(edges_with_data)
            return G
        except Exception as e:
            raise RuntimeError(f"textwindow {window_id} text: {e}")

    def _load_window_attributes(self):
        """loadwindow_attributestextsplit_info"""
        try:
            pkl_files = glob.glob(os.path.join(self.processor_dir, "Email_window_attributes.pkl"))
            if not pkl_files:
                raise FileNotFoundError(f"text{self.processor_dir}directorytextnot foundwindow_attributes.pklfile")

            with open(pkl_files[0], 'rb') as f:
                data = pickle.load(f)
                self.window_attributes = data['window_attributes']
                self.split_info = data['split_info']

                logger.info(f"loadwindow_attributessuccess")
                logger.info(f"text {len(self.window_attributes)} textwindowtext")
                logger.info(f"text {len(self.split_info)} textsplittext")

        except Exception as e:
            raise RuntimeError(f"loadwindow_attributestext: {e}")

    def compute_centrality_scores(self, window_id: int, nodes: List) -> Dict:
        """textcalculatetextcentralitymetric，calculatetextground truthtext"""
        centrality_scores = {}

        try:
            G = self.build_temporal_graph(window_id)
            if G.number_of_nodes() == 0:
                raise ValueError(f"window {window_id} textis empty")

            centralities = self.centrality_calculator.compute_normalized_centralities(G, self.directed)

            node_index_map = {node: idx for idx, node in enumerate(G.nodes())}

            raw_scores = []
            node_list = []

            for node in nodes:
                if node not in node_index_map:
                    raise ValueError(f"node {node} textcurrentwindowtext，text")

                idx = node_index_map[node]
                degree_score = centralities['degree'][idx]
                # closeness_score = centralities['closeness'][idx]
                # betweenness_score = centralities['betweenness'][idx]
                # katz_score = centralities['katz'][idx]
                #
                # avg_score = (degree_score + closeness_score + betweenness_score + katz_score) / 4.0
                avg_score = degree_score

                if avg_score < 0:
                    print(f"window {window_id} textnode {node} text {idx} text")
                    print(f"node {node} textcentralitytext：")
                    print(f"  Degree: {degree_score}")
                    # print(f"  Closeness: {closeness_score}")
                    # print(f"  Betweenness: {betweenness_score}")
                    # print(f"  Katz: {katz_score}")

                raw_scores.append(avg_score)
                node_list.append(node)

            if not raw_scores:
                raise ValueError("missingtextnodetextcalculate")

            raw_scores_array = np.array(raw_scores)
            min_score = np.min(raw_scores_array)
            max_score = np.max(raw_scores_array)

            if max_score > min_score:
                normalized_scores = (raw_scores_array - min_score) / (max_score - min_score)
            else:
                normalized_scores = raw_scores_array
                logger.warning(f"window {window_id} textallnodetext，textmin-maxnormalized，textusetext")

            for node, norm_score in zip(node_list, normalized_scores):
                centrality_scores[node] = float(norm_score)

            del G, centralities
            gc.collect()

        except Exception as e:
            raise RuntimeError(f"calculatecentralitytext: {e}")

        return centrality_scores

    def rank_nodes_by_scores(self, degree: Dict[int, float]) -> List[Dict]:
        """
        textdegreetextnodetextranking（useaveragerankingtext）
        """

        sorted_items = sorted(degree.items(), key=lambda x: x[1], reverse=True)

        ranking_list = []

        i = 0
        while i < len(sorted_items):
            j = i
            current_score = sorted_items[i][1]
            while j < len(sorted_items) and abs(sorted_items[j][1] - current_score) < 1e-9:
                j += 1

            average_rank = (i + 1 + j) / 2.0

            for k in range(i, j):
                node_id, score = sorted_items[k]
                ranking_list.append({
                    'node': node_id,
                    'degree_score': score,
                    'rankings': average_rank
                })

            i = j

        return ranking_list

    def calculate_true_ranking(self):
        try:
            logger.info("textcalculateallwindowtextground truthranking...")

            all_split_indices = sorted(self.processor.split_to_windows.keys())
            logger.info(f"text {len(all_split_indices)} textsplit: {all_split_indices}")

            all_rankings = {}

            for split_index in tqdm(all_split_indices, desc="calculatetextsplitranking"):
                try:
                    if split_index not in self.split_info:
                        raise RuntimeError(f"split {split_index} textsplit_infotext")

                    split_type = self.split_info[split_index]['type']
                    windows = self.processor.split_to_windows[split_index]
                    if not windows:
                        raise RuntimeError(f"split{split_index}missingwindow")

                    target_window = windows[-1]
                    logger.info(f"calculatesplit{split_index}textwindow{target_window}textranking")

                    G = self.build_temporal_graph(target_window)
                    if G.number_of_nodes() == 0:
                        raise ValueError(f"window{target_window}textis empty")

                    all_nodes = sorted(list(G.nodes()))
                    scores = self.compute_centrality_scores(target_window, all_nodes)

                    rankings = self.rank_nodes_by_scores(scores)

                    ranking_key = f"split_{split_index}_window_{target_window}"
                    all_rankings[ranking_key] = rankings

                    logger.info(f"split {split_type}_split_{split_index} window {target_window} rankingcalculatecompleted，text {len(rankings)} textnode")

                    del G, scores
                    gc.collect()

                except Exception as e:
                    raise RuntimeError(f"textsplit {split_index} text: {e}")

            self._save_rankings(all_rankings)

        except Exception as e:
            raise RuntimeError(f"calculateground truthrankingtext: {e}")

    def _save_rankings(self, all_rankings: Dict):
        """
        saverankingresults
        """
        try:
            logger.info("textsaverankingresults...")

            for ranking_key, rankings in all_rankings.items():
                nodes = np.array([str(it['node']) for it in rankings], dtype=str)
                scores = np.array([float(it.get('degree_score', 0.0)) for it in rankings], dtype=np.float32)
                ranks = np.array([int(it.get('rankings', 0)) for it in rankings], dtype=np.int32)

                df = pd.DataFrame({
                    "nodes": nodes,
                    "scores": scores,
                    "ranks": ranks,
                })

                parquet_path = os.path.join(self.output_dir, f"{ranking_key}_ranks.parquet")
                df.to_parquet(parquet_path, index=False)

                logger.info(f"saverankingfile: {parquet_path}")

        except Exception as e:
            raise RuntimeError(f"saverankingresultstext: {e}")

def main():
    """text"""
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


    config = {
        'processor_dir': "./data/processed_data",
        'output_dir': "./data/background_rankings",
        'gnn_stride': 3,
        'pred_stride': 1,
        'num_workers': 18
    }

    try:
        logger.info("configurationparameter:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        if not os.path.exists(config['processor_dir']):
            raise RuntimeError(f"textdatadirectorytext: {config['processor_dir']}")

        calculator = BackgroundCalculator(**config)
        calculator.calculate_true_ranking()

    except Exception as e:
        logger.error(f"calculateground truthrankingtext: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
