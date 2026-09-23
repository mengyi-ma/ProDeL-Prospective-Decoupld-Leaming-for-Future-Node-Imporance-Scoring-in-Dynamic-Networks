import torch
from torch.utils.data import Dataset
from types import SimpleNamespace
from typing import  List
import sys
import os
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logger = logging.getLogger("TemporalGraphDataset")
logger.setLevel(logging.WARNING)

class SplitTemporalGraphDataset(Dataset):
    """
    text = textsplit
    inputwindow：textsplittext gnn_stride textwindow
    textwindow：textsplittextwindow（split_windows[-1]）
    """
    def __init__(self, processor, config: dict, split_indices: List[int], split_type: str = "train"):
        super().__init__()
        self.processor = processor
        self.config = config
        self.split_indices = [int(x) for x in split_indices]
        self.split_type = split_type
        self.gnn_stride = int(config.get("gnn_stride", 3))

        self._split_windows = {}
        self._window_cache = {}

        for split_idx in self.split_indices:
            windows = self.processor.get_split_windows(split_idx)
            if len(windows) < self.gnn_stride:
                raise RuntimeError(
                    f"split {split_idx} textwindowtext：need >= gnn_stride({self.gnn_stride})，actual {len(windows)}"
                )
            self._split_windows[split_idx] = windows

            for wid in windows:
                wid = int(wid)
                if wid in self._window_cache:
                    continue
                graph_data = self.processor.graphs[wid]

                if not hasattr(graph_data, "x") or graph_data.x is None:
                    raise RuntimeError(f"window {wid} missingnode features")
                if not hasattr(graph_data, "edge_index") or graph_data.edge_index is None:
                    raise RuntimeError(f"window {wid} missingedge index")
                if not hasattr(graph_data, "adj_matrix") or graph_data.adj_matrix is None:
                    raise RuntimeError(f"window {wid} missingadjacency matrix")
                if not hasattr(graph_data, "laplacian_matrix") or graph_data.laplacian_matrix is None:
                    raise RuntimeError(f"window {wid} missingLaplacian matrix")
                if not hasattr(graph_data, "window_info"):
                    raise RuntimeError(f"window {wid} missing window_info")

                self._window_cache[wid] = {
                    "node_features": graph_data.x.detach().cpu().float(),
                    "edge_index": graph_data.edge_index.detach().cpu().long(),
                    "adj_matrix": graph_data.adj_matrix.detach().cpu().float(),
                    "laplacian_matrix": graph_data.laplacian_matrix.detach().cpu(),
                    "window_info": graph_data.window_info,
                }

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, idx):
        split_idx = int(self.split_indices[idx])
        split_windows = self._split_windows[split_idx]

        input_window_ids = [int(w) for w in split_windows[:self.gnn_stride]]
        target_window_id = int(split_windows[-1])

        windows = []
        for wid in input_window_ids:
            w = self._window_cache[wid]
            windows.append(SimpleNamespace(
                split_idx=torch.tensor([split_idx], dtype=torch.long),
                split_type=self.split_type,
                window_id=torch.tensor([wid], dtype=torch.long),
                node_features=w["node_features"],
                edge_index=w["edge_index"],
                adj_matrix=w["adj_matrix"],
                laplacian_matrix=w["laplacian_matrix"],
                window_info=w["window_info"]
            ))

        return {
            "split_idx": split_idx,
            "target_window_id": target_window_id,
            "windows": windows
        }
