import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import logging
import types

logger = logging.getLogger("gnn")


class GCNLayer(nn.Module):
    """ textGCN """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        # LXW + b

        support = torch.matmul(x.to(torch.float32), self.weights.to(torch.float32))  # [num_nodes, output_dim]
        output = torch.sparse.mm(laplacian.to(torch.float32), support) + self.bias.to(torch.float32)  # [num_nodes, output_dim]

        output = output.to(x.dtype)

        return torch.tanh(output)


class GCNModel(nn.Module):
    """ texteachwindow：textGCN  """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:

        h1 = self.gcn1(x, laplacian)
        h1 = self.activation(h1)
        h1 = self.dropout(h1)

        z = self.gcn2(h1, laplacian)
        z = self.activation(z)

        if self.training and z.requires_grad:
            if torch.isnan(z).any() or torch.isinf(z).any():
                logger.warning(f"GCNModeloutputcontainstext: NaN={torch.isnan(z).any()}, Inf={torch.isinf(z).any()}")
        # =========================================

        return z


class GRUWrapper(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """GRUtext，textused fortext"""
        output, hidden_state = self.gru(x, hidden_state)

        if self.training:
            if output.requires_grad and (torch.isnan(output).any() or torch.isinf(output).any()):
                logger.warning(f"GRUWrapperoutputcontainstext: NaN={torch.isnan(output).any()}, Inf={torch.isinf(output).any()}")
            if hidden_state is not None and hidden_state.requires_grad:
                if torch.isnan(hidden_state).any() or torch.isinf(hidden_state).any():
                    logger.warning(f"GRUWrappertextcontainstext")
        # =========================================

        return output, hidden_state

class BinClassMLP(nn.Module):
    """
    MLP for node classification: predict probability of being in top_K
    """

    def __init__(self, input_dim: int,dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.mlp(x).squeeze(-1)

        if self.training and output.requires_grad:
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.warning(
                    f"BinClassMLPoutputcontainstext: NaN={torch.isnan(output).any()}, Inf={torch.isinf(output).any()}")
        # =========================================

        return output

class NodeScoreMLP(nn.Module):
    """
    MLP for node scoring from GRU output
    """

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.mlp(x).squeeze(-1)

        if self.training and output.requires_grad:
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.warning(
                    f"NodeScoreMLPoutputcontainstext: NaN={torch.isnan(output).any()}, Inf={torch.isinf(output).any()}")
        # =========================================

        return output

class GraphAutoEncoder(nn.Module):
    """text：usenodetextadjacency matrix A_hat """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """calculatetextadjacency matrix A_hat = sigmoid(ZZ^T) """
        zz_t = torch.matmul(z, z.t())  # [num_nodes, num_nodes]
        a_hat = torch.sigmoid(zz_t)

        if self.training and a_hat.requires_grad:
            if torch.isnan(a_hat).any() or torch.isinf(a_hat).any():
                logger.warning(f"GraphAutoEncoderoutputcontainstext: NaN={torch.isnan(a_hat).any()}, Inf={torch.isinf(a_hat).any()}")
        # ================================================

        return a_hat

class GCNBasedTemporalGraphModel(nn.Module):
    """
    Modified GNN model based on GCN and GRU architecture
    1. Two-layer GCN for each window
    2. GRU for sequential processing
    3. Node scoring via MLP
    4. Link prediction via graph autoencoder
    """

    def __init__(self,
                 node_feat_dim: int = 4,
                 gcn_hidden_dim: int = 64,
                 gcn_output_dim: int = 32,
                 gru_output_dim: int = 64,
                 gru_num_layers: int = 2,
                 dropout: float = 0.2,
                 gnn_stride: int = 3,
                 pred_stride: int = 1):
        super().__init__()

        # Parameters
        self.node_feat_dim = node_feat_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gcn_output_dim = gcn_output_dim
        self.gru_output_dim = gru_output_dim
        self.gnn_stride = gnn_stride
        self.pred_stride = pred_stride

        # Two-layer GCN for each window
        self.gcn = GCNModel(
            input_dim=node_feat_dim,
            hidden_dim=gcn_hidden_dim,
            output_dim=gcn_output_dim,
            dropout=dropout
        )

        # GRU for sequential processing
        self.gru = GRUWrapper(
            input_dim=gcn_output_dim,
            hidden_dim=gru_output_dim,
            num_layers=gru_num_layers,
            dropout=dropout
        )

        self.beta = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        # Node scoring MLP
        self.node_scoring_mlp = NodeScoreMLP(
            input_dim=gru_output_dim,
            dropout=dropout
        )

        # Node classification MLP for top_K prediction
        self.node_classification_mlp = BinClassMLP(
            input_dim=gru_output_dim,
            dropout=dropout
        )

        self.training_stage = 'pretraining'

        self.node_norm = nn.LayerNorm(gcn_output_dim, eps=1e-5)

        self.graph_ae = GraphAutoEncoder()

        self.output_global_nodes_by_window_key = {}

        self._reset_parameters()

    def set_training_stage(self, stage: str):
        """settrainingstage"""
        assert stage in ['pretraining', 'finetuning'], f"texttrainingstage: {stage}"
        self.training_stage = stage

        if stage == 'pretraining':
            for param in self.node_scoring_mlp.parameters():
                param.requires_grad = False
            # for param in self.temporal_attention.parameters():
            #     param.requires_grad = True
            self.beta.requires_grad = True # β
            for param in self.node_classification_mlp.parameters():
                param.requires_grad = True
            for param in self.gcn.parameters():
                param.requires_grad = True
            for param in self.gru.parameters():
                param.requires_grad = True
        else:
            for param in self.gcn.parameters():
                param.requires_grad = True
            for param in self.node_classification_mlp.parameters():
                param.requires_grad = False
            self.beta.requires_grad = True
            for param in self.gru.parameters():
                param.requires_grad = True
            for param in self.node_scoring_mlp.parameters():
                param.requires_grad = True

    def _reset_parameters(self):

        uninitialized = []
        for name, param in self.named_parameters():
            if param.data is None or (isinstance(param.data, torch.Tensor) and param.data.numel() == 0):
                uninitialized.append(name)

        if uninitialized:
            logger.warning(f"textparametertextinitialize: {uninitialized}")
        # ================================================

    def pad_matrices(self, gcn_outputs: List[torch.Tensor],
                     window_global_nodes: List[List[str]],
                     device: torch.device) -> Tuple[torch.Tensor, List[List[str]], Dict[str, int]]:
        """
        textwindowtextGCNoutputtext

        Args:
            gcn_outputs: textwindowtextGCNoutputtext，eachtext [num_nodes_i, gcn_output_dim]
            window_global_nodes: textwindowtextnodeIDtext
            device: calculatedevice

        Returns:
            padded_outputs: textGCNoutput [num_windows, total_global_nodes, gcn_output_dim]
            padded_global_nodes: textnodeIDtext
            global_node_to_idx: textnodetext
        """
        if not gcn_outputs:
            raise RuntimeError(f"gcn_outputs = empty")

        all_global_nodes = set()
        for global_nodes in window_global_nodes:
            all_global_nodes.update(global_nodes)

        sorted_global_nodes = sorted(list(all_global_nodes))
        total_global_nodes = len(sorted_global_nodes)

        global_node_to_idx = {node: idx for idx, node in enumerate(sorted_global_nodes)}

        num_windows = len(gcn_outputs)
        gcn_output_dim = gcn_outputs[0].shape[1]

        padded_list = []

        for i, (z, global_nodes) in enumerate(zip(gcn_outputs, window_global_nodes)):
            window_tensor = torch.zeros(
                total_global_nodes, gcn_output_dim,
                device=device, dtype=z.dtype
            )

            indices = torch.tensor([global_node_to_idx[node] for node in global_nodes],
                                   device=device, dtype=torch.long)

            window_tensor[indices] = z
            padded_list.append(window_tensor)

        padded_outputs = torch.stack(padded_list, dim=0)
        padded_outputs.requires_grad_(True)

        if self.training and padded_outputs.requires_grad:
            if torch.isnan(padded_outputs).any() or torch.isinf(padded_outputs).any():
                logger.warning(f"pad_matricesoutputcontainstext")
        # =========================================

        return padded_outputs, [sorted_global_nodes] * num_windows, global_node_to_idx

    def forward(
            self,
            batch_windows: List[List[types.SimpleNamespace]],
            split_indices: List[int],
            target_window_ids: List[int]
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[Optional[Dict[str, torch.Tensor]]], List[
            Optional[Dict[str, torch.Tensor]]]
    ]:
        device = next(self.parameters()).device

        B = len(batch_windows)
        if not (len(split_indices) == B and len(target_window_ids) == B):
            raise ValueError(
                f"batchtext: windows={B}, split_indices={len(split_indices)}, target_window_ids={len(target_window_ids)}")

        padded_list = []
        padded_global_nodes_list = []
        lengths = []
        last_gnn_window_ids = []

        for b in range(B):
            split_idx = int(split_indices[b])
            window_data_list = batch_windows[b]
            target_window_id = int(target_window_ids[b])

            if len(window_data_list) != self.gnn_stride:
                raise ValueError(
                    f"split {split_idx} inputwindowtext==gnn_stride({self.gnn_stride})，actual {len(window_data_list)}")

            window_ids = []
            gcn_outputs = []
            window_global_nodes_list = []
            node_features_list = []
            laplacians_list = []

            for i in range(self.gnn_stride):
                data = window_data_list[i]
                window_id = int(data.window_id)
                window_ids.append(window_id)

                node_features_list.append(data.node_features.to(device, dtype=torch.float32))

                if 'window_info' not in data.__dict__:
                    raise ValueError(f"split {split_idx} window {window_id} missing window_info")

                window_info = data.window_info
                if ('global_nodes' not in window_info) or ('global_to_local' not in window_info):
                    raise ValueError(
                        f"split {split_idx} window {window_id} missing global_nodes/global_to_local in window_info")

                window_global_nodes_list.append(window_info['global_nodes'])

                if hasattr(data, 'laplacian_matrix'):
                    laplacians_list.append(data.laplacian_matrix.to(device, dtype=torch.float32))
                else:
                    raise RuntimeError(f"split {split_idx} window {window_id} missingtextcalculatelaplacian_matrix")

            target_time = float(target_window_id)

            window_times = torch.tensor([float(wid) for wid in window_ids], device=device, dtype=torch.float32)
            time_diffs = target_time - window_times
            if (time_diffs < 0).any():
                raise ValueError(f"split {split_idx} time windowtexterror：target({target_window_id}) textinputwindow")

            beta = torch.nn.functional.softplus(self.beta)
            beta_clamped = torch.clamp(beta, max=10.0)
            exponent = torch.clamp(-beta_clamped * time_diffs, min=-20.0, max=20.0)
            decays = torch.clamp(torch.exp(exponent), min=1e-12, max=1e12)

            for i in range(self.gnn_stride):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                    z = self.gcn(node_features_list[i], laplacians_list[i])

                z = self.node_norm(z.float())
                z = z.to(node_features_list[i].dtype)
                z = z * decays[i]

                gcn_outputs.append(z)

            padded_gcn_outputs, padded_global_nodes, _ = self.pad_matrices(gcn_outputs, window_global_nodes_list,
                                                                           device)

            # padded_gcn_outputs: [T, N_b, D]
            padded_list.append(padded_gcn_outputs)
            padded_global_nodes_list.append(padded_global_nodes[0])  # List[str]
            lengths.append(padded_gcn_outputs.shape[1])

            last_gnn_window_id = int(window_data_list[self.gnn_stride - 1].window_id)
            last_gnn_window_ids.append(last_gnn_window_id)

        concat_inputs = torch.cat(padded_list, dim=1)

        hidden_state = None
        all_gru_outputs = []

        seq_len = concat_inputs.shape[0]

        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            for t in range(seq_len):
                current_input = concat_inputs[t:t + 1]  # [1, sum_N, D]
                gru_output, hidden_state = self.gru(current_input, hidden_state)
                all_gru_outputs.append(gru_output)

        all_gru_states = torch.cat(all_gru_outputs, dim=0)  # [T, sum_N, D_gru]
        last_state = all_gru_states[-1]

        # split back
        next_repr_list = []
        a_hat_list = []
        pred_score_list = []
        node_probs_list = []

        start = 0
        for b in range(B):
            split_idx = int(split_indices[b])
            Nb = int(lengths[b])
            end = start + Nb

            next_window_pred = last_state[start:end]  # [Nb, D_gru]
            a_hat = self.graph_ae(next_window_pred)

            if self.training_stage == 'finetuning':
                node_scores = self.node_scoring_mlp(next_window_pred)
                min_score = node_scores.min()
                max_score = node_scores.max()
                if max_score - min_score > 1e-8:
                    node_scores_normalized = (node_scores - min_score) / (max_score - min_score)
                else:
                    node_scores_normalized = node_scores
                    logger.warning(f"alltext")

                pred_score_dict = {
                    node_id: score for node_id, score in zip(padded_global_nodes_list[b], node_scores_normalized)
                }
                node_probs_dict = None
            else:
                node_probs = self.node_classification_mlp(next_window_pred)
                node_probs_dict = {
                    node_id: prob for node_id, prob in zip(padded_global_nodes_list[b], node_probs)
                }
                pred_score_dict = None

            key = (int(split_idx), int(last_gnn_window_ids[b]))
            self.output_global_nodes_by_window_key[key] = padded_global_nodes_list[b]

            next_repr_list.append(next_window_pred)
            a_hat_list.append(a_hat)
            pred_score_list.append(pred_score_dict)
            node_probs_list.append(node_probs_dict)

            start = end

        return next_repr_list, a_hat_list, pred_score_list, node_probs_list



