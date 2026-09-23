import numpy as np
import torch
import logging
from torch import Tensor
import torch.nn.functional as F
from sklearn.metrics import ndcg_score, roc_auc_score, f1_score

logger = logging.getLogger("Metrics")


def get_common_nodes_list(pred_score_dict: dict, true_score_dict: dict):
    """
    “textnode”，returntextnodetext（textID）。
    """
    if not pred_score_dict:
        raise ValueError("pred_score_dict is empty，textnode")
    if not true_score_dict:
        raise ValueError("true_score_dict is empty，textnode")

    pred_nodes = set(str(n) for n in pred_score_dict.keys())
    true_nodes = set(str(n) for n in true_score_dict.keys())
    common_nodes = pred_nodes & true_nodes

    if not common_nodes:
        raise ValueError("textmissingtextnode")

    return sorted(list(common_nodes), key=lambda x: int(x))


def get_filtered_new_ranking(true_score_dict: dict, pred_score_dict: dict, split_idx=0):
    if not pred_score_dict:
        raise ValueError("predictiontextis empty，textcalculatetextranking")
    if not true_score_dict:
        raise ValueError("ground truthtextis empty，textcalculatetextranking")

    pred_nodes = set(str(n) for n in pred_score_dict.keys())
    true_nodes = set(str(n) for n in true_score_dict.keys())

    common_nodes = pred_nodes & true_nodes
    if not common_nodes:
        raise ValueError("predictiontextground truthtextmissingtextnode，textcalculatemetric")

    common_nodes_list = sorted(list(common_nodes), key=lambda x: int(x))

    pred_scores_list = []
    true_scores_list = []

    first_pred_score = next(iter(pred_score_dict.values()))
    if isinstance(first_pred_score, torch.Tensor):
        target_device = first_pred_score.device
    else:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for node_id in common_nodes_list:
        pred_score = pred_score_dict[node_id]
        true_score = true_score_dict[node_id]

        if isinstance(pred_score, torch.Tensor):
            pred_scores_list.append(pred_score)
        else:
            pred_scores_list.append(torch.tensor(pred_score, dtype=torch.float32, device=target_device))

        if isinstance(true_score, torch.Tensor):
            true_scores_list.append(true_score)
        else:
            true_scores_list.append(torch.tensor(true_score, dtype=torch.float32, device=target_device))

    pred_scores_orig = torch.stack(pred_scores_list)
    true_scores_orig = torch.stack(true_scores_list)

    new_true_score_dict = {}
    new_pred_score_dict = {}

    for i, node_id in enumerate(common_nodes_list):
        new_true_score_dict[node_id] = true_scores_orig[i]
        new_pred_score_dict[node_id] = pred_scores_orig[i]

    def compute_average_ranks(scores):
        """textcalculateaverageranking"""
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_scores = scores[sorted_indices]

        n = len(scores)
        ranks = torch.zeros(n, dtype=torch.float32, device=scores.device)

        diff = torch.abs(sorted_scores[1:] - sorted_scores[:-1])
        group_boundaries = torch.where(diff > 1e-9)[0] + 1
        group_boundaries = torch.cat([torch.tensor([0], device=scores.device),
                                      group_boundaries,
                                      torch.tensor([n], device=scores.device)])

        for i in range(len(group_boundaries) - 1):
            start_idx = group_boundaries[i]
            end_idx = group_boundaries[i + 1]
            avg_rank = (start_idx + end_idx + 1) / 2.0
            ranks[sorted_indices[start_idx:end_idx]] = avg_rank

        return ranks

    new_true_rank = compute_average_ranks(true_scores_orig)
    new_pred_rank = compute_average_ranks(pred_scores_orig)

    if new_true_rank.numel() != new_pred_rank.numel():
        raise RuntimeError("textnodetextground truth/predictionrankingtext，textcalculatemetric")

    return {
        "common_nodes": common_nodes_list,
        "new_true_rank": new_true_rank,
        "new_pred_rank": new_pred_rank,
        "new_true_score_dict": new_true_score_dict,
        "new_pred_score_dict": new_pred_score_dict,
        "true_scores": true_scores_orig,
        "pred_scores": pred_scores_orig
    }


class RankingMetrics:
    """rankingmetriccalculatetext，textrankingmetric"""

    def __init__(self, true_score_dict: dict, pred_score_dict: dict, device=None, split_idx=0):
        """
        initializerankingmetriccalculatetext

        Args:
            true_score_dict: ground truthtext {node_id: score}
            pred_score_dict: predictiontext {node_id: score}
            device: calculatedevice
            split_idx: splittext
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.split_idx = split_idx

        with torch.no_grad():
            rank_data = get_filtered_new_ranking(true_score_dict, pred_score_dict, split_idx)

        self.common_nodes = rank_data['common_nodes']
        self.true_ranks = rank_data['new_true_rank'].detach().to(self.device)
        self.pred_ranks = rank_data['new_pred_rank'].detach().to(self.device)
        self.true_scores = rank_data['true_scores'].detach().to(self.device)
        self.pred_scores = rank_data['pred_scores'].detach().to(self.device)

        self.true_score_dict = rank_data['new_true_score_dict']
        self.pred_score_dict = rank_data['new_pred_score_dict']

        self._results = {}

    def compute_all(self, ndcg_ks, hit_rate_k, split_type=None, split_idx=None):
        """
        textcalculateallmetric

        Args:
            ndcg_ks: NDCGtextktext
            hit_rate_k: textktext

        Returns:
            dict: containsallmetricresultstext
        """
        results = {}

        results['ndcg'] = self.compute_ndcg(ndcg_ks)

        results['hit_rate'] = self.compute_hit_rate(hit_rate_k)

        auc, f1, mrr = self.compute_auc_f1_mrr(hit_rate_k)
        results['auc'] = auc
        results['f1'] = f1
        results['mrr'] = mrr

        self._results = results
        return results

    def compute_auc_f1_mrr(self,k):
        """
        calculateAUCtextF1
        Args:
            k: textktextnodetext
        Returns:
            tuple: (auc, f1)
        """
        k = 20
        with torch.no_grad():
            actual_k = min(k, len(self.common_nodes))
            if actual_k <= 0 or len(self.common_nodes) <= 1 or actual_k == len(self.common_nodes):
                return 0.5, 0.0,0.0

            _, true_topk_idx = torch.topk(self.true_scores, actual_k, largest=True)
            y_true = torch.zeros(len(self.true_scores), dtype=torch.int32, device=self.true_scores.device)
            y_true[true_topk_idx] = 1

            _, pred_topk_idx = torch.topk(self.pred_scores, actual_k, largest=True)
            y_pred = torch.zeros(len(self.pred_scores), dtype=torch.int32, device=self.pred_scores.device)
            y_pred[pred_topk_idx] = 1

            y_true_np = y_true.cpu().numpy()
            y_scores_np = self.pred_scores.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()

            try:
                auc = roc_auc_score(y_true_np, y_scores_np)
            except Exception:
                raise ValueError(f"AUCcalculation failed")

            try:
                f1 = f1_score(y_true_np, y_pred_np)
            except Exception:
                raise ValueError(f"F1calculation failed")

            try:
                pos_indices = set(true_topk_idx.tolist())
                _, pred_sorted_idx = torch.sort(self.pred_scores, descending=True)

                mrr = 0.0
                for rank_minus_1, idx in enumerate(pred_sorted_idx.tolist()):
                    if idx in pos_indices:
                        mrr = 1.0 / (rank_minus_1 + 1.0)
                        break
            except Exception:
                raise ValueError(f"MRRcalculation failed")

            return float(auc), float(f1), float(mrr)

    def compute_ndcg(self, ks):
        """calculatetextktextNDCG """
        ndcg_results = {}

        with torch.no_grad():
            max_rank = self.true_ranks.max().item()
            min_rank = self.true_ranks.min().item()

            if torch.any(torch.isnan(self.true_ranks)):
                raise ValueError("ground truthrankingcontainsNaNtext，textcalculateNDCG")

            if torch.any(torch.isinf(self.true_ranks)):
                raise ValueError("ground truthrankingcontainsInftext，textcalculateNDCG")

            if max_rank == min_rank:
                logger.warning("allground truthrankingtext，textNDCGcalculate")
                for k in ks:
                    ndcg_results[k] = 0.0
                return ndcg_results

            relevance = 1.0 - (self.true_ranks.to(torch.float32) - float(min_rank)) / float(max_rank - min_rank)
            relevance = relevance.to(torch.float32)

            relevance_np = relevance.cpu().numpy().reshape(1, -1)
            pred_scores_np = self.pred_scores.cpu().numpy().reshape(1, -1)

            for k in ks:
                actual_k = min(k, len(self.common_nodes))

                if len(self.common_nodes) <= 1:
                    logger.warning(f"number of nodestext{len(self.common_nodes)}，textcalculateNDCG@{k}")
                    ndcg_results[k] = 0.0
                    continue

                try:
                    ndcg_val = ndcg_score(
                        relevance_np,
                        pred_scores_np,
                        k=actual_k,
                        ignore_ties=True
                    )
                    ndcg_results[k] = float(ndcg_val)
                except Exception as e:
                    raise RuntimeError(f"calculateNDCG@{k}text: {e}")

        torch.cuda.empty_cache()
        return ndcg_results

    def compute_hit_rate(self, k):
        """
        calculatetext

        Args:
            k: textktextnode

        Returns:
            float: text
        """
        with torch.no_grad():
            actual_k = min(k, len(self.common_nodes))
            _, true_topk_idx = torch.topk(self.true_scores, actual_k, largest=False)
            true_topk_nodes = set([self.common_nodes[idx] for idx in true_topk_idx.tolist()])

            _, pred_topk_idx = torch.topk(self.pred_scores, actual_k, largest=False)
            pred_topk_nodes = set([self.common_nodes[idx] for idx in pred_topk_idx.tolist()])

            hit_count = len(true_topk_nodes & pred_topk_nodes)

            torch.cuda.empty_cache()

            return float(hit_count / actual_k)

    def get_rank_data(self):
        """ textrankingdata  Returns:  dict: containsrankingdatatext   """
        return {
            'common_nodes': self.common_nodes,
            'true_ranks': self.true_ranks,
            'pred_ranks': self.pred_ranks,
            'true_scores': self.true_scores,
            'pred_scores': self.pred_scores,
            'true_score_dict': self.true_score_dict,
            'pred_score_dict': self.pred_score_dict
        }


class A_hat_Calculator:
    """adjacency matrixtextcalculatetext"""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.gnn_stride = config.get('gnn_stride', 3)
        self.is_undirected = config.get('is_undirected', True)

        self.adj_cache = {}
        self.global_nodes_cache = {}
        self.union_cache = {}

    def get_window_adjacency(self, processor, window_id):
        """textwindowtextadjacency matrix"""
        cache_key = f"window_{window_id}"

        if cache_key in self.adj_cache:
            return self.adj_cache[cache_key]

        try:
            window_data = processor.graphs.get(window_id)
            if window_data is None:
                raise RuntimeError(f"window {window_id} datais empty")

            adj_matrix = window_data.adj_matrix
            if adj_matrix is None:
                raise RuntimeError(f"window {window_id} missingadjacency matrix")

            adj_matrix = adj_matrix.to(self.device)
            self.adj_cache[cache_key] = adj_matrix

            return adj_matrix

        except Exception as e:
            raise RuntimeError(f"textwindow {window_id} datatext: {str(e)}")

    def compute_union_adjacency_for_split(self, processor, split_idx):
        """
        textsplitcalculatetextadjacency matrix

        Args:
            processor: datatext
            split_idx: splittext

        Returns:
            union_adj: textadjacency matrix
            global_node_to_idx: textnodetext
        """
        cache_key = f"split_{split_idx}_union"

        if cache_key in self.union_cache:
            return self.union_cache[cache_key]

        try:
            split_windows = processor.get_split_windows(split_idx)
            gnn_window_ids = split_windows[:self.gnn_stride]

            if not gnn_window_ids:
                raise RuntimeError("GNNwindowIDtextis empty")

            all_global_nodes = set()
            all_src_global_nodes = []
            all_dst_global_nodes = []

            for window_id in gnn_window_ids:
                window_data = processor.graphs.get(window_id)
                if window_data is None:
                    raise RuntimeError(f"window {window_id} datais empty")

                window_info = window_data.window_info
                local_to_global = window_info.get('local_to_global', {})
                global_nodes = window_info.get('global_nodes', [])

                if not local_to_global:
                    raise RuntimeError(f"window {window_id} missing local_to_global text")
                if not global_nodes:
                    raise RuntimeError(f"window {window_id} missingtextnodetext")

                all_global_nodes.update(global_nodes)

                adj_matrix = window_data.adj_matrix
                if adj_matrix is None:
                    raise RuntimeError(f"window {window_id} missingadjacency matrix")

                if hasattr(adj_matrix, 'coo'):
                    row, col, _ = adj_matrix.coo()
                    row = row.cpu().numpy()
                    col = col.cpu().numpy()
                else:
                    raise RuntimeError(f"missingtextadjacency matrix")

                for src_local, dst_local in zip(row, col):
                    if src_local in local_to_global and dst_local in local_to_global:
                        src_global = str(local_to_global[src_local])
                        dst_global = str(local_to_global[dst_local])

                        all_src_global_nodes.append(src_global)
                        all_dst_global_nodes.append(dst_global)
                    else:
                        raise RuntimeError(
                            f"window {window_id}: text {src_local} text {dst_local} text local_to_global text")

            sorted_global_nodes = sorted(list(all_global_nodes))
            total_global_nodes = len(sorted_global_nodes)

            global_node_to_idx = {node: idx for idx, node in enumerate(sorted_global_nodes)}

            union_adj = torch.zeros((total_global_nodes, total_global_nodes), device=self.device, dtype=torch.float32)

            edge_src_indices = []
            edge_dst_indices = []

            for src_global, dst_global in zip(all_src_global_nodes, all_dst_global_nodes):
                src_idx = global_node_to_idx[src_global]
                dst_idx = global_node_to_idx[dst_global]

                if self.is_undirected:
                    if src_idx <= dst_idx:
                        edge_src_indices.append(src_idx)
                        edge_dst_indices.append(dst_idx)
                else:
                    edge_src_indices.append(src_idx)
                    edge_dst_indices.append(dst_idx)

            edge_src_tensor = torch.tensor(edge_src_indices, device=self.device, dtype=torch.long)
            edge_dst_tensor = torch.tensor(edge_dst_indices, device=self.device, dtype=torch.long)

            union_adj[edge_src_tensor, edge_dst_tensor] = 1.0

            if self.is_undirected:
                union_adj[edge_dst_tensor, edge_src_tensor] = 1.0

            self.union_cache[cache_key] = (union_adj, global_node_to_idx)

            return union_adj, global_node_to_idx

        except Exception as e:
            raise RuntimeError(f"calculatesplit {split_idx} textadjacency matrixtext: {str(e)}")

    def build_recon_mats_intersection(
            self,
            processor,
            split_idx: int,
            target_window_id: int,
            model_global_nodes: list,
            a_hat: torch.Tensor
    ):
        """
        textused forGAEtextlosstext (A_hat_sub, A_sub)：

        (1) text：eachsplittextwindownode text textwindownodetext；- “textwindownode”text
        (2) textwindowtextnodetextground truthtext A_sub
        (3) A_hat textnodetext A_hat_sub
        (4) returnused for BCE textlosstext A_hat_sub text A_sub
        """
        split_idx = int(split_idx)
        target_window_id = int(target_window_id)

        if not model_global_nodes:
            raise RuntimeError("model_global_nodes is empty，text a_hat textnodetext")

        split_windows = processor.get_split_windows(split_idx)
        gnn_window_ids = split_windows[:self.gnn_stride]
        if len(gnn_window_ids) != self.gnn_stride:
            raise RuntimeError(f"split {split_idx}: gnn_window_ids numbertext gnn_stride({self.gnn_stride})")

        first3_union_nodes = set()
        for wid in gnn_window_ids:
            wdata = processor.graphs.get(int(wid))
            if wdata is None:
                raise RuntimeError(f"split {split_idx}: window {wid} datais empty")
            winfo = wdata.window_info
            if ('global_nodes' not in winfo) or (not winfo['global_nodes']):
                raise RuntimeError(f"split {split_idx}: window {wid} text global_nodes")
            first3_union_nodes.update(str(n) for n in winfo['global_nodes'])

        target_data = processor.graphs.get(target_window_id)
        if target_data is None:
            raise RuntimeError(f"split {split_idx}: textwindow {target_window_id} datais empty")

        target_info = target_data.window_info
        if ('global_nodes' not in target_info) or (not target_info['global_nodes']):
            raise RuntimeError(f"textwindow {target_window_id} text global_nodes")

        target_nodes_set = set(str(n) for n in target_info['global_nodes'])

        inter_nodes_list = get_common_nodes_list(
            pred_score_dict={nid: 1 for nid in first3_union_nodes},
            true_score_dict={nid: 1 for nid in target_nodes_set}
        )

        L = len(inter_nodes_list)
        if L <= 1:
            return None, None

        model_global_nodes = [str(n) for n in model_global_nodes]
        model_node_to_idx = {nid: i for i, nid in enumerate(model_global_nodes)}

        idxs = []
        for nid in inter_nodes_list:
            if nid not in model_node_to_idx:
                raise RuntimeError(f"split {split_idx}: textnode {nid} text model_global_nodes text，text a_hat")
            idxs.append(int(model_node_to_idx[nid]))

        a_hat = a_hat.to(self.device, non_blocking=True)
        idxs_t = torch.tensor(idxs, device=a_hat.device, dtype=torch.long)
        a_hat_sub = a_hat.index_select(0, idxs_t).index_select(1, idxs_t)

        if ('global_to_local' not in target_info) or (not target_info['global_to_local']):
            raise RuntimeError(f"textwindow {target_window_id} text global_to_local text")

        global_to_local = {str(k): int(v) for k, v in target_info['global_to_local'].items()}

        inter_local = []
        for nid in inter_nodes_list:
            if nid not in global_to_local:
                raise RuntimeError(f"textwindow {target_window_id}: textnode {nid} text global_to_local text")
            inter_local.append(global_to_local[nid])

        inter_local_t = torch.tensor(inter_local, device=self.device, dtype=torch.long)

        adj_matrix = target_data.adj_matrix
        if adj_matrix is None:
            raise RuntimeError(f"textwindow {target_window_id} missingadjacency matrix")
        if not hasattr(adj_matrix, "coo"):
            raise RuntimeError(f"textwindow {target_window_id} adjacency matrixtextCOOtext")

        row, col, _ = adj_matrix.coo()  # torch tensors
        row = row.to(self.device, non_blocking=True)
        col = col.to(self.device, non_blocking=True)

        mask = torch.isin(row, inter_local_t) & torch.isin(col, inter_local_t)
        row_f = row[mask]
        col_f = col[mask]

        max_local = int(inter_local_t.max().item())
        lookup = torch.full((max_local + 1,), -1, device=self.device, dtype=torch.long)
        lookup[inter_local_t] = torch.arange(L, device=self.device, dtype=torch.long)

        sub_r = lookup[row_f]
        sub_c = lookup[col_f]

        if (sub_r < 0).any() or (sub_c < 0).any():
            raise RuntimeError(f"split {split_idx}: text")

        A_sub = torch.zeros((L, L), device=self.device, dtype=torch.float32)
        A_sub[sub_r, sub_c] = 1.0
        A_sub[sub_c, sub_r] = 1.0

        if a_hat_sub.shape != A_sub.shape:
            raise RuntimeError(f"split {split_idx}: A_hat_sub text A_sub text {a_hat_sub.shape} vs {A_sub.shape}")

        return a_hat_sub, A_sub

def compute_topk_bce_loss(
        true_score_dict: dict,
        pred_prob_dict: dict,
        top_k: int = 50,
        device=None,
        split_idx: int = 0
) -> Tensor:
    """calculatenodetexttop_ktextBCEloss"""
    if not pred_prob_dict:
        raise ValueError("predictiontextis empty，textcalculateBCEloss")
    if not true_score_dict:
        raise ValueError("ground truthtextis empty，textcalculateBCEloss")

    try:
        rank_data = get_filtered_new_ranking(
            true_score_dict=true_score_dict,
            pred_score_dict=pred_prob_dict,
            split_idx=split_idx
        )

        true_scores = rank_data['true_scores']
        pred_probs = rank_data['pred_scores']

        actual_k = min(top_k, len(true_scores))

        total_nodes = len(true_scores)
        pos_weight_value = (total_nodes - actual_k) / actual_k
        pos_weight = torch.tensor([pos_weight_value], device=pred_probs.device, dtype=torch.float32)

        _, topk_indices = torch.topk(true_scores, actual_k, largest=True)

        labels = torch.zeros(len(true_scores), dtype=torch.float32, device=pred_probs.device)
        labels[topk_indices] = 1.0

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_probs,
            labels,
            reduction='mean',
            pos_weight=pos_weight
        )

        _, true_topk_indices = torch.topk(true_scores, actual_k, largest=True)

        _, pred_topk_indices = torch.topk(pred_probs, actual_k, largest=True)

        true_topk_set = set(true_topk_indices.cpu().numpy())
        pred_topk_set = set(pred_topk_indices.cpu().numpy())

        topk_accuracy = len(true_topk_set & pred_topk_set) / actual_k

        if not torch.isfinite(bce_loss):
            raise ValueError(f"split{split_idx}: BCElosstext")

        del rank_data
        torch.cuda.empty_cache()

        return bce_loss

    except Exception as e:
        raise RuntimeError(f"split{split_idx}: calculateBCElosstexterror: {str(e)}")


def compute_score_mse_loss(true_score_dict: dict,
                           pred_score_dict: dict,
                           top_k: int = 50,
                           device=None,
                           split_idx=0
                           ) -> Tensor:
    """ textMSElosscalculate，textcalculatetexttop_Ktextnode """
    if not pred_score_dict:
        raise ValueError("predictiontextis empty，textcalculateMSEloss")
    if not true_score_dict:
        raise ValueError("ground truthtextis empty，textcalculateMSEloss")

    rank_data = get_filtered_new_ranking(true_score_dict, pred_score_dict, split_idx)

    pred_scores_tensor = rank_data['pred_scores']
    true_scores_tensor = rank_data['true_scores']

    if pred_scores_tensor.shape != true_scores_tensor.shape:
        raise ValueError(f"predictiontextground truthtext: {pred_scores_tensor.shape} vs {true_scores_tensor.shape}")

    if len(pred_scores_tensor) == 0:
        raise RuntimeError("MSElosscalculate - textnodeis empty，return0loss")

    mse_loss = F.mse_loss(pred_scores_tensor, true_scores_tensor, reduction='mean')

    del rank_data
    torch.cuda.empty_cache()

    return mse_loss


def compute_reconstruction_loss(a_hat, a):
    """
    calculatetextloss：textloss
    Args:
        a_hat: textadjacency matrix [N, N]
        a: textadjacency matrix [N, N]
    Returns:
        reconstruction_loss: textlosstext
    """
    reconstruction_loss = torch.nn.functional.binary_cross_entropy(
        a_hat,
        a,
        reduction='mean'
    )

    return reconstruction_loss


def compute_total_loss(mse_loss: torch.Tensor,
                       recon_loss: torch.Tensor,
                       gamma: float = 0.5) -> torch.Tensor:
    if not torch.isfinite(mse_loss) or not torch.isfinite(recon_loss):
        raise ValueError("textlosstext")

    return mse_loss + gamma * recon_loss


def macro_average_roc(fprs, tprs, num_points=1001):
    if (not fprs) or (not tprs):
        raise ValueError("text fprs/tprs text，textaverage")
    grid = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    interp_tprs = []
    for fpr, tpr in zip(fprs, tprs):
        fpr = np.asarray(fpr, dtype=np.float64).reshape(-1)
        tpr = np.asarray(tpr, dtype=np.float64).reshape(-1)
        interp = np.interp(grid, fpr, tpr)
        interp_tprs.append(interp)
    mean_tpr = np.mean(np.stack(interp_tprs, axis=0), axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    return grid, mean_tpr
