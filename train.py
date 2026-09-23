import copy
import sys
import os
import random
import pickle
import torch.backends.cudnn
import types
import yaml
import torch
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import gc
import logging
import math
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger("Trainer")

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from models.gnn import GCNBasedTemporalGraphModel
from Project.metrics import RankingMetrics, A_hat_Calculator, compute_total_loss, compute_score_mse_loss, \
    compute_topk_bce_loss, compute_reconstruction_loss
from Project.visualization import plot_window_loss_curve_split, plot_train_val_loss_curve, plot_ndcg_curve, \
    plot_pred_true_scatter, plot_hit_rate_curve
from data.dataloader import TemporalGraphProcessor
from models.graph import SplitTemporalGraphDataset


def split_collate(batch):
    # batch: List[{"split_idx": int, "target_window_id": int, "windows": List[SimpleNamespace]}]
    return batch



class Trainer:
    def __init__(self, config, exp_id):
        self.config = config
        self.exp_id = exp_id
        self.config['results_dir'] = os.path.join(str(self.config['results_dir']), str(self.exp_id))
        self.gamma = config.get('gamma', 0.5)
        self.gnn_stride = config.get('gnn_stride', 3)
        self.pred_stride = config.get('pred_stride', 1)
        self.batch_size = config.get('batch_size', 4)
        self.gcn_hidden_dim = config.get('gcn_hidden_dim', 64)
        self.gcn_output_dim = config.get('gcn_output_dim', 32)
        self.gru_output_dim = config.get('gru_output_dim', 64)
        self.gru_num_layers = config.get('gru_num_layers', 2)
        self.ndcg_ks = config.get('ndcg_ks', [10, 20, 30])
        self.hit_rate_k = config.get('hit_rate_k', 30)
        self.is_undirected = self.config['is_undirected']

        self.pretrain_learning_rate = config.get('pretrain_learning_rate', 0.001)
        self.finetune_learning_rate = config.get('finetune_learning_rate', 0.001)

        self.training_stage = 'pretraining'

        if not torch.cuda.is_available():
            raise RuntimeError("need NVIDIA GPU (4090)，text CUDA device")

        device_id = 0
        torch.cuda.set_device(device_id)
        self.device = torch.device(f'cuda:{device_id}')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.90)

        self._processor = None
        self._model = None
        self._optimizer = None
        self._lr_scheduler = None
        self._optim_steps = 0
        self._lr_total_epochs = None

        self.a_hat_calculator = A_hat_Calculator(config, self.device)

        self.scaler = torch.amp.GradScaler('cuda', enabled=False)

        self.current_split_idx = -1
        self.best_train_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = config.get('early_stop_patience', 10)
        self.warmup_epochs = config.get('warmup_epochs', 10)
        self.min_improvement_threshold  = config.get('min_improvement_threshold ',0.1)

        self.split_representations = {}
        self.split_a_hat = {}
        self.split_pred_score_dict = {}
        self.split_node_probs_dict = {}

        self.train_split_loss_history = {}
        self.val_split_loss_history = {}
        self.test_split_loss_history = {}

        self._true_score_dict_cache = {}
        self._split_windows_cache = {}
        self._window_attributes_cache = {}
        self.train_split_indices = []
        self.val_split_indices = []
        self.test_split_indices = []
        self.all_split_data = {}

        self._load_all_datasets()

        self.param_update_history = []
        self._params_before_cache = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = TemporalGraphProcessor.load_processed_data(self.config['processor_dir'])
        return self._processor

    @property
    def model(self):
        """textloadGNNmodel"""
        if self._model is None:
            self._model = GCNBasedTemporalGraphModel(
                node_feat_dim=self.config['node_feat_dim'],
                gcn_hidden_dim=self.config.get('gcn_hidden_dim', 64),
                gcn_output_dim=self.config.get('gcn_output_dim', 32),
                gru_output_dim=self.config.get('gru_output_dim', 64),
                gru_num_layers=self.config.get('gru_num_layers', 2),
                dropout=self.config['dropout'],
                gnn_stride=self.config.get('gnn_stride', 3),
                pred_stride=self.config.get('pred_stride', 1)
            ).to(self.device)

            self.model.set_training_stage(self.training_stage)
        return self._model

    @property
    def optimizer(self):
        """texttrainingstageusetextoptimizationtextlearning rate"""
        if self._optimizer is None:
            if self.training_stage == 'pretraining':
                params_to_train = []
                for name, param in self.model.named_parameters():
                    if 'node_scoring_mlp' not in name and param.requires_grad:
                        params_to_train.append(param)

                current_lr = self.pretrain_learning_rate
            else:
                params_to_train = []
                for name, param in self.model.named_parameters():
                    if  param.requires_grad:
                        params_to_train.append(param)

                current_lr = self.finetune_learning_rate

            if not params_to_train:
                raise RuntimeError(f"{self.training_stage}stagemissingneedtrainingtextparameter")

            self._optimizer = optim.AdamW(
                params_to_train,
                lr=current_lr,
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=True
            )
        return self._optimizer

    @property
    def lr_scheduler(self):
        """text（text epoch text；text，text）"""
        if self._lr_scheduler is None:
            total_epochs = self._lr_total_epochs

            def lr_lambda(epoch):
                e = min(max(int(epoch), 0), total_epochs - 1)
                progress = e / float(total_epochs - 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self._lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda, last_epoch=-1)
        return self._lr_scheduler

    def set_training_stage(self, stage: str):
        """settrainingstage"""
        assert stage in ['pretraining', 'finetuning'], f"texttrainingstage: {stage}"
        self.training_stage = stage

        if self._model is not None:
            self.model.set_training_stage(stage)

        self._optimizer = None
        self._lr_scheduler = None
        self._lr_total_epochs = None

    def setup_experiment(self):
        """setexperimenttext"""
        logger.info("experimenttextsetcompleted")

    def load_ranking_file(self, split_index, window_id):
        """
        loadrankingfile
        input: splittype, splittext, windowID
        return: rankingdatatext {nodeID: text}
        """
        try:
            base_dir = "./data/background_rankings"
            base_name = f"split_{split_index}_window_{window_id}_ranks"
            parquet_path = os.path.join(base_dir, base_name + ".parquet")

            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"rankingfiletext: {parquet_path}")

            df = pd.read_parquet(parquet_path)

            node_ids = df["nodes"].astype(str).str.strip().values
            scores = df["scores"].astype(float).values
            ranking_dict = dict(zip(node_ids, scores))

            if not ranking_dict:
                raise ValueError(f"rankingfile {parquet_path} is empty")

            return ranking_dict

        except Exception as e:
            raise RuntimeError(f"loadrankingfiletext: {e}")

    def _load_all_datasets(self):
        """loadalldatasettext"""
        logger.info("textloadalldataset...")

        window_attributes_path = os.path.join(self.config['processor_dir'], "Email_window_attributes.pkl")
        if os.path.exists(window_attributes_path):
            with open(window_attributes_path, 'rb') as f:
                window_data = pickle.load(f)
                split_info = window_data['split_info']
            logger.info(f"successloadwindowtextfile: {window_attributes_path}")
        else:
            raise RuntimeError(f"windowtextfiletext: {window_attributes_path}")

        available_splits = self.processor.get_all_splits()

        for split_idx in range(len(available_splits)):
            try:
                split_windows = self.processor.get_split_windows(split_idx)

                if len(split_windows) == 0:
                    raise RuntimeError(f"len(split_windows) == 0")

                target_window_id = int(split_windows[-1])

                if split_idx in split_info:
                    split_type = split_info[split_idx]['type']
                else:
                    raise RuntimeError(f"split_idx text split_infotext")

                if split_type is None:
                    raise RuntimeError(f"textsplit {split_idx} texttype")

                self.all_split_data[split_idx] = {
                    'gnn_loader': None,
                    'target_window_id': target_window_id,
                    'windows': split_windows,
                    'split_type': split_type
                }

                if split_type == "train":
                    self.train_split_indices.append(split_idx)
                elif split_type == "val":
                    self.val_split_indices.append(split_idx)
                elif split_type == "test":
                    self.test_split_indices.append(split_idx)

            except Exception as e:
                raise RuntimeError(f"loadsplit {split_idx} text: {str(e)}")

        logger.info(f"dataloadcompleted - trainingsplit: {len(self.train_split_indices)}, "
                    f"validationsplit: {len(self.val_split_indices)}, "
                    f"testsplit: {len(self.test_split_indices)}")

    def get_split_batch_loader(self, split_indices, split_type: str):

        dataset = SplitTemporalGraphDataset(
            processor=self.processor,
            config=self.config,
            split_indices=split_indices,
            split_type=split_type
        )

        num_workers = int(self.config.get('num_workers', 20))
        prefetch_factor = self.config.get('prefetch_factor', 4)
        persistent_workers = False

        def worker_init_fn(worker_id):
            worker_seed = self.config.get('seed', 42)
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.config.get('seed', 42))

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=split_collate,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=str(self.device),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        return loader

    def pretrain_epoch(self):
        """Pretrainstage"""
        self.set_training_stage('pretraining')

        pretrain_dir = os.path.join(self.config['results_dir'], "Pretraining")
        os.makedirs(pretrain_dir, exist_ok=True)

        self.lr_record = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        best_model_state = None
        best_train_loss = float('inf')
        best_epoch = -1

        self.best_ndcg_30 = 0.0

        self.epoch_train_losses = []

        self.epoch_val_ndcgs_10 = []
        self.epoch_val_ndcgs_20 = []
        self.epoch_val_ndcgs_30 = []
        self.epoch_val_hit_rates = []
        self.epoch_val_aucs = []
        self.epoch_val_f1s = []
        self.epoch_val_mrrs = []
        self.epoch_val_losses = []

        self.epoch_test_ndcgs_10 = []
        self.epoch_test_ndcgs_20 = []
        self.epoch_test_ndcgs_30 = []
        self.epoch_test_hit_rates = []
        self.epoch_test_aucs = []
        self.epoch_test_f1s = []
        self.epoch_test_mrrs = []
        self.epoch_test_losses = []

        train_split_indices = self.train_split_indices
        if not train_split_indices:
            raise RuntimeError("missingtexttrainingsplit")

        val_split_indices = self.val_split_indices
        if not val_split_indices:
            raise RuntimeError("missingtextvalidationsplit")

        train_true_scores_cache = {}
        for split_idx in train_split_indices:
            split_data = self.all_split_data[split_idx]
            target_window_id = split_data['target_window_id']
            train_true_scores_cache[split_idx] = self.load_ranking_file(split_idx, target_window_id)

        pretrain_epochs = 60
        self._lr_total_epochs = pretrain_epochs
        self._lr_scheduler = None

        for epoch in range(pretrain_epochs):
            logger.info(f"============================== Pretrainstage Epoch {epoch + 1}/{pretrain_epochs} ====")

            torch.cuda.empty_cache()
            gc.collect()
            self.model.train()

            self._record_params_before_update()

            self.optimizer.zero_grad(set_to_none=True)

            self._forward_propagation_for_split(train_split_indices, mode="train")

            total_bce_loss = []
            total_splits = 0

            bce_loss_sum = None

            for split_idx in train_split_indices:
                try:
                    true_score_dict = train_true_scores_cache[split_idx]

                    split_metrics = self.train_for_split_pretraining(split_idx, true_score_dict)

                    total_bce_loss.append(split_metrics['bce_loss_value'])
                    total_splits += 1

                    if bce_loss_sum is None:
                        bce_loss_sum = split_metrics['bce_loss_tensor']
                    else:
                        bce_loss_sum = bce_loss_sum + split_metrics['bce_loss_tensor']

                except Exception as e:
                    raise RuntimeError(f"split {split_idx} Pretraintext: {str(e)}")

            if total_splits == 0:
                raise RuntimeError(f"Epoch {epoch + 1} missingsuccesstextsplittraining")

            avg_bce_loss = bce_loss_sum / float(total_splits)
            self.scaler.scale(avg_bce_loss).backward()

            logger.info(
                f"Pretrain Epoch {epoch + 1} BCEloss: {avg_bce_loss:.4f}, "
            )

            has_gradients = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-10:
                        has_gradients = True
                        logger.debug(f"parameter {name} text，text: {grad_norm:.4e}")

            if not has_gradients:
                logger.warning(f"Pretrain Epoch {epoch + 1}: warning！missingtext")

            for name, p in self.model.named_parameters():
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    nan_ct = torch.isnan(p.grad).sum().item()
                    inf_ct = torch.isinf(p.grad).sum().item()
                    raise RuntimeError(f"text: param={name}, nan={nan_ct}, inf={inf_ct}")

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._optim_steps += 1

            self._check_param_updates(epoch + 1, 'pretraining')

            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_record[epoch + 1] = float(current_lr)
            logger.info(f"=======Pretrain Epoch {epoch + 1} learning rate:{current_lr}=====")

            self._save_lr_record(self.lr_record, pretrain_dir)

            train_metrics = {
                'bce_loss': float(avg_bce_loss.detach().cpu()),
                'learning_rate': current_lr
            }

            self.epoch_train_losses.append(train_metrics['bce_loss'])

            if train_metrics['bce_loss'] < best_train_loss:
                best_train_loss = train_metrics['bce_loss']
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                logger.info(f"Pretrain Epoch {epoch + 1}: texttrainingbce_losstext {best_train_loss:.4f}")

            val_metrics = self.validate(val_split_indices, mode='val')
            test_metrics = self.test(split_type="test")

            avg_val_loss = val_metrics['total_loss']

            self.epoch_val_losses.append(avg_val_loss)
            self.epoch_test_losses.append(test_metrics['total_loss'])

            self.epoch_val_ndcgs_10.append(val_metrics[f'ndcg@10'])
            self.epoch_val_ndcgs_20.append(val_metrics[f'ndcg@20'])
            self.epoch_val_ndcgs_30.append(val_metrics[f'ndcg@30'])
            self.epoch_val_hit_rates.append(val_metrics[f'hit_rate@{self.hit_rate_k}'])
            self.epoch_val_aucs.append(val_metrics['auc'])
            self.epoch_val_f1s.append(val_metrics['f1'])
            self.epoch_val_mrrs.append(val_metrics['mrr'])

            self.epoch_test_ndcgs_10.append(test_metrics['ndcg@10'])
            self.epoch_test_ndcgs_20.append(test_metrics['ndcg@20'])
            self.epoch_test_ndcgs_30.append(test_metrics['ndcg@30'])
            self.epoch_test_hit_rates.append(test_metrics[f'hit_rate@{self.hit_rate_k}'])
            self.epoch_test_aucs.append(test_metrics['auc'])
            self.epoch_test_f1s.append(test_metrics['f1'])
            self.epoch_test_mrrs.append(test_metrics['mrr'])

            ndcg_hit_rate_dir = os.path.join(pretrain_dir, "ndcg_hit_rate")
            os.makedirs(ndcg_hit_rate_dir, exist_ok=True)

            for k in self.ndcg_ks:
                val_ndcg_history = getattr(self, f'epoch_val_ndcgs_{k}')
                test_ndcg_history = getattr(self, f'epoch_test_ndcgs_{k}')
                plot_ndcg_curve(
                    val_ndcg_history=val_ndcg_history,
                    test_ndcg_history=test_ndcg_history,
                    exp_id=self.exp_id,
                    save_dir=ndcg_hit_rate_dir,
                    k=k,
                    gamma=self.gamma,
                    stage='pretraining'
                )

            plot_hit_rate_curve(
                val_hit_rate_history=self.epoch_val_hit_rates,
                test_hit_rate_history=self.epoch_test_hit_rates,
                exp_id=self.exp_id,
                save_dir=ndcg_hit_rate_dir,
                k=self.hit_rate_k,
                gamma=self.gamma,
                stage='pretraining'
            )

            logger.info(
                f"Pretrain Epoch {epoch + 1} validationmetric - "
                f"NDCG@10: {val_metrics['ndcg@10']:.4f}, "
                f"NDCG@20: {val_metrics['ndcg@20']:.4f}, "
                f"NDCG@30: {val_metrics['ndcg@30']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}"
                f"MRR: {val_metrics['mrr']:.4f}"
            )

            loss_dir = os.path.join(pretrain_dir, "loss")
            os.makedirs(loss_dir, exist_ok=True)
            plot_train_val_loss_curve(
                train_loss_history=self.epoch_train_losses,
                val_loss_history=self.epoch_val_losses,
                test_loss_history=self.epoch_test_losses,
                exp_id=self.exp_id,
                save_dir=loss_dir,
                gamma=self.gamma,
                stage='pretraining'
            )

            training_loss_dir = os.path.join(loss_dir, "training_loss")
            os.makedirs(training_loss_dir, exist_ok=True)
            for split_idx in train_split_indices:
                if hasattr(self, 'train_split_loss_history') and split_idx in self.train_split_loss_history:
                    if 'bce_loss' in self.train_split_loss_history[split_idx]:
                        window_losses = {'bce_loss': self.train_split_loss_history[split_idx]['bce_loss']}
                        plot_window_loss_curve_split(
                            window_losses=window_losses,
                            split_idx=split_idx,
                            exp_id=self.exp_id,
                            save_dir=training_loss_dir,
                            gamma=self.gamma,
                            stage='pretraining'
                        )
                    else:
                        raise RuntimeError(f"pretraining-split {split_idx} texttrain_losstextmissing'bce_loss'text")

            val_loss_dir = os.path.join(loss_dir, "val_loss")
            os.makedirs(val_loss_dir, exist_ok=True)
            for split_idx in val_split_indices:
                if hasattr(self, 'val_split_loss_history') and split_idx in self.val_split_loss_history:
                    window_losses = {'bce_loss': self.val_split_loss_history[split_idx]['bce_loss']}
                    plot_window_loss_curve_split(
                        window_losses=window_losses,
                        split_idx=split_idx,
                        exp_id=self.exp_id,
                        save_dir=val_loss_dir,
                        gamma=self.gamma,
                        stage='pretraining'
                    )
                else:
                    raise RuntimeError(f"pretraining-split {split_idx} textval_losstextmissing'bce_loss'text")

            test_loss_dir = os.path.join(loss_dir, "test_loss")
            os.makedirs(test_loss_dir, exist_ok=True)
            for split_idx in self.test_split_indices:
                if hasattr(self, 'test_split_loss_history') and split_idx in self.test_split_loss_history:
                    window_losses = {'bce_loss': self.test_split_loss_history[split_idx]['bce_loss']}
                    plot_window_loss_curve_split(
                        window_losses=window_losses,
                        split_idx=split_idx,
                        exp_id=self.exp_id,
                        save_dir=test_loss_dir,
                        gamma=self.gamma,
                        stage='pretraining'
                    )
                else:
                    raise RuntimeError(f"pretraining-split {split_idx} texttest_losstextmissing'bce_loss'text")

            # self._plot_roc_curves(val_split_indices, self.test_split_indices, pretrain_dir, "Pretraining")
            self._clear_forward_caches(train_split_indices)
            torch.cuda.empty_cache()
            gc.collect()

        pretrain_best_model_path = os.path.join(str(pretrain_dir), "best_pretrain_model.pth")
        torch.save(best_model_state, pretrain_best_model_path)
        logger.info(f"savetextPretrainmodeltext: {pretrain_best_model_path}")

        logger.info("=== usetextPretrainparametertext ===")
        self.model.load_state_dict(best_model_state)

        final_val_metrics = self.validate(val_split_indices, mode='val')
        final_test_metrics = self.test(split_type="test")

        self._generate_ranking_summary(
            final_val_metrics, final_test_metrics,
            pretrain_dir, "pretrain_ranking_summary.yaml",
            stage_name="Pretrain"
        )

        return best_model_state, best_train_loss

    def finetune_epoch(self, pretrained_model_state):
        """Fine_tuningstage"""
        self.set_training_stage('finetuning')

        finetune_dir = os.path.join(self.config['results_dir'], "Finetuning")
        os.makedirs(finetune_dir, exist_ok=True)

        self.lr_record = {}

        if pretrained_model_state is not None:
            self.model.load_state_dict(pretrained_model_state)
            logger.info("textloadPretrainmodelparameter")

        self.train_split_loss_history = {}
        self.val_split_loss_history = {}
        self.test_split_loss_history = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        best_model_state = None
        best_train_loss = float('inf')
        best_epoch = -1

        self.early_stop_counter = 0
        best_val_ndcg_30 = 0.0

        self.epoch_train_losses = []

        self.epoch_val_losses = []
        self.epoch_val_ndcgs_10 = []
        self.epoch_val_ndcgs_20 = []
        self.epoch_val_ndcgs_30 = []
        self.epoch_val_hit_rates = []
        self.epoch_val_aucs = []
        self.epoch_val_f1s = []
        self.epoch_val_mrrs = []

        self.epoch_test_losses = []
        self.epoch_test_ndcgs_10 = []
        self.epoch_test_ndcgs_20 = []
        self.epoch_test_ndcgs_30 = []
        self.epoch_test_hit_rates = []
        self.epoch_test_aucs = []
        self.epoch_test_f1s = []
        self.epoch_test_mrrs = []

        train_split_indices = self.train_split_indices
        if not train_split_indices:
            raise RuntimeError("missingtexttrainingsplit")

        val_split_indices = self.val_split_indices
        if not val_split_indices:
            raise RuntimeError("missingtextvalidationsplit")

        train_true_scores_cache = {}
        for split_idx in train_split_indices:
            split_data = self.all_split_data[split_idx]
            target_window_id = split_data['target_window_id']
            train_true_scores_cache[split_idx] = self.load_ranking_file(split_idx, target_window_id)

        finetune_epochs = 100
        self._lr_total_epochs = finetune_epochs
        self._lr_scheduler = None

        warmup_epochs = self.warmup_epochs
        patience = self.early_stop_patience
        min_improvement_threshold = self.min_improvement_threshold

        for epoch in range(finetune_epochs):
            logger.info(f"========================== Fine_tuningstage Epoch {epoch + 1}/{finetune_epochs} ====")

            torch.cuda.empty_cache()
            gc.collect()
            self.model.train()

            self._record_params_before_update()

            self.optimizer.zero_grad(set_to_none=True)

            self._forward_propagation_for_split(train_split_indices, mode="train")

            total_epoch_loss = []
            total_mse_loss = []
            total_recon_loss = []
            total_splits = 0

            total_loss_sum = None

            for split_idx in train_split_indices:
                try:
                    true_score_dict = train_true_scores_cache[split_idx]

                    split_metrics = self.train_for_split_finetuning(split_idx, true_score_dict)

                    total_epoch_loss.append(split_metrics['total_loss_value'])
                    total_mse_loss.append(split_metrics['mse_loss_value'])
                    total_recon_loss.append(split_metrics['recon_loss_value'])

                    total_splits += 1

                    if total_loss_sum is None:
                        total_loss_sum = split_metrics['total_loss_tensor']
                    else:
                        total_loss_sum = total_loss_sum + split_metrics['total_loss_tensor']

                except Exception as e:
                    raise RuntimeError(f"split {split_idx} Fine_tuningtext: {str(e)}")

            if total_splits == 0:
                raise RuntimeError(f"Epoch {epoch + 1} missingsuccesstextsplittraining")

            mean_total_loss = total_loss_sum / float(total_splits)
            self.scaler.scale(mean_total_loss).backward()

            avg_total_loss = sum(total_epoch_loss) / total_splits
            avg_mse_loss = sum(total_mse_loss) / total_splits
            avg_recon_loss = sum(total_recon_loss) / total_splits

            logger.info(
                f"Finetune Epoch {epoch + 1} trainingaverageloss - "
                f"textloss: {avg_total_loss:.4f}, "
                f"mseloss: {avg_mse_loss:.4f}, "
                f"textloss: {avg_recon_loss:.4f}"
            )

            has_gradients = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-10:
                        has_gradients = True
                        logger.debug(f"parameter {name} text，text: {grad_norm:.4e}")

            if not has_gradients:
                logger.warning(f"Fine_tuning Epoch {epoch + 1}: warning！missingtext")

            for name, p in self.model.named_parameters():
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    nan_ct = torch.isnan(p.grad).sum().item()
                    inf_ct = torch.isinf(p.grad).sum().item()
                    raise RuntimeError(f"text: param={name}, nan={nan_ct}, inf={inf_ct}")

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._optim_steps += 1

            self._check_param_updates(epoch + 1, 'finetuning')

            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_record[epoch + 1] = float(current_lr)
            logger.info(f"==================Fine_tuning Epoch {epoch + 1} learning rate:{current_lr}=====")

            self._save_lr_record(self.lr_record, finetune_dir)
            train_metrics = {
                'total_loss': avg_total_loss,
                'mse_loss': avg_mse_loss,
                'recon_loss': avg_recon_loss,
                'learning_rate': current_lr
            }
            self.epoch_train_losses.append(train_metrics['total_loss'])

            if epoch + 1 > warmup_epochs:
                if train_metrics['mse_loss'] < best_train_loss:
                    best_train_loss = train_metrics['mse_loss']
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    logger.info(f"Fine_tuning Epoch {epoch + 1} texttrainingMSElosstext: {best_train_loss:.4f}")

            val_metrics = self.validate(val_split_indices, mode='val')
            test_metrics = self.test(split_type="test")

            avg_val_loss = val_metrics['total_loss']

            current_val_ndcg_30 = val_metrics['ndcg@30']
            if epoch + 1 > warmup_epochs:
                if epoch + 1 == warmup_epochs + 1:
                    best_val_ndcg_30 = current_val_ndcg_30
                    self.early_stop_counter = 0
                    logger.info(
                        f"Fine_tuning Epoch {epoch + 1}: textstage，textNDCG@30: {best_val_ndcg_30:.4f}，textearly stoppingtext"
                    )

                else:
                    improvement = current_val_ndcg_30 - best_val_ndcg_30

                    if improvement > min_improvement_threshold:
                        best_val_ndcg_30 = current_val_ndcg_30
                        self.early_stop_counter = 0
                        logger.info(
                            f"Fine_tuning Epoch {epoch + 1}: text {improvement:.4f}，"
                            f"text: {best_val_ndcg_30:.4f}，textearly stoppingtext"
                        )
                    else:
                        self.early_stop_counter += 1

                        if self.early_stop_counter >= patience:
                            logger.info(
                                f"Fine_tuning Epoch {epoch + 1}: textearly stopping！text {patience} textepochtextNDCG@30text {min_improvement_threshold}"
                            )
                            break

            self.epoch_val_losses.append(avg_val_loss)

            self.epoch_test_losses.append(test_metrics['total_loss'])

            self.epoch_val_ndcgs_10.append(val_metrics[f'ndcg@10'])
            self.epoch_val_ndcgs_20.append(val_metrics[f'ndcg@20'])
            self.epoch_val_ndcgs_30.append(val_metrics[f'ndcg@30'])
            self.epoch_val_hit_rates.append(val_metrics[f'hit_rate@{self.hit_rate_k}'])
            self.epoch_val_aucs.append(val_metrics['auc'])
            self.epoch_val_f1s.append(val_metrics['f1'])
            self.epoch_val_mrrs.append(val_metrics['mrr'])

            self.epoch_test_ndcgs_10.append(test_metrics['ndcg@10'])
            self.epoch_test_ndcgs_20.append(test_metrics['ndcg@20'])
            self.epoch_test_ndcgs_30.append(test_metrics['ndcg@30'])
            self.epoch_test_hit_rates.append(test_metrics[f'hit_rate@{self.hit_rate_k}'])
            self.epoch_test_aucs.append(test_metrics['auc'])
            self.epoch_test_f1s.append(test_metrics['f1'])
            self.epoch_test_mrrs.append(test_metrics['mrr'])

            ndcg_hit_rate_dir = os.path.join(finetune_dir, "ndcg_hit_rate")
            os.makedirs(ndcg_hit_rate_dir, exist_ok=True)

            for k in self.ndcg_ks:
                val_ndcg_history = getattr(self, f'epoch_val_ndcgs_{k}')
                test_ndcg_history = getattr(self, f'epoch_test_ndcgs_{k}')
                plot_ndcg_curve(
                    val_ndcg_history=val_ndcg_history,
                    test_ndcg_history=test_ndcg_history,
                    exp_id=self.exp_id,
                    save_dir=ndcg_hit_rate_dir,
                    k=k,
                    gamma=self.gamma,
                    stage='finetuning'
                )

            plot_hit_rate_curve(
                val_hit_rate_history=self.epoch_val_hit_rates,
                test_hit_rate_history=self.epoch_test_hit_rates,
                exp_id=self.exp_id,
                save_dir=ndcg_hit_rate_dir,
                k=self.hit_rate_k,
                gamma=self.gamma,
                stage='finetuning'
            )

            loss_dir = os.path.join(finetune_dir, "loss")
            os.makedirs(loss_dir, exist_ok=True)
            plot_train_val_loss_curve(
                train_loss_history=self.epoch_train_losses,
                val_loss_history=self.epoch_val_losses,
                test_loss_history=self.epoch_test_losses,
                exp_id=self.exp_id,
                save_dir=loss_dir,
                gamma=self.gamma,
                stage='finetuning'
            )

            training_loss_dir = os.path.join(loss_dir, "training_loss")
            os.makedirs(training_loss_dir, exist_ok=True)
            for split_idx in train_split_indices:
                if hasattr(self, 'train_split_loss_history') and split_idx in self.train_split_loss_history:
                    window_losses = {
                        'total_loss': self.train_split_loss_history[split_idx]['total_loss'],
                        'mse_loss': self.train_split_loss_history[split_idx]['mse_loss'],
                        'recon_loss': self.train_split_loss_history[split_idx]['recon_loss'],
                    }
                    plot_window_loss_curve_split(
                        window_losses=window_losses,
                        split_idx=split_idx,
                        exp_id=self.exp_id,
                        save_dir=training_loss_dir,
                        gamma=self.gamma,
                        stage='finetuning'
                    )

            val_loss_dir = os.path.join(loss_dir, "val_loss")
            os.makedirs(val_loss_dir, exist_ok=True)
            for split_idx in val_split_indices:
                if hasattr(self, 'val_split_loss_history') and split_idx in self.val_split_loss_history:
                    window_losses = {
                        'total_loss': self.val_split_loss_history[split_idx]['total_loss'],
                        'mse_loss': self.val_split_loss_history[split_idx]['mse_loss'],
                        'recon_loss': self.val_split_loss_history[split_idx]['recon_loss'],
                    }
                    plot_window_loss_curve_split(
                        window_losses=window_losses,
                        split_idx=split_idx,
                        exp_id=self.exp_id,
                        save_dir=val_loss_dir,
                        gamma=self.gamma,
                        stage='finetuning'
                    )
                else:
                    raise RuntimeError(f"Pretrain-split {split_idx} textval_losstextmissing'mse_loss'text")

            test_loss_dir = os.path.join(loss_dir, "test_loss")
            os.makedirs(test_loss_dir, exist_ok=True)
            for split_idx in self.test_split_indices:
                if hasattr(self, 'test_split_loss_history') and split_idx in self.test_split_loss_history:
                    window_losses = {
                        'total_loss': self.test_split_loss_history[split_idx]['total_loss'],
                        'mse_loss': self.test_split_loss_history[split_idx]['mse_loss'],
                        'recon_loss': self.test_split_loss_history[split_idx]['recon_loss'],
                    }
                    plot_window_loss_curve_split(
                        window_losses=window_losses,
                        split_idx=split_idx,
                        exp_id=self.exp_id,
                        save_dir=test_loss_dir,
                        gamma=self.gamma,
                        stage='finetuning'
                    )
                else:
                    raise RuntimeError(f"Pretrain-split {split_idx} texttest_losstextmissing'mse_loss'text")

            # self._plot_roc_curves(val_split_indices, self.test_split_indices,finetune_dir, "Finetuning")

            self._clear_forward_caches(train_split_indices)
            torch.cuda.empty_cache()
            gc.collect()

        finetune_best_model_path = os.path.join(str(finetune_dir), "best_finetune_model.pth")
        if best_model_state is not None:
            torch.save(best_model_state, finetune_best_model_path)
            logger.info(f"savetextFine_tuningmodeltext: {finetune_best_model_path}")

        logger.info("=== usetextFine_tuningparametertext ===")
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        logger.info("=== saveFinetuningstagetext ===")
        self._save_score_file_and_scatter_plot()

        final_val_metrics = self.validate(val_split_indices, mode='val')
        final_test_metrics = self.test(split_type="test")

        self._generate_ranking_summary(
            final_val_metrics, final_test_metrics,
            finetune_dir, "finetune_ranking_summary.yaml",
            stage_name="Fine_tuning"
        )

        return best_model_state, best_train_loss

    def train_epoch(self, split_type="train"):
        """texttrainingtext，containsPretraintextFine_tuningtextstage"""
        try:
            logger.info("=== textPretrainstage ===")

            pretrained_model_state, pretrain_best_loss = self.pretrain_epoch()

            pretrain_final_test_metrics = self.test(split_type="test")
            pretrain_test_ndcg_10 = pretrain_final_test_metrics['ndcg@10']
            pretrain_test_ndcg_20 = pretrain_final_test_metrics['ndcg@20']
            pretrain_test_ndcg_30 = pretrain_final_test_metrics['ndcg@30']
            pretrain_test_hit_rate_30 = pretrain_final_test_metrics[f'hit_rate@{self.hit_rate_k}']
            pretrain_test_auc = pretrain_final_test_metrics['auc']
            pretrain_test_f1 = pretrain_final_test_metrics['f1']
            pretrain_test_mrr = pretrain_final_test_metrics['mrr']

            logger.info("=== textFine_tuningstage ===")

            best_model_state, finetune_best_loss = self.finetune_epoch(pretrained_model_state)

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            logger.info("=== textvalidationtextteststage ===")
            val_metrics = self.validate(self.val_split_indices, mode='val')

            finetune_test_metrics = self.test(split_type="test")

            return (
                pretrain_best_loss,
                finetune_best_loss,
                val_metrics['ndcg@10'],
                val_metrics['ndcg@20'],
                val_metrics['ndcg@30'],
                pretrain_test_ndcg_10,
                pretrain_test_ndcg_20,
                pretrain_test_ndcg_30,
                pretrain_test_hit_rate_30,
                pretrain_test_auc,
                pretrain_test_f1,
                pretrain_test_mrr,
                finetune_test_metrics['ndcg@10'],
                finetune_test_metrics['ndcg@20'],
                finetune_test_metrics['ndcg@30'],
                finetune_test_metrics[f'hit_rate@{self.hit_rate_k}'],
                finetune_test_metrics['auc'],
                finetune_test_metrics['f1'],
                finetune_test_metrics['mrr'],
                best_model_state
            )

        except Exception as e:
            raise RuntimeError(f"textstagetrainingtext: {str(e)}")

    def _forward_propagation_for_split(self, split_indices, mode: str):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode text 'train'/'val'/'test'，actualtext: {mode}")

        logger.info(f"text - text: {mode}")

        split_type = "train" if mode == "train" else mode
        split_loader = self.get_split_batch_loader(split_indices, split_type=split_type)

        if mode == "train":
            self.model.train()
            grad_ctx = torch.enable_grad()
        else:
            self.model.eval()
            grad_ctx = torch.no_grad()

        with grad_ctx:
            for batch in split_loader:
                # batch: List[{"split_idx": int, "target_window_id": int, "windows": List[SimpleNamespace]}]
                batch_split_indices = [int(item["split_idx"]) for item in batch]
                batch_target_window_ids = [int(item["target_window_id"]) for item in batch]

                batch_windows = []
                for item in batch:
                    win_list = []
                    for w in item["windows"]:
                        win_list.append(types.SimpleNamespace(
                            node_features=w.node_features.to(self.device, non_blocking=True, dtype=torch.float32),
                            edge_index=w.edge_index.to(self.device, non_blocking=True),
                            window_id=int(w.window_id.item()),
                            window_info=w.window_info,
                            adj_matrix=w.adj_matrix.to(self.device, non_blocking=True),
                            laplacian_matrix=w.laplacian_matrix.to(self.device, non_blocking=True)
                        ))
                    batch_windows.append(win_list)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                    next_repr_list, a_hat_list, pred_score_list, node_probs_list = self.model(
                        batch_windows=batch_windows,
                        split_indices=batch_split_indices,
                        target_window_ids=batch_target_window_ids
                    )

                for i, split_idx in enumerate(batch_split_indices):
                    self.split_representations[split_idx] = next_repr_list[i] if mode == "train" else next_repr_list[
                        i].detach()
                    self.split_a_hat[split_idx] = a_hat_list[i] if mode == "train" else a_hat_list[i].detach()

                    if self.training_stage == 'pretraining':
                        self.split_node_probs_dict[split_idx] = node_probs_list[i]
                        self.split_pred_score_dict[split_idx] = None
                    else:
                        self.split_pred_score_dict[split_idx] = pred_score_list[i]
                        self.split_node_probs_dict[split_idx] = None

                    last_gnn_window_id = int(batch_windows[i][self.gnn_stride - 1].window_id)
                    key = (int(split_idx), int(last_gnn_window_id))
                    if (not hasattr(self.model, "output_global_nodes_by_window_key")) or (
                            key not in self.model.output_global_nodes_by_window_key):
                        raise RuntimeError(
                            f"text split={split_idx}, window={last_gnn_window_id} textoutputtext")

                del batch_windows
                torch.cuda.empty_cache()

    def train_for_split_pretraining(self, split_idx, true_score_dict):
        """Pretrainstage：textdatasplittraining """
        self.model.train()

        if split_idx not in self.split_representations:
            raise RuntimeError(f"split {split_idx} textresultstext")

        with torch.amp.autocast(device_type='cuda', enabled=False):
            node_probs_dict = self.split_node_probs_dict[split_idx]

        try:
            bce_loss = compute_topk_bce_loss(
                true_score_dict=true_score_dict,
                pred_prob_dict=node_probs_dict,
                top_k=self.hit_rate_k,
                device=self.device,
                split_idx=split_idx
            )

            if not torch.isfinite(bce_loss):
                raise ValueError(f"split {split_idx}: BCElosstext: {bce_loss.item():.4e}")

        except Exception as e:
            raise RuntimeError(f"calculatesplit {split_idx} textBCElosstext: {str(e)}")

        if not torch.isfinite(bce_loss):
            raise RuntimeError(
                f"textloss: bce={float(bce_loss)}")

        bce_loss_value = float(bce_loss.detach().cpu())

        if not hasattr(self, 'train_split_loss_history'):
            self.train_split_loss_history = {}
        if split_idx not in self.train_split_loss_history:
            self.train_split_loss_history[split_idx] = {
                'bce_loss': []}

        self.train_split_loss_history[split_idx]['bce_loss'].append(bce_loss_value)

        return {
            'bce_loss_tensor': bce_loss,
            'bce_loss_value': bce_loss_value,
        }

    def train_for_split_finetuning(self, split_idx, true_score_dict):
        """Fine_tuningstage：textdatasplittraining """
        self.model.train()

        if split_idx not in self.split_representations:
            raise RuntimeError(f"split {split_idx} textresultstext")

        with torch.amp.autocast(device_type='cuda', enabled=False):
            pred_score_dict = self.split_pred_score_dict[split_idx]
            a_hat = self.split_a_hat[split_idx]

        try:
            mse_loss = compute_score_mse_loss(
                true_score_dict=true_score_dict,
                pred_score_dict=pred_score_dict,
                top_k=self.hit_rate_k,
                device=self.device,
                split_idx=split_idx
            )

            if not torch.isfinite(mse_loss):
                raise RuntimeError(f"MSElosstext: {mse_loss.item()}")

            if mse_loss.grad_fn is None:
                logger.warning(f"split {split_idx}: MSElosstextmodelparameter")

        except Exception as e:
            raise RuntimeError(f"calculate mse losstext: {str(e)}")

        try:
            split_windows = self.processor.get_split_windows(split_idx)
            if len(split_windows) < (self.gnn_stride + 1):
                raise RuntimeError(f"split {split_idx}: windowtextcontainstextwindow")

            target_window_id = int(split_windows[-1])
            last_gnn_window_id = int(split_windows[self.gnn_stride - 1])

            key = (int(split_idx), int(last_gnn_window_id))
            if (not hasattr(self.model, "output_global_nodes_by_window_key")) or (
                    key not in self.model.output_global_nodes_by_window_key):
                raise RuntimeError(f"split {split_idx}: textwindow {last_gnn_window_id} textoutputtextnodetext")
            model_global_nodes = self.model.output_global_nodes_by_window_key[key]

            a_hat_sub, A_sub = self.a_hat_calculator.build_recon_mats_intersection(
                processor=self.processor,
                split_idx=split_idx,
                target_window_id=target_window_id,
                model_global_nodes=model_global_nodes,
                a_hat=a_hat
            )

            if a_hat_sub is None or A_sub is None:
                recon_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            else:
                recon_loss = compute_reconstruction_loss(a_hat_sub, A_sub)


        except Exception as e:
            raise RuntimeError(f"calculatesplit {split_idx} textlosstext: {str(e)}")

        total_loss = compute_total_loss(mse_loss, recon_loss, gamma=self.gamma)

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"textloss: total={float(total_loss)}, recon={float(recon_loss)}, mse={float(mse_loss)}")

        total_loss_value = float(total_loss.detach().cpu())
        total_mse_loss_value = float(mse_loss.detach().cpu())
        total_recon_loss_value = float(recon_loss.detach().cpu())

        if not hasattr(self, 'train_split_loss_history'):
            self.train_split_loss_history = {}

        if split_idx not in self.train_split_loss_history:
            self.train_split_loss_history[split_idx] = {
                'total_loss': [],
                'mse_loss': [],
                'recon_loss': []
            }
        elif 'mse_loss' not in self.train_split_loss_history[split_idx]:
            raise RuntimeError(f"missing'mse_loss'text")

        self.train_split_loss_history[split_idx]['total_loss'].append(total_loss_value)
        self.train_split_loss_history[split_idx]['mse_loss'].append(total_mse_loss_value)
        self.train_split_loss_history[split_idx]['recon_loss'].append(total_recon_loss_value)

        return {
            'total_loss_tensor': total_loss,
            'total_loss_value': total_loss_value,
            'mse_loss_value': total_mse_loss_value,
            'recon_loss_value': total_recon_loss_value
        }

    def validate(self, split_indices, mode: str = 'val'):
        self.model.eval()
        split_indices = [int(x) for x in split_indices]
        if len(split_indices) == 0:
            raise RuntimeError("validate textsplittext")

        self._forward_propagation_for_split(split_indices, mode=mode)

        all_ndcgs_10 = []
        all_ndcgs_20 = []
        all_ndcgs_30 = []
        all_hit_rates = []
        all_aucs = []
        all_f1s = []
        all_mrrs = []
        total_val_loss = 0.0
        total_windows = 0

        split_data_list = []
        gnn_stride = int(self.config.get('gnn_stride', 3))
        for split_idx in split_indices:
            split_windows = self.processor.get_split_windows(split_idx)
            if len(split_windows) < gnn_stride:
                raise RuntimeError(
                    f"validate: split {split_idx} windowtext: {len(split_windows)} < gnn_stride({gnn_stride})")

            target_window_id = int(split_windows[-1])
            last_gnn_window = int(split_windows[gnn_stride - 1])

            split_data_list.append({
                'split_idx': split_idx,
                'target_window_id': target_window_id,
                'last_gnn_window': last_gnn_window,
                'gnn_stride': gnn_stride
            })

        for split_data in split_data_list:
            split_idx = split_data['split_idx']
            target_window_id = split_data['target_window_id']

            if split_idx not in self.split_representations:
                raise RuntimeError(f"split {split_idx} textresultstext")

            a_hat = self.split_a_hat[split_idx]

            try:
                true_score_dict = self.load_ranking_file(split_idx, target_window_id)
            except Exception as e:
                raise RuntimeError(f"loadground truthrankingtext: {str(e)}")

            if self.training_stage == 'pretraining':
                node_probs_dict = self.split_node_probs_dict[split_idx]

                if node_probs_dict is not None:
                    pred_score_dict_no_grad = {
                        node_id: prob.detach().item()
                        for node_id, prob in node_probs_dict.items()
                    }
                else:
                    raise ValueError(f"ValidatetextPretrainstagenode_probs_dict = None")

                bce_loss = compute_topk_bce_loss(
                    true_score_dict=true_score_dict,
                    pred_prob_dict=node_probs_dict,
                    top_k=self.hit_rate_k,
                    device=self.device,
                    split_idx=split_idx
                )

                split_loss = bce_loss

                if mode == 'val':
                    if split_idx not in self.val_split_loss_history:
                        self.val_split_loss_history[split_idx] = {'bce_loss': []}
                    elif 'bce_loss' not in self.val_split_loss_history[split_idx]:
                        self.val_split_loss_history[split_idx]['bce_loss'] = []
                    self.val_split_loss_history[split_idx]['bce_loss'].append(float(bce_loss.detach().cpu()))
                elif mode == 'test':
                    if split_idx not in self.test_split_loss_history:
                        self.test_split_loss_history[split_idx] = {'bce_loss': []}
                    elif 'bce_loss' not in self.test_split_loss_history[split_idx]:
                        self.test_split_loss_history[split_idx]['bce_loss'] = []
                    self.test_split_loss_history[split_idx]['bce_loss'].append(float(bce_loss.detach().cpu()))

                try:
                    metrics_calculator = RankingMetrics(
                        true_score_dict=true_score_dict,
                        pred_score_dict=pred_score_dict_no_grad,
                        device=self.device,
                        split_idx=split_idx
                    )

                    metrics_results = metrics_calculator.compute_all(
                        ndcg_ks=self.ndcg_ks,
                        hit_rate_k=self.hit_rate_k,
                        split_type=mode,
                        split_idx=split_idx
                    )

                    ndcg_vals = metrics_results['ndcg']
                    for k in self.ndcg_ks:
                        if k in ndcg_vals:
                            if k == 10:
                                all_ndcgs_10.append(ndcg_vals[k])
                            elif k == 20:
                                all_ndcgs_20.append(ndcg_vals[k])
                            elif k == 30:
                                all_ndcgs_30.append(ndcg_vals[k])
                    all_hit_rates.append(metrics_results['hit_rate'])
                    all_aucs.append(metrics_results['auc'])
                    all_f1s.append(metrics_results['f1'])
                    all_mrrs.append(metrics_results['mrr'])
                except Exception as e:
                    raise ValueError(f"Pretrainstagesplit {split_idx} rankingmetriccalculation failed: {str(e)}")

            else:
                pred_score_dict = self.split_pred_score_dict[split_idx]

                if pred_score_dict is not None:
                    pred_score_dict_no_grad = {
                        node_id: score.detach().item()
                        for node_id, score in pred_score_dict.items()
                    }
                else:
                    raise ValueError(f"ValidatetextFine_tuningstagepred_score_dict = None")

                mse_loss = compute_score_mse_loss(
                    true_score_dict,
                    pred_score_dict_no_grad,
                    top_k=self.hit_rate_k,
                    device=self.device,
                    split_idx=split_idx
                )

                key = (int(split_idx), int(split_data['last_gnn_window']))
                if key not in self.model.output_global_nodes_by_window_key:
                    raise RuntimeError(
                        f"split {split_idx}: textwindow {split_data['last_gnn_window']} textoutputtextnodetext")
                model_global_nodes = self.model.output_global_nodes_by_window_key[key]

                a_hat_sub, A_sub = self.a_hat_calculator.build_recon_mats_intersection(
                    processor=self.processor,
                    split_idx=split_idx,
                    target_window_id=target_window_id,
                    model_global_nodes=model_global_nodes,
                    a_hat=a_hat
                )

                if a_hat_sub is None or A_sub is None:
                    recon_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                else:
                    recon_loss = compute_reconstruction_loss(a_hat_sub, A_sub)

                split_loss = compute_total_loss(mse_loss, recon_loss, gamma=self.gamma)

                if mode == 'val':
                    if split_idx not in self.val_split_loss_history:
                        self.val_split_loss_history[split_idx] = {
                            'total_loss': [],
                            'mse_loss': [],
                            'recon_loss': []
                        }
                    self.val_split_loss_history[split_idx]['total_loss'].append(float(split_loss.detach().cpu()))
                    self.val_split_loss_history[split_idx]['mse_loss'].append(float(mse_loss.detach().cpu()))
                    self.val_split_loss_history[split_idx]['recon_loss'].append(float(recon_loss.detach().cpu()))
                elif mode == 'test':
                    if split_idx not in self.test_split_loss_history:
                        self.test_split_loss_history[split_idx] = {
                            'total_loss': [],
                            'mse_loss': [],
                            'recon_loss': []
                        }
                    self.test_split_loss_history[split_idx]['total_loss'].append(float(split_loss.detach().cpu()))
                    self.test_split_loss_history[split_idx]['mse_loss'].append(float(mse_loss.detach().cpu()))
                    self.test_split_loss_history[split_idx]['recon_loss'].append(float(recon_loss.detach().cpu()))

                try:
                    metrics_calculator = RankingMetrics(
                        true_score_dict=true_score_dict,
                        pred_score_dict=pred_score_dict_no_grad,
                        device=self.device,
                        split_idx=split_idx
                    )

                    metrics_results = metrics_calculator.compute_all(
                        ndcg_ks=self.ndcg_ks,
                        hit_rate_k=self.hit_rate_k,
                        split_type=mode,
                        split_idx=split_idx
                    )

                    ndcg_vals = metrics_results['ndcg']
                    for k in self.ndcg_ks:
                        if k in ndcg_vals:
                            if k == 10:
                                all_ndcgs_10.append(ndcg_vals[k])
                            elif k == 20:
                                all_ndcgs_20.append(ndcg_vals[k])
                            elif k == 30:
                                all_ndcgs_30.append(ndcg_vals[k])
                    all_hit_rates.append(metrics_results['hit_rate'])
                    all_aucs.append(metrics_results['auc'])
                    all_f1s.append(metrics_results['f1'])
                    all_mrrs.append(metrics_results['mrr'])

                except Exception as e:
                    raise RuntimeError(f"split {split_idx} rankingmetriccalculation failed: {str(e)}")

            total_val_loss += float(split_loss.detach().cpu())
            total_windows += 1

        avg_val_loss = total_val_loss / total_windows if total_windows > 0 else 0.0
        avg_ndcg_10 = float(np.mean(all_ndcgs_10)) if all_ndcgs_10 else 0.0
        avg_ndcg_20 = float(np.mean(all_ndcgs_20)) if all_ndcgs_20 else 0.0
        avg_ndcg_30 = float(np.mean(all_ndcgs_30)) if all_ndcgs_30 else 0.0
        avg_hit_rate = float(np.mean(all_hit_rates)) if all_hit_rates else 0.0
        avg_auc = float(np.mean(all_aucs)) if all_aucs else 0.0
        avg_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0
        avg_mrr = float(np.mean(all_mrrs)) if all_mrrs else 0.0

        out = {
            f'ndcg@10': round(float(avg_ndcg_10), 4),
            'ndcg@20': round(float(avg_ndcg_20), 4),
            'ndcg@30': round(float(avg_ndcg_30), 4),
            f'hit_rate@{self.hit_rate_k}': round(float(avg_hit_rate), 4),
            'auc': round(float(avg_auc), 4),
            'f1': round(float(avg_f1), 4),
            'mrr': round(float(avg_mrr), 4),
            'total_loss': round(float(avg_val_loss), 4)
        }

        return out

    def test(self, split_type="test"):
        """testtext """
        test_splits = self.test_split_indices

        if not test_splits:
            raise ValueError(f"missingtext{split_type}split")

        test_metrics = self.validate(test_splits, mode='test')

        return test_metrics

    def _record_params_before_update(self):
        """textoptimizationtextparametertext（textallparameter，containstext）"""
        self._params_before_cache = {}
        for param_name, param in self.model.named_parameters():
            self._params_before_cache[param_name] = param.data.detach().clone()

    def _check_param_updates(self, epoch, stage_name):
        """checkparametertext（text：containstext；textoutputtext）"""

        if self._params_before_cache is None:
            return

        module_stats = {}

        for param_name, param in self.model.named_parameters():
            before = self._params_before_cache.get(param_name, None)
            if before is None:
                continue

            mod = param_name.split('.')[0] if '.' in param_name else param_name

            if mod not in module_stats:
                module_stats[mod] = {
                    'total_params': 0,
                    'trainable_params': 0,
                    'frozen_params': 0,
                    'updated_trainable': 0,
                    'unchanged_trainable': 0,
                    'updated_frozen': 0,
                    'param_names_updated_trainable': [],
                    'param_names_updated_frozen': []
                }

            st = module_stats[mod]
            st['total_params'] += 1

            is_trainable = bool(param.requires_grad)
            if is_trainable:
                st['trainable_params'] += 1
            else:
                st['frozen_params'] += 1

            changed = not torch.allclose(param.data, before, rtol=1e-6, atol=1e-8)

            if is_trainable:
                if changed:
                    st['updated_trainable'] += 1
                    st['param_names_updated_trainable'].append(param_name)
                else:
                    st['unchanged_trainable'] += 1
            else:
                if changed:
                    st['updated_frozen'] += 1
                    st['param_names_updated_frozen'].append(param_name)

        ordered_modules = ['gcn', 'gru', 'beta', 'node_classification_mlp', 'node_scoring_mlp']
        order_map = {m: i for i, m in enumerate(ordered_modules)}
        default_order = len(ordered_modules)

        logger.info(f"======== {stage_name} Epoch {epoch}: parametertext（text） ========")

        for mod in sorted(module_stats.keys(), key=lambda m: order_map.get(m, default_order)):
            st = module_stats[mod]
            logger.info(
                f"text '{mod}': parameternumber={st['total_params']} ， "
                f"text={st['updated_trainable']} ， "
                f"text={st['frozen_params']}"
            )

            if st['updated_frozen'] > 0:
                raise ValueError("text：textparameter: " + ", ".join(st['param_names_updated_frozen']))


    def _save_lr_record(self, lr_record,save_dir):
        """savelearning ratetextYAMLfile"""
        try:
            lr_file_path = os.path.join(save_dir, "lr_update_record.yaml")

            sorted_epochs = sorted(lr_record.keys())

            lr_list = []
            for epoch in sorted_epochs:
                lr = lr_record[epoch]
                lr_list.append({
                    "epoch": epoch,
                    "learning_rate": float(lr)
                })

            with open(lr_file_path, 'w') as f:
                yaml.dump(lr_list, f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            raise RuntimeError(f"savelearning ratetext: {str(e)}")

    def _save_score_file_and_scatter_plot(self):
        """savetext（textFine_tuningstagetext）"""
        if self.training_stage != 'finetuning':
            return

        finetune_dir = os.path.join(self.config['results_dir'], "Finetuning")

        self.model.eval()

        train_scores_dir = os.path.join(finetune_dir, "train_scores")
        self._calculate_final_score_and_rank(self.train_split_indices, train_scores_dir, "train")

        val_scores_dir = os.path.join(finetune_dir, "val_scores")
        self._calculate_final_score_and_rank(self.val_split_indices, val_scores_dir, "val")

        test_scores_dir = os.path.join(finetune_dir, "test_scores")
        self._calculate_final_score_and_rank(self.test_split_indices, test_scores_dir, "test")

    def _generate_ranking_summary(self, val_metrics, test_metrics, save_dir, filename, stage_name):
        """textrankingtextYAMLfile"""
        try:
            hyperparameters = {
                'betweenness_params': {
                    'gnn_stride': self.config.get('gnn_stride', 3),
                    'pred_stride': self.config.get('pred_stride', 1)
                },
                'model_params': {
                    'pretrain_learning_rate': self.config.get('pretrain_learning_rate'),
                    'finetune_learning_rate': self.config.get('finetune_learning_rate'),
                    'batch_size': self.config.get('batch_size'),
                    'gcn_hidden_dim': self.config.get('gcn_hidden_dim'),
                    'gcn_output_dim': self.config.get('gcn_output_dim'),
                    'gru_output_dim': self.config.get('gru_output_dim'),
                    'gru_num_layers': self.config.get('gru_num_layers'),
                    'dropout': self.config.get('dropout'),
                    'weight_decay': self.config.get('weight_decay'),
                    'gamma': self.config.get('gamma', 1.0)
                },
                'evaluation_params': {
                    'ndcg_ks': self.config.get('ndcg_ks', [10, 30, 20]),
                    'hit_rate_k': self.config.get('hit_rate_k', 30)
                }
            }

            ranking_summary = {
                'exp_id': self.exp_id,
                'stage': stage_name,
                'hyperparameters': hyperparameters,
                'test_metrics': {
                    'ndcg_10': round(float(test_metrics['ndcg@10']), 4),
                    'ndcg_20': round(float(test_metrics['ndcg@20']), 4),
                    'ndcg_30': round(float(test_metrics['ndcg@30']), 4),
                    f'hit_rate@{self.hit_rate_k}': round(float(test_metrics[f'hit_rate@{self.hit_rate_k}']), 4),
                    'auc': round(float(test_metrics['auc']), 4),
                    'f1': round(float(test_metrics['f1']), 4),
                    'mrr': round(float(test_metrics['mrr']), 4),
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            summary_path = os.path.join(save_dir, filename)
            with open(summary_path, 'w') as f:
                yaml.dump(ranking_summary, f, default_flow_style=False)

            logger.info(f"text{stage_name} ranking_summary.yaml: {summary_path}")

        except Exception as e:
            logger.error(f"text {stage_name} ranking_summary.yaml text: {str(e)}")

    def _calculate_final_score_and_rank(self, split_indices, save_dir, split_type):
        """savetextsplittypetext（containsranking）"""
        os.makedirs(save_dir, exist_ok=True)

        for split_idx in split_indices:
            try:
                split_data = self.all_split_data[split_idx]
                target_window_id = split_data['target_window_id']

                if split_idx in self.split_pred_score_dict:
                    pred_score_dict = self.split_pred_score_dict[split_idx]
                    if pred_score_dict is None:
                        raise ValueError(f"text， pred_score_dict = None")

                    pred_score_dict_no_grad = {
                        node_id: score.detach().item()
                        for node_id, score in pred_score_dict.items()
                    }

                    try:
                        true_score_dict = self.load_ranking_file(split_idx, target_window_id)

                        vis_dir = os.path.join(str(self.config['results_dir']), "Visualization")
                        os.makedirs(vis_dir, exist_ok=True)
                        save_path = os.path.join(vis_dir, f"split_{split_idx}_visual.png")
                        _title = f"Split {split_idx} • Target window {target_window_id}"
                        plot_pred_true_scatter(
                            pred_score_dict_no_grad,
                            true_score_dict,
                            save_path,
                            title=_title,
                            gamma=self.gamma,
                            stage='finetuning'
                        )
                    except Exception as e:
                        raise RuntimeError(f"plot{split_type}split{split_idx}text: {str(e)}")

                    node_ids = list(pred_score_dict.keys())
                    node_scores = [score.detach().cpu().item() for score in pred_score_dict.values()]
                    node_scores_np = np.array(node_scores)

                    sorted_indices = np.argsort(-node_scores_np)
                    sorted_scores = node_scores_np[sorted_indices]

                    ranks = []
                    i = 0
                    while i < len(sorted_scores):
                        j = i
                        current_score = sorted_scores[i]
                        while j < len(sorted_scores) and abs(sorted_scores[j] - current_score) < 1e-9:
                            j += 1

                        average_rank = (i + 1 + j) / 2.0

                        for k in range(i, j):
                            ranks.append(average_rank)

                        i = j

                    rows = []
                    for idx, node_id in enumerate(node_ids):
                        rows.append({
                            "split_idx": int(split_idx),
                            "window_id": int(target_window_id),
                            "node_ID": node_id,
                            "score": node_scores[idx],
                            "rank": ranks[sorted_indices.tolist().index(idx)]
                        })

                    score_df = pd.DataFrame(rows, columns=["split_idx", "window_id", "node_ID", "score", "rank"])
                    score_df = score_df.sort_values(by="rank", ascending=True)
                    score_file = os.path.join(save_dir, f"split_{split_idx}_score.parquet")
                    score_df.to_parquet(score_file, index=False)

            except Exception as e:
                logger.warning(f"save{split_type}split{split_idx}text: {str(e)}")

    def _clear_forward_caches(self, split_indices):
        """text"""
        for s in split_indices:
            if s in self.split_representations:
                try:
                    del self.split_representations[s]
                except Exception as e:
                    raise RuntimeError(f"textsplit {s}text: {e}")

            if s in self.split_a_hat:
                try:
                    del self.split_a_hat[s]
                except Exception as e:
                    raise RuntimeError(f"textsplit {s}textA^text: {e}")

            if s in self.split_pred_score_dict:
                try:
                    del self.split_pred_score_dict[s]
                except Exception as e:
                    raise RuntimeError(f"textsplit {s}textnodetext: {e}")

            if s in self.split_node_probs_dict:
                try:
                    del self.split_node_probs_dict[s]
                except Exception as e:
                    raise RuntimeError(f"textsplit {s}textnodetext: {e}")

        torch.cuda.empty_cache()
        gc.collect()
