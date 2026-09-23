import torch
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import random
import traceback
import gc
import yaml
import platform
import torch.backends.cudnn
import multiprocessing as mp
import torch.multiprocessing as tmp
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.importance import get_param_importances

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

_current = mp.get_start_method(allow_none=True)
if _current is None:
    if platform.system().lower().startswith("win"):
        mp.set_start_method("spawn", force=True)
    else:
        mp.set_start_method("fork", force=True)

tmp.set_sharing_strategy("file_system")

from Project.train import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Email_two_stages")

logging.getLogger("data.Negative_sampler").setLevel(logging.WARNING)
logging.getLogger("Project.train").setLevel(logging.WARNING)
logging.getLogger("data.dataloader").setLevel(logging.WARNING)
logging.getLogger("models.graph").setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.WARNING)

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.9'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["NUMEXPR_MAX_THREADS"] = "24"
os.environ['OMP_NUM_THREADS'] = "24"
os.environ['MKL_NUM_THREADS'] = "24"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_SHM_DISABLE'] = '1'
os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
os.environ['NCCL_SOCKET_NTHREADS'] = '4'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

COMPLETE_HYPERPARAMETER_CONFIG = {
    'betweenness_params': {
        'gnn_stride': 3,
        'pred_stride': 1,
    },

    'model_params': {
        'pretrain_learning_rate': 0.00134,
        'finetune_learning_rate': 0.00144,
        'batch_size': 8,
        'gcn_hidden_dim': 32,
        'gcn_output_dim': 128,
        'gru_output_dim': 256,
        'gru_num_layers': 1,
        'dropout': 0.5,
        'weight_decay': 0.03,
        'gamma': 0.01
    },
    'evaluation_params': {
        'ndcg_ks': [10, 20, 30],
        'hit_rate_k': 30,
    },
    'base_config': {
        'processor_dir': "./data/processed_data",
        'background_rankings_dir': "./data/background_rankings",
        'results_dir': './AUC_and_F1_MRR(K=20)',
        'num_workers': 20,
        'seed': 42,
        'node_feat_dim': 4,
        'is_undirected': True,
        'early_stop_patience': 20,
        'warmup_epochs': 60,
        'min_improvement_threshold ': 0.1,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'ndcg_ks': [10, 20, 30],
        'hit_rate_k': 30,
    }
}

def set_global_random_seed(seed: int):
    """settextrandom seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def run_model_experiment(config: dict, exp_id: int, global_seed: int = 42) -> dict:
    """textmodelexperiment """
    try:
        torch.cuda.empty_cache()
        gc.collect()

        merged_config = {
            **config['base_config'],
            **config['betweenness_params'],
            **config['model_params'],
            **config['evaluation_params'],
            'exp_id': f"exp_{exp_id}"
        }

        logger.info(f"Starting experiment {exp_id}")

        trainer = Trainer(merged_config, f"exp_{exp_id}")

        if not hasattr(trainer, 'model') or trainer.model is None:
            raise RuntimeError("GNNmodeltextinitialize")

        trainer.setup_experiment()
        (pretrain_best_loss, finetune_best_loss,
         val_ndcg_10, val_ndcg_20, val_ndcg_30,
         pretrain_test_ndcg_10, pretrain_test_ndcg_20, pretrain_test_ndcg_30,
         pretrain_test_hit_rate_30, pretrain_test_auc, pretrain_test_f1,pretrain_test_mrr,
         finetune_test_ndcg_10, finetune_test_ndcg_20, finetune_test_ndcg_30,
         finetune_test_hit_rate_30, finetune_test_auc, finetune_test_f1,finetune_test_mrr,
         best_model_state) = trainer.train_epoch(split_type="train")

        return {
            'exp_id': exp_id,
            'status': 'success',
            'pretrain_best_loss': pretrain_best_loss,
            'finetune_best_loss': finetune_best_loss,
            'val_ndcg_10': val_ndcg_10,
            'val_ndcg_20': val_ndcg_20,
            'val_ndcg_30': val_ndcg_30,
            'pretrain_test_ndcg_10': pretrain_test_ndcg_10,
            'pretrain_test_ndcg_20': pretrain_test_ndcg_20,
            'pretrain_test_ndcg_30': pretrain_test_ndcg_30,
            'pretrain_test_hit_rate_30': pretrain_test_hit_rate_30,
            'pretrain_test_auc': pretrain_test_auc,
            'pretrain_test_f1': pretrain_test_f1,
            'pretrain_test_mrr': pretrain_test_mrr,
            'finetune_test_ndcg_10': finetune_test_ndcg_10,
            'finetune_test_ndcg_20': finetune_test_ndcg_20,
            'finetune_test_ndcg_30': finetune_test_ndcg_30,
            'finetune_test_hit_rate_30': finetune_test_hit_rate_30,
            'finetune_test_auc': finetune_test_auc,
            'finetune_test_f1': finetune_test_f1,
            'finetune_test_mrr': finetune_test_mrr,
            'best_model_state': best_model_state,
            'config': merged_config
        }

    except Exception as e:
        raise RuntimeError(f"experiment {exp_id} text: {str(e)}")

def hyperparameter_optimization(config: dict, num_trials: int = 5) -> dict:
    """useOptunatexthyperparameteroptimization，textoptimizationtextparameter"""
    seed = int(config['base_config']['seed'])
    sampler = TPESampler(seed=seed)
    pruner = optuna.pruners.NopPruner()

    fixed_betweenness = config['betweenness_params'].copy()
    fixed_base = config['base_config'].copy()

    fixed_model_params = {}
    for param, values in config['model_params'].items():
        fixed_model_params[param] = values[0]

    fixed_evaluation_params = config['evaluation_params'].copy()

    def objective(trial: optuna.Trial) -> float:
        """text：textvalidationtextNDCG@30"""
        pretrain_lr = trial.suggest_float('pretrain_learning_rate', 3e-4, 5e-2, log=True)

        finetune_lr = trial.suggest_float('finetune_learning_rate', pretrain_lr / 10.0, pretrain_lr / 3.0, log=True)

        optimized_params = {
            #'pretrain_learning_rate': pretrain_lr,
            #'finetune_learning_rate': finetune_lr,
            # 'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            #'gcn_hidden_dim': trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64]),  #  gcn_hidden_dim
            #'gcn_output_dim': trial.suggest_categorical('gcn_output_dim', [64, 128, 512]),  #  gcn_output_dim
            #'gru_output_dim': trial.suggest_categorical('gru_output_dim', [64, 128, 256, 512]),
            # 'gamma': trial.suggest_categorical('gamma',  [0.1, 0.2, 0.3,0.4,0.5,0.01,0.05,0.08]),
            # 'dropout': trial.suggest_categorical('dropout', [0.1, 0.3, 0.5]),
            #'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }

        model_params = fixed_model_params.copy()
        model_params.update(optimized_params)

        hp_config = {
            'betweenness_params': fixed_betweenness,
            'model_params': model_params,
            'evaluation_params': fixed_evaluation_params,
            'base_config': fixed_base
        }

        result = run_model_experiment(hp_config, exp_id=trial.number, global_seed=seed)
        if result.get('status') != 'success':
            raise RuntimeError("text")

        trial.set_user_attr('config', hp_config)
        trial.set_user_attr('metrics', {
            "pretrain_test_ndcg_10": float(result.get("pretrain_test_ndcg_10", 0)),
            "pretrain_test_ndcg_20": float(result.get("pretrain_test_ndcg_20", 0)),
            "pretrain_test_ndcg_30": float(result.get("pretrain_test_ndcg_30", 0)),
            "pretrain_test_hit_rate_30": float(result.get("pretrain_test_hit_rate_30", 0)),
            "pretrain_test_auc": float(result.get("pretrain_test_auc", 0)),
            "pretrain_test_f1": float(result.get("pretrain_test_f1", 0)),
            "pretrain_test_mrr": float(result.get("pretrain_test_mrr", 0)),
            "finetune_test_ndcg_10": float(result.get("finetune_test_ndcg_10", 0)),
            "finetune_test_ndcg_20": float(result.get("finetune_test_ndcg_20", 0)),
            "finetune_test_ndcg_30": float(result.get("finetune_test_ndcg_30", 0)),
            "finetune_test_hit_rate_30": float(result.get("finetune_test_hit_rate_30", 0)),
            "finetune_test_auc": float(result.get("finetune_test_auc", 0)),
            "finetune_test_f1": float(result.get("finetune_test_f1", 0)),
            "finetune_test_mrr": float(result.get("finetune_test_mrr", 0)),
            })


        return float(result['val_ndcg_30'])

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name="hyperparameter_optimization"
    )

    logger.info(f"textOptunaoptimization，text: {num_trials}")
    study.optimize(objective, n_trials=int(num_trials), n_jobs=1)

    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(config['base_config']['results_dir'], "optuna_results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    _export_trials_to_disk(study, save_dir=run_dir)

    best_trial = study.best_trial
    if 'config' not in best_trial.user_attrs:
        raise RuntimeError("textconfiguration")

    best_config = best_trial.user_attrs['config']

    best_config_file = os.path.join(run_dir, "best_config.yaml")
    with open(best_config_file, 'w', encoding='utf-8') as f:
        yaml.dump(best_config, f, allow_unicode=True, sort_keys=False)

    try:
        importances = get_param_importances(study)

        importance_file = os.path.join(run_dir, "parameter_importances.yaml")
        with open(importance_file, 'w') as f:
            yaml.dump({k: float(v) for k, v in importances.items()}, f, default_flow_style=False)

        logger.info(f"parametertextsavetext: {run_dir}")
    except Exception as e:
        logger.warning(f"parametertext: {e}")

    study_summary = {
        'best_trial_number': best_trial.number,
        'best_exp_id': f"exp_{best_trial.number}",
        'best_value': float(best_trial.value),
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'timestamp': timestamp
    }

    summary_file = os.path.join(run_dir, "study_summary.yaml")
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump(study_summary, f, allow_unicode=True, sort_keys=False)

    logger.info(f"Optunaoptimizationcompleted，resultsdirectory: {run_dir}")
    logger.info(f"textexperimentID: exp_{best_trial.number}")

    return {
        'best_config': best_config,
        'best_val_ndcg_30': best_trial.value,
        'best_model_state': best_trial.user_attrs.get('metrics', {}),
        'best_exp_id': f"exp_{best_trial.number}",
        'run_dir': run_dir,
        'study': study
    }

def _export_trials_to_disk(study: optuna.Study, save_dir: str):
    """text‘alltextresults’text CSV/YAML。"""
    os.makedirs(save_dir, exist_ok=True)

    try:
        formatted_rows = []

        for t in study.trials:
            trial_num = int(t.number)
            params = t.params or {}
            user_attrs = t.user_attrs or {}

            config = user_attrs.get("config", {})
            metrics = user_attrs.get("metrics", {})

            if not isinstance(metrics, dict) or len(metrics) == 0:
                raise RuntimeError(
                    f"text {trial_num}: metricstextis empty！user_attrs keys: {list(user_attrs.keys())}"
                )

            model_params = config.get("model_params", {})

            gcn_hidden_dim = params.get("gcn_hidden_dim", model_params.get("gcn_hidden_dim", 32))
            gcn_output_dim = params.get("gcn_output_dim", model_params.get("gcn_output_dim", 32))
            gru_output_dim = params.get("gru_output_dim", model_params.get("gru_output_dim", 64))
            gamma = params.get("gamma", model_params.get("gamma", 0.1))
            batch_size = params.get("batch_size", model_params.get("batch_size", 16))
            pretrain_learning_rate = params.get(
                "pretrain_learning_rate", model_params.get("pretrain_learning_rate", 0.02)
            )
            finetune_learning_rate = params.get(
                "finetune_learning_rate", model_params.get("finetune_learning_rate", 0.02)
            )

            dropout = model_params.get("dropout", 0.3)
            weight_decay = model_params.get("weight_decay", 1.0e-4)

            pretrain_test_ndcg_10 = metrics["pretrain_test_ndcg_10"]
            pretrain_test_ndcg_20 = metrics["pretrain_test_ndcg_20"]
            pretrain_test_ndcg_30 = metrics["pretrain_test_ndcg_30"]
            pretrain_test_hit_rate_30 = metrics["pretrain_test_hit_rate_30"]
            pretrain_test_auc = metrics.get("pretrain_test_auc", 0.0)
            pretrain_test_f1 = metrics.get("pretrain_test_f1", 0.0)
            pretrain_test_mrr = metrics.get("pretrain_test_mrr", 0.0)

            finetune_test_ndcg_10 = metrics["finetune_test_ndcg_10"]
            finetune_test_ndcg_20 = metrics["finetune_test_ndcg_20"]
            finetune_test_ndcg_30 = metrics["finetune_test_ndcg_30"]
            finetune_test_hit_rate_30 = metrics["finetune_test_hit_rate_30"]
            finetune_test_auc = metrics.get("finetune_test_auc", 0.0)
            finetune_test_f1 = metrics.get("finetune_test_f1", 0.0)
            finetune_test_mrr = metrics.get("finetune_test_mrr", 0.0)

            formatted_rows.append({
                "trial_number": trial_num,
                "gcn_hidden_dim": int(gcn_hidden_dim),
                "gcn_output_dim": int(gcn_output_dim),
                "gru_output_dim": int(gru_output_dim),
                "gamma": float(gamma),
                "batch_size": int(batch_size),
                "dropout": float(dropout),
                "weight_decay": float(weight_decay),
                "pretrain_learning_rate": format(float(pretrain_learning_rate), ".5f"),
                "pretrain_ndcg@10": float(pretrain_test_ndcg_10),
                "pretrain_ndcg@20": float(pretrain_test_ndcg_20),
                "pretrain_ndcg@30": float(pretrain_test_ndcg_30),
                "pretrain_hit_rate@30": float(pretrain_test_hit_rate_30),
                "pretrain_auc": float(pretrain_test_auc),
                "pretrain_f1": float(pretrain_test_f1),
                "pretrain_mrr": float(pretrain_test_mrr),
                "finetune_learning_rate": format(float(finetune_learning_rate), ".5f"),
                "finetune_ndcg@10": float(finetune_test_ndcg_10),
                "finetune_ndcg@20": float(finetune_test_ndcg_20),
                "finetune_ndcg@30": float(finetune_test_ndcg_30),
                "finetune_hit_rate@30": float(finetune_test_hit_rate_30),
                "finetune_auc": float(finetune_test_auc),
                "finetune_f1": float(finetune_test_f1),
                "finetune_mrr": float(finetune_test_mrr),
            })

        columns_order = [
            "trial_number",
            "gcn_hidden_dim",
            "gcn_output_dim",
            "gru_output_dim",
            "gamma",
            "batch_size",
            "dropout",
            "weight_decay",
            "pretrain_learning_rate",
            "pretrain_ndcg@10",
            "pretrain_ndcg@20",
            "pretrain_ndcg@30",
            "pretrain_hit_rate@30",
            "pretrain_auc",
            "pretrain_f1",
            "pretrain_mrr",
            "finetune_learning_rate",
            "finetune_ndcg@10",
            "finetune_ndcg@20",
            "finetune_ndcg@30",
            "finetune_hit_rate@30",
            "finetune_auc",
            "finetune_f1",
            "finetune_mrr",
        ]

        result_df = pd.DataFrame(formatted_rows, columns=columns_order)
        result_df.to_csv(os.path.join(save_dir, "all_trials.csv"), index=False, encoding="utf-8")
        logger.info(f"successtextCSVfiletext: {os.path.join(save_dir, 'all_trials.csv')}")

    except Exception as e:
        logger.warning(f"text all_trials.csv text：{e}")
        import traceback
        traceback.print_exc()

def main():
    """text - useOptunatexthyperparameteroptimization"""
    try:

        logger.info("texttraining（texthyperparameter，texthyperparametertext）")

        seed = int(COMPLETE_HYPERPARAMETER_CONFIG['base_config']['seed'])
        set_global_random_seed(seed)

        # optuna_results = hyperparameter_optimization(
        #     COMPLETE_HYPERPARAMETER_CONFIG,
        # )
        # best_exp_id = optuna_results['best_exp_id']

        result = run_model_experiment(
            COMPLETE_HYPERPARAMETER_CONFIG,
            exp_id=1,  # Example experiment ID
            global_seed=42
        )

        exp_id = result['exp_id']

        logger.info("=" * 60)
        logger.info(f"experimentcompleted: experimentID: {exp_id}")

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Optunahyperparameteroptimizationtext: {str(e)}")


if __name__ == "__main__":
    main()
