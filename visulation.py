import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def build_figure_title(gamma: float, stage: str = None, base: str = None) -> str:
    """
    textfiguretitle：
    - textstageparametertexttitletext
    """
    if stage == 'finetuning':
        loss_formula_title = f"Finetuning:  mse_loss + {gamma:.1f} * recon_loss"
    elif stage == 'pretraining':
        loss_formula_title = f"Pretraining: bce Loss"
    else:
        raise ValueError('stage must be finetuning or pretraining.')

    if base:
        return f"{loss_formula_title}\n{base}"
    return loss_formula_title

def plot_window_loss_curve_split(window_losses, split_idx, exp_id, save_dir, gamma: float = 0.5, stage: str = None):
    """plothyperparametertrainingtextwindowtextlosstext（text）"""
    os.makedirs(save_dir, exist_ok=True)

    if stage == 'pretraining':
        current_stage = stage
        bce_losses = window_losses.get('bce_loss', [])
        loss_labels = ['BCE Loss']
        colors = ['blue']
        losses_to_plot = [bce_losses]

    else:  # 'finetuning'
        current_stage = stage
        total_losses = window_losses.get('total_loss', [])
        mse_losses = window_losses.get('mse_loss', [])
        recon_losses = window_losses.get('recon_loss', [])
        loss_labels = ['Total Loss', 'MSE Loss', 'Reconstruction Loss']
        colors = ['red', 'orange', 'green']
        losses_to_plot = [total_losses, mse_losses, recon_losses]

    min_length = min(len(loss) for loss in losses_to_plot)
    epochs = range(1, min_length + 1)

    plt.figure(figsize=(12, 7))

    for i, (loss_values, label, color) in enumerate(zip(losses_to_plot, loss_labels, colors)):
        plt.plot(epochs, loss_values[:min_length], 'o-', color=color, linewidth=2, markersize=4, label=label)

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')

    base_title = f'{exp_id}_Split_{split_idx}_Loss'
    plt.title(build_figure_title(gamma, current_stage, base_title))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if "val_loss" in save_dir:
        file_name = f"split_{split_idx}_val_loss.png"
    elif "test_loss" in save_dir:
        file_name = f"split_{split_idx}_test_loss.png"
    else:
        file_name = f"split_{split_idx}_training_loss.png"

    plt.savefig(os.path.join(save_dir, file_name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_auc_epoch_curve(val_auc_history, test_auc_history, exp_id, save_dir,gamma: float = 0.5,stage: str = None):
    """plotAUCtextepochtext"""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 7))

    min_length = min(len(val_auc_history), len(test_auc_history))
    epochs = range(1, min_length + 1)

    plt.plot(epochs, val_auc_history[:min_length], 'o-', color='blue',
             linewidth=2, markersize=6, label='Validation AUC', alpha=0.8)

    plt.plot(epochs, test_auc_history[:min_length], 's-', color='orange',
             linewidth=2, markersize=5, label='Test AUC', alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC Value', fontsize=12)

    plt.ylim(0.0, 1.05)

    plt.grid(True, alpha=0.3, linestyle='--')

    base_title = f'{exp_id} - AUC'
    plt.title(build_figure_title(gamma,stage,base_title), fontsize=14, pad=15)

    plt.legend(loc='best', fontsize=11)

    if len(val_auc_history) > 0:
        last_val_auc = val_auc_history[-1]
        plt.annotate(f'{last_val_auc:.4f}',
                     xy=(len(val_auc_history), last_val_auc),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, color='blue')

    if len(test_auc_history) > 0:
        last_test_auc = test_auc_history[-1]
        plt.annotate(f'{last_test_auc:.4f}',
                     xy=(len(test_auc_history), last_test_auc),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, color='orange')

    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "auc_epoch_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_train_val_loss_curve(train_loss_history, val_loss_history,  test_loss_history, exp_id, save_dir,gamma: float = 0.5,stage: str = None):
    """plottrainingtextvalidationtextaveragetextlosstext"""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    min_length = min(len(train_loss_history), len(val_loss_history))
    epochs = range(1, min_length + 1)

    plt.plot(epochs, train_loss_history[:min_length], 'o-', color='blue',
             linewidth=2, markersize=4, label='Train Loss')
    plt.plot(epochs, val_loss_history[:min_length], 'o-', color='orange',
             linewidth=2, markersize=4, label='Validation Loss')
    plt.plot(epochs, test_loss_history[:min_length], 'o-', color='purple',
             linewidth=2, markersize=4, label='Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Average Total Loss')

    base_title = f'{exp_id} - Train vs Validation Loss'
    plt.title(build_figure_title(gamma,stage,base_title))

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, "train_val_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ndcg_curve(val_ndcg_history, test_ndcg_history, exp_id, save_dir, k=50, gamma: float = 0.5, stage: str = None):
    """plotvalidationtexttestNDCGtext"""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    min_length = min(len(val_ndcg_history), len(test_ndcg_history))
    epochs = range(1, min_length + 1)

    plt.plot(epochs, val_ndcg_history[:min_length], 'o-', color='blue',
             linewidth=2, markersize=4, label=f'Validation NDCG@{k}')

    plt.plot(epochs, test_ndcg_history[:min_length], 's-', color='orange',
             linewidth=2, markersize=4, label=f'Test NDCG@{k}')

    plt.xlabel('Epoch')
    plt.ylabel(f'NDCG@{k}')

    base_title = f'{exp_id} - NDCG@{k}'
    plt.title(build_figure_title(gamma, stage, base_title))

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(os.path.join(save_dir, f"ndcg@{k}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_hit_rate_curve(val_hit_rate_history, test_hit_rate_history, exp_id, save_dir, k=50, gamma: float = 0.5,
                             stage: str = None):
    """plotvalidationtexttesttext"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    min_length = min(len(val_hit_rate_history), len(test_hit_rate_history))
    epochs = range(1, min_length + 1)

    plt.plot(epochs, val_hit_rate_history[:min_length], 'o-', color='blue',
             linewidth=2, markersize=4, label=f'Validation Hit Rate@{k}')

    plt.plot(epochs, test_hit_rate_history[:min_length], 's-', color='orange',
             linewidth=2, markersize=4, label=f'Test Hit Rate@{k}')

    plt.xlabel('Epoch')
    plt.ylabel(f'Hit Rate@{k}')

    base_title = f'{exp_id} - Hit Rate@{k}'
    plt.title(build_figure_title(gamma, stage, base_title))

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(os.path.join(save_dir, f"hit_rate@{k}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_macro_roc(curves, exp_id: str, save_dir: str,gamma: float = 0.5,stage: str = None):

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    xs = np.linspace(0.0, 1.0, 1001)
    plt.plot(xs, xs, linestyle='--', linewidth=1, label='Random')

    for curve in curves:
        label, fpr_grid, mean_tpr = curve
        auc_value = np.trapz(mean_tpr, fpr_grid)
        plt.plot(fpr_grid, mean_tpr, linewidth=2, label=f'{label} (AUC = {auc_value:.4f})')

    plt.xlabel('FPR')
    plt.ylabel('TPR')

    base_title = f'Averaged ROC - {exp_id}'
    plt.title(build_figure_title(gamma,stage,base_title))

    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "ROC_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_pred_true_scatter(pred_scores: dict,
                           true_scores: dict,
                           save_path: str,
                           title: str = None,
                           gamma: float = 0.5,
                           stage: str = None) -> None:
    """
    - textground truthtextpredictiontextnodetext 0，text；
      text，text。
    """
    true_scores_str = {str(k): float(v) for k, v in true_scores.items()}
    pred_scores_str = {str(k): float(v) for k, v in pred_scores.items()}

    common_nodes = sorted(set(true_scores_str.keys()) & set(pred_scores_str.keys()))
    if not common_nodes:
        raise ValueError("true_scores text pred_scores missingtextnode id，text。")

    x_true = np.array([true_scores_str[node] for node in common_nodes], dtype=float)
    y_pred = np.array([pred_scores_str[node] for node in common_nodes], dtype=float)

    std_x = np.std(x_true)
    std_y = np.std(y_pred)
    can_regress = (std_x > 1e-12) and (std_y > 1e-12)

    if can_regress:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_true, y_pred)
        x_line = np.linspace(x_true.min(), x_true.max(), 200)
        y_line = slope * x_line + intercept
        reg_label = (
            f'Regression: ŷ = {slope:.3f}x + {intercept:.3f}\n'
            f'R = {r_value:.3f}, R² = {r_value ** 2:.3f}'
        )
    else:
        slope = intercept = np.nan
        x_line = y_line = None
        reg_label = 'Regression: undefined (all True or Pred scores identical)'

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

    ax.scatter(
        x_true,
        y_pred,
        alpha=0.8,
        s=25,
        color='green',
        label='Predicted Score'
    )

    true_y_level = y_pred.min()
    ax.scatter(
        x_true,
        np.full_like(x_true, true_y_level),
        alpha=0.7,
        s=20,
        color='red',
        label='True Score'
    )

    if x_line is not None:
        ax.plot(
            x_line,
            y_line,
            linewidth=2,
            color='blue',
            label=reg_label
        )
    else:
        ax.scatter([], [], alpha=0.0, label=reg_label)

    x_min, x_max = x_true.min(), x_true.max()
    y_min, y_max = y_pred.min(), y_pred.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    ax.set_xlabel("True Score")
    ax.set_ylabel("Predicted Score")

    if title:
        full_title = build_figure_title(gamma,stage,title)
    else:
        full_title = build_figure_title(gamma,stage,None)
    ax.set_title(full_title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)







