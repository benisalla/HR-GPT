import os
import time
import torch
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from model.config import DatasetConfig, GPTConfig, TrainConfig
from model.model import HRGPT
from dataset.tokenizer import MyTokenizer
from dataset.BatchLoader import BatchLoader


class Trainer:
    def __init__(self, tr_config, model, train_loader, val_loader=None):
        self.tr_config = tr_config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = model.get_optimizer()
        self.callbacks = defaultdict(list)
        self.device = self.tr_config.device
        
        self.model = self.model.to(self.device)
        print(f"Running on device: {self.device}")

        # variables for logging and tracking
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.best_val_loss = float('inf')

        # dirs
        self.ckpt_dir = os.path.join(tr_config.checkpoint_dir, tr_config.experiment_name)
        self.log_dir  = os.path.join(tr_config.log_dir, tr_config.experiment_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)   
        os.makedirs(self.log_dir,  exist_ok=True)  

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model on validation set: loss + per-task metrics."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # aggregate per-task losses
        task_losses = defaultdict(list)

        # aggregate per-task metrics
        cls_correct = defaultdict(int)   # classification: correct predictions
        cls_total   = defaultdict(int)

        reg_sse = defaultdict(float)     # regression: sum of squared errors
        reg_sae = defaultdict(float)     # regression: sum of absolute errors
        reg_cnt = defaultdict(int)       # regression: count

        for x_batch, x_mask, y_list, task_names in self.val_loader:
            # to device
            x_batch = x_batch.to(self.device)
            x_mask  = x_mask.to(self.device)
            y_list  = [y.to(self.device) for y in y_list]

            # compute loss (model already groups by task internally)
            avg_loss, batch_task_losses = self.model(x_batch, x_mask, y_list, task_names)
            total_loss += avg_loss.item()
            for task, loss in batch_task_losses.items():
                task_losses[task].append(loss)
            num_batches += 1

            # group indices by task
            task_to_indices = defaultdict(list)
            for i, tname in enumerate(task_names):
                task_to_indices[tname].append(i)

            for tname, idxs in task_to_indices.items():
                indices = torch.tensor(idxs, dtype=torch.long, device=self.device)
                x_task  = x_batch.index_select(0, indices)
                y_task  = torch.stack([y_list[i] for i in idxs])

                spec = self.model.config.tasks[tname]
                # use model's predict for consistent head logic
                out = self.model.predict(x_task, tname)

                if spec.task_type in ("binary", "multiclass"):
                    # y_task is class indices
                    pred = out["pred"]                    # [N]
                    cls_correct[tname] += (pred == y_task).sum().item()
                    cls_total[tname]   += y_task.numel()
                elif spec.task_type == "regression":
                    # y_task and predicted value are floats
                    val  = out["value"].view(-1)          # [N]
                    yvec = y_task.view(-1)
                    diff = val - yvec
                    reg_sse[tname] += torch.sum(diff * diff).item()
                    reg_sae[tname] += torch.sum(diff.abs()).item()
                    reg_cnt[tname]  += yvec.numel()
                else:
                    raise ValueError(f"Unknown task type for metrics: {spec.task_type}")

        # finalize metrics
        val_metrics = {
            "val/total_loss": total_loss / max(1, num_batches),
        }
        # average task losses
        for task, losses in task_losses.items():
            val_metrics[f"val/{task}_loss"] = sum(losses) / len(losses)

        # classification accuracies
        for task, tot in cls_total.items():
            if tot > 0:
                acc = cls_correct[task] / tot
                val_metrics[f"val/{task}_acc"] = acc * 100.0 

        # regression MSE/MAE
        for task, cnt in reg_cnt.items():
            if cnt > 0:
                mse = reg_sse[task] / cnt
                mae = reg_sae[task] / cnt
                val_metrics[f"val/{task}_mse"] = mse
                val_metrics[f"val/{task}_mae"] = mae

        self.model.train()
        return val_metrics

    def _grad_stats(self):
        """Return global grad stats and (name, l2_norm) per-parameter."""
        total_sq = 0.0
        max_abs = 0.0
        per_param = []
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            # l2 norm of this tensor
            n = g.norm(2).item()
            total_sq += n * n
            max_abs = max(max_abs, g.abs().max().item())
            per_param.append((name, n))
        total_l2 = total_sq ** 0.5
        return {"grad_norm": total_l2, "grad_max_abs": max_abs}, per_param

    def _log_grad_histograms(self, step, every=1000, topk=20):
        """Log histograms for the top-k largest-norm params every `every` steps."""
        if step % every != 0:
            return
        # rank by current grad l2 norm to limit noise/size
        stats, per_param = self._grad_stats()
        per_param.sort(key=lambda x: x[1], reverse=True)
        for name, _ in per_param[:topk]:
            p = dict(self.model.named_parameters())[name]
            if p.grad is not None:
                self.writer.add_histogram(f"grads/{name}", p.grad, step)

    def log_metrics(self, metrics, step, tag: str):
        """Log everything to TensorBoard; print a short, organized line to console."""
        # TensorBoard (full)
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

        # Console (short)
        if step % self.tr_config.log_interval == 0:
            if tag == "train":
                self._print_train_line(step, metrics)
            elif tag == "val":
                self._print_eval_line(step, metrics)

    def save_checkpoint(self, tag: str = "latest"):
        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "iter": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.tr_config.__dict__,
        }, path)
        return path

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.iter_num = ckpt.get("iter", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float('inf'))

    def _key_metric_for_task(self, task_name: str, metrics: dict) -> tuple[str, float | None]:
        """Pick one console metric per task: acc for classification, mse for regression."""
        spec = self.model.config.tasks[task_name]
        if spec.task_type in ("binary", "multiclass"):
            key = f"val/{task_name}_acc"
        elif spec.task_type == "regression":
            key = f"val/{task_name}_mse"  
        else:
            return ("", None)
        return (key, metrics.get(key, None))

    def _print_train_line(self, step: int, metrics: dict):
        it_s = (1.0 / max(self.iter_dt, 1e-9))
        line = (
            f"[{step}] "
            f"train_loss={metrics.get('train/total_loss', float('nan')):.4f} "
            f"lr={metrics.get('train/learning_rate', 0.0):.2e} "
            f"it/s={it_s:.2f} "
            f"best_val={self.best_val_loss:.4f}"
        )
        print(line)

    def _print_eval_line(self, step: int, metrics: dict):
        base = f"[{step}] val_loss={metrics.get('val/total_loss', float('nan')):.4f}"

        task_names = list(self.model.config.tasks.keys())
        shown = []
        for t in task_names[:4]:
            k, v = self._key_metric_for_task(t, metrics)
            if k and v is not None:
                shown.append(f"{t}:{k.split('_')[-1]}={v:.4f}") 

        more = ""
        if len(task_names) > 4:
            more = f" (+{len(task_names) - 4} more)"

        if shown:
            print(base + " | " + " | ".join(shown) + more)
        else:
            print(base)

    def run(self):
        """Main training loop"""
        # setup the dataloader
        self.model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        
        print(f"Starting training for {self.tr_config.max_iters} iterations...")
        print(f"Training batches per epoch: {len(self.train_loader)}")

        while self.iter_num < self.tr_config.max_iters:
            for x_batch, x_mask, y_list, task_names in self.train_loader:
                # Move to device
                x_batch = x_batch.to(self.device)
                x_mask = x_mask.to(self.device)  
                y_list = [y.to(self.device) for y in y_list]

                # Forward -> Backprop and update parameters
                scaler = torch.cuda.amp.GradScaler(enabled=self.tr_config.amp and self.device.startswith("cuda"))
                with torch.cuda.amp.autocast(enabled=self.tr_config.amp and self.device.startswith("cuda")):
                    avg_loss, task_losses = self.model(x_batch, x_mask, y_list, task_names)
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(avg_loss).backward()
                scaler.unscale_(self.optimizer) # we have to make sure gradients are in their real scale

                # Get gradient statistics
                pre, _ = self._grad_stats()
                pre_norm = pre["grad_norm"]
                pre_max  = pre["grad_max_abs"]

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tr_config.grad_norm_clip)

                # post gradient clipping stats
                post, _ = self._grad_stats()
                post_norm = post["grad_norm"]
                post_max  = post["grad_max_abs"]

                # step optimizer ( change w and b )
                scaler.step(self.optimizer)
                scaler.update()

                # Logging
                if self.iter_num % self.tr_config.log_interval == 0:
                    metrics = {
                        "train/total_loss": avg_loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/grad_norm_pre": pre_norm,
                        "train/grad_max_abs_pre": pre_max,
                        "train/grad_norm_post": post_norm,
                        "train/grad_max_abs_post": post_max,
                        "train/grad_clipped": float(post_norm < pre_norm),  
                    }
                    
                    # task losses
                    for task, loss in task_losses.items():
                        metrics[f'train/{task}_loss'] = loss
                    
                    # distribution of gradient values
                    self._log_grad_histograms(self.iter_num, every=max(1000, self.tr_config.log_interval * 5), topk=20)

                    # Log to TensorBoard
                    self.log_metrics(metrics, self.iter_num, tag="train")

                # Evaluation
                if self.iter_num % tr_config.eval_interval == 0 and self.val_loader is not None:
                    val_metrics = self.evaluate()
                    if val_metrics:
                        self.log_metrics(val_metrics, self.iter_num, tag="val")
                        
                        # Save best model
                        val_loss = val_metrics['val/total_loss']
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(f"best_model_step_{self.iter_num}.pt")

                # Trigger callbacks
                self.trigger_callbacks('on_batch_end')
                
                # Update timing
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                # Check termination condition
                if self.iter_num >= tr_config.max_iters:
                    break
            
            # Break outer loop if max_iters reached
            if self.iter_num >= tr_config.max_iters:
                break

        # Final checkpoint
        self.save_checkpoint(f"final_model_step_{self.iter_num}.pt")
        
        # Close TensorBoard writer
        self.writer.close()
        print("Training completed!")


# let's run the training process

# Setup ml_config
dt_config = DatasetConfig(
    data_dir = "toy_data",
    toks_in_batch = 1000,
    batch_size = 16,
    data_stats_path = "data_stats.json",
)
ml_config = GPTConfig()
tr_config = TrainConfig(
    num_workers=4,
    max_iters=1_00_000,
    learning_rate=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    grad_norm_clip=2.0,
    log_interval=100,
    eval_interval=500,
    device= "cuda" if torch.cuda.is_available() else "cpu",
    amp=False, # stop using amp ( half precision training ) for now,
)

# Create tokenizer  
tokenizer = MyTokenizer()
ml_config.vocab_size = len(tokenizer.tokenizer)

# Create model
model = HRGPT(ml_config, tr_config)

# Load pretrained weights if available
model.get_pretrained_backbone()

# Create datasets
train_loader = BatchLoader(ml_config, "train", tokenizer, device=tr_config.device)
val_loader = BatchLoader(ml_config, "val", tokenizer, device=tr_config.device)

# Create trainer
trainer = Trainer(tr_config, model, train_loader, val_loader)

# Start training
trainer.run()
