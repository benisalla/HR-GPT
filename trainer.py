import os
import time
import torch
from collections import defaultdict
from model.utils import CfgNode as CN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.ml_config import GPTConfig, TrainConfig
from model.model import HRGPT
from dataset.tokenizer import MyTokenizer
from dataset.BatchLoader import BatchLoader


class Trainer:
    def __init__(self, tr_config, model, train_loader, val_loader=None):
        self.tr_config = tr_config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = model.get_optimizer(tr_config)
        self.callbacks = defaultdict(list)
        self.device = tr_config.device
        
        self.model = self.model.to(self.device)
        print(f"Running on device: {self.device}")

        # variables for logging and tracking
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.best_val_loss = float('inf')

        # TensorBoard logging
        log_dir = os.path.join(tr_config.log_dir, tr_config.experiment_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model on validation set"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_loader = DataLoader(
            self.val_loader,
            batch_size=self.tr_config.batch_size,
            shuffle=False,
            num_workers=self.tr_config.num_workers,
            pin_memory=True
        )
        
        total_loss = 0.0
        task_losses = defaultdict(list)
        num_batches = 0
        
        for batch_data in val_loader:
            x_batch, x_mask, y_list, task_names = batch_data
            
            # Move to device
            x_batch = x_batch.to(self.device)
            x_mask = x_mask.to(self.device)
            y_list = [y.to(self.device) for y in y_list]
            
            # Forward pass
            avg_loss, batch_task_losses = self.model(x_batch, x_mask, y_list, task_names)
            
            total_loss += avg_loss.item()
            for task, loss in batch_task_losses.items():
                task_losses[task].append(loss)
            num_batches += 1
        
        # Compute averages
        val_metrics = {
            'val/total_loss': total_loss / num_batches,
        }
        
        for task, losses in task_losses.items():
            val_metrics[f'val/{task}_loss'] = sum(losses) / len(losses)
        
        self.model.train()
        return val_metrics

    def log_metrics(self, metrics, step):
        """Log metrics to TensorBoard and console"""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        
        # Console logging
        if step % self.tr_config.log_interval == 0:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Step {step} | {metrics_str} | dt: {self.iter_dt:.3f}s")

    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'iter_num': self.iter_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tr_config': self.tr_config.to_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Checkpoint loaded from {filepath}")

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

                # Forward pass
                avg_loss, task_losses = model(x_batch, x_mask, y_list, task_names)
                self.loss = avg_loss.item()

                # Backprop and update parameters
                self.optimizer.zero_grad(set_to_none=True)
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr_config.grad_norm_clip)
                self.optimizer.step()

                # Logging
                if self.iter_num % tr_config.log_interval == 0:
                    metrics = {
                        'train/total_loss': avg_loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    }
                    
                    # task losses
                    for task, loss in task_losses.items():
                        metrics[f'train/{task}_loss'] = loss
                    
                    self.log_metrics(metrics, self.iter_num)

                # Evaluation
                if self.iter_num % tr_config.eval_interval == 0 and self.val_loader is not None:
                    val_metrics = self.evaluate()
                    if val_metrics:
                        self.log_metrics(val_metrics, self.iter_num)
                        
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
ml_config = GPTConfig()
tr_config = TrainConfig(
    num_workers=4,
    batch_size=16,
    max_iters=1000,
    learning_rate=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    grad_norm_clip=1.0,
    log_interval=100,
    eval_interval=500,
    log_dir="runs",
    experiment_name="hr_gpt_training",
    device= "cuda" if torch.cuda.is_available() else "cpu",
    amp=True,
)

# Create tokenizer  
tokenizer = MyTokenizer("gpt2")
ml_config.vocab_size = len(tokenizer.tokenizer)

# Create model
model = HRGPT(ml_config)

# Create datasets
train_loader = BatchLoader(ml_config, "train", tokenizer, device="cpu")
val_loader = BatchLoader(ml_config, "val", tokenizer, device="cpu")

# Create trainer
trainer = Trainer(tr_config, model, train_loader, val_loader)

# Start training
trainer.run()
