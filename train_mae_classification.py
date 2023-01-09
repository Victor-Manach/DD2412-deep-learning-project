# Code copied from a Jax tutorial and adapted to work with our model and loss functions
# --------------------------------------------------------
# References:
# Jax tutorial: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
# --------------------------------------------------------

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state, checkpoints
import flax.linen as nn
from tqdm.auto import tqdm
from collections import defaultdict
from flax.core.frozen_dict import freeze

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/mae_classification/"

class TrainModule:
    def __init__(self, model, pretrained_encoder_vars, exmp_imgs, dataset_name, model_arch, num_epochs, num_steps_per_epoch, mask_ratio, seed=42):
        super().__init__()
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = model
        self.mask_ratio = mask_ratio
        # Prepare directory to save and load trained models
        self.log_dir = os.path.join(CHECKPOINT_PATH, dataset_name, model_arch, f"{num_epochs}_epochs")
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(pretrained_encoder_vars, exmp_imgs, num_epochs, num_steps_per_epoch)

    def create_functions(self):
        """ Initialize the functions needed to train and evaluate the model and jit those functions
        to speed up the computations.
        """
        # Function to calculate the classification loss and accuracy for a model
        def compute_loss(params, key, batch, train):
            imgs, labels = batch
            
            key, dropout_apply_rng, drop_path_apply_rng, masked_rng = jax.random.split(key, 4)
            logits = self.model.apply({"params": params}, x=imgs, mask_ratio=self.mask_ratio, train=train, key=masked_rng, rngs={"dropout": dropout_apply_rng, "drop_path": drop_path_apply_rng})
            
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            preds = jax.nn.one_hot(logits.argmax(axis=-1), labels.shape[1])
            acc = (preds == labels).mean()
            
            return loss, (acc, key)
        
        # Training function
        def train_step(state, batch, key):
            loss_fn = lambda params: compute_loss(params, key, batch, train=True)
            #t1 = time.time()
            (loss, (acc, key)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)  # Get loss and gradients for loss
            #print(f"(Train step) Time to compute the gradient of the loss func: {time.time()-t1:.4f}s")
            #t1 = time.time()
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            #print(f"(Train step) Time to update the gradient parameters: {time.time()-t1:.4f}s")
            return state, loss, acc, key
        self.train_step = jax.jit(train_step)

        # Eval function
        def eval_step(state, batch, key):
            loss, (acc, rng) = compute_loss(state.params, key, batch, train=False)
            return loss, acc, rng
        self.eval_step = jax.jit(eval_step)

    def init_model(self, pretrained_encoder_vars, exmp_imgs, num_epochs, num_steps_per_epoch):
        """ Initialize the MAE model with the proper random keys, the learning rate and the optimizer.
        """
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng, drop_path_init_rng, masking_rng = jax.random.split(self.rng, 5)
        variables = self.model.init({"params": init_rng, "dropout": dropout_init_rng, "drop_path": drop_path_init_rng}, x=exmp_imgs, mask_ratio=self.mask_ratio, train=True, key=masking_rng)
        
        params = variables['params']
        params = params.unfreeze()
        params["backbone"] = pretrained_encoder_vars["params"]
        params = freeze(params)
        
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=int(num_steps_per_epoch*num_epochs*0.6),
            decay_steps=500*(num_epochs+1),
            end_value=1e-5
        )
        
        partition_optimizers = {'trainable': optax.adamw(lr_schedule), 'frozen': optax.set_to_zero()}
        param_partitions = freeze(flax.traverse_util.path_aware_map(
            lambda path, v: 'frozen' if 'backbone' in path else 'trainable', params))
        
        tx = optax.multi_transform(partition_optimizers, param_partitions)
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    def train_model(self, train_data, val_data, num_epochs):
        """ Train the model for a given number of epochs.
        """
        metrics = defaultdict(list)
        pbar = tqdm(total=num_epochs)
        for epoch_idx in range(1, num_epochs+1):
            t1 = time.time()
            avg_loss, avg_acc = self.train_epoch(train_data=train_data, epoch=epoch_idx)
            pbar.set_description(f"Epoch {epoch_idx} | avg loss {avg_loss:.4f} | avg acc {avg_acc:.4f} | time {avg_acc:.4f} | {time.time()-t1:.2f}s")
            pbar.update(1)
            metrics["loss"].append(avg_loss)
            metrics["acc"].append(avg_acc)
        pbar.close()
        return np.asarray(metrics["loss"]), np.asarray(metrics["acc"])

    def train_epoch(self, train_data, epoch):
        """ Train model for one epoch, and log the average loss.
        """
        metrics = defaultdict(list)
        pbar = tqdm(total=len(train_data))
        for batch in train_data:
            #print("(Train epoch) Call the train_step inside train_epoch")
            t1 = time.time()
            self.state, loss, acc, self.rng = self.train_step(self.state, batch, self.rng)
            pbar.set_description(f"Epoch {epoch} | loss {loss:.4f} | acc {acc:.4f} | time {time.time()-t1:.2f}s")
            pbar.update(1)
            #print(f"(Train epoch) Finished train_step: {time.time()-t1:.4f}s")
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)
        pbar.close()
        
        avg_loss = np.stack(jax.device_get(metrics["loss"])).mean()
        avg_acc = np.stack(jax.device_get(metrics["acc"])).mean()
        #print(f"Epoch {epoch}: avg_train_loss={avg_loss:.4f}")
        return avg_loss, avg_acc

    def eval_model(self, data_loader):
        """ Test the model on all images of a data loader and return the average loss.
        """
        metrics = defaultdict(list)
        batch_sizes = []
        for batch in data_loader:
            loss, acc, self.rng = self.eval_step(self.state, batch, self.rng)
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(metrics["loss"]))
        accs_np = np.stack(jax.device_get(metrics["acc"]))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        avg_acc = (accs_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss, avg_acc

    def save_model(self, step=0):
        """ Save current model at certain training iteration.
        """
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, step=step, overwrite=True)
    
    def load_model(self):
        """ Load a saved pre-trained model.
        """
        params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)