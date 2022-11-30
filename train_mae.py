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

from mae import mae_loss, mae_norm_pix_loss

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"

class TrainModule:
    def __init__(self, model, train, exmp_imgs, dataset_name, seed=42, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = model
        # Prepare logging
        self.exmp_imgs = exmp_imgs
        self.log_dir = os.path.join(CHECKPOINT_PATH, dataset_name)
        self.dataset_name = dataset_name
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(train)

    def create_functions(self):
        # Training function
        if self.model.norm_pix_loss:
            loss_func = mae_norm_pix_loss
        else:
            loss_func = mae_loss
        def train_step(state, batch, key):
            loss_fn = lambda params: loss_func(model=self.model, params=params, x=batch, train=True, key=key)
            #t1 = time.time()
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)  # Get loss and gradients for loss
            loss, key = ret[0], ret[1]
            #print(f"(Train step) Time to compute the gradient of the loss func: {time.time()-t1:.4f}s")
            #t1 = time.time()
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            #print(f"(Train step) Time to update the gradient parameters: {time.time()-t1:.4f}s")
            return state, loss, key
        self.train_step = jax.jit(train_step)
        # Eval function
        def eval_step(state, batch, key):
            loss, rng = mae_loss(model=self.model, params=state.params, x=batch, train=False, key=key)
            return loss, rng
        self.eval_step = jax.jit(eval_step)

    def init_model(self, train_data):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng, masking_rng = jax.random.split(self.rng, 4)
        params = self.model.init({"params": init_rng, "dropout": dropout_init_rng}, x=self.exmp_imgs, train=True, key=masking_rng)["params"]
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=100,
            decay_steps=500*len(train_data),
            end_value=1e-5
        )
        optimizer = optax.chain(
            optax.clip(1.0),  # Clip gradients at 1
            optax.adam(lr_schedule)
        )
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)

    def train_model(self, train_data, val_data, num_epochs=500):
        # Train model for defined number of epochs
        best_eval = np.inf
        pbar = tqdm(total=num_epochs)
        for epoch_idx in range(1, num_epochs+1):
            t1 = time.time()
            avg_loss = self.train_epoch(train_data=train_data, epoch=epoch_idx)
            pbar.set_description(f"Epoch {epoch_idx} - avg loss {avg_loss:.4f} - train epoch time {time.time()-t1}s")
            pbar.update(1)
            if epoch_idx % 10 == 0:
                eval_loss = self.eval_model(val_data)
                #print(f"Epoch {epoch_idx}: val_loss={eval_loss:.4f}")
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    self.save_model(step=epoch_idx)
        pbar.close()
        return self.state.params

    def train_epoch(self, train_data, epoch):
        # Train model for one epoch, and log avg loss
        losses = []
        pbar = tqdm(total=len(train_data))
        for batch in train_data:
            #print("(Train epoch) Call the train_step inside train_epoch")
            t1 = time.time()
            self.state, loss, self.rng = self.train_step(self.state, batch, self.rng)
            pbar.set_description(f"Epoch {epoch} - train loss {loss:.4f} - train step time {time.time()-t1}s")
            pbar.update(1)
            #print(f"(Train epoch) Finished train_step: {time.time()-t1:.4f}s")
            losses.append(loss)
        pbar.close()
        losses_np = np.stack(jax.device_get(losses))
        avg_loss = losses_np.mean()
        #print(f"Epoch {epoch}: avg_train_loss={avg_loss:.4f}")
        return avg_loss

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss, self.rng = self.eval_step(self.state, batch, self.rng)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=self.dataset_name, step=step)
    
    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=self.dataset_name)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f"{self.dataset_name}.ckpt"), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f"{self.dataset_name}.ckpt"))