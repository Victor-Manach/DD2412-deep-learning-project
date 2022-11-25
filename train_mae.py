import os
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
        init_key = jax.random.PRNGKey(self.seed)
        self.init_model(train, init_key)

    def create_functions(self):
        # Training function
        if self.model.norm_pix_loss:
            loss_func = mae_norm_pix_loss
        else:
            loss_func = mae_loss
        def train_step(state, batch, key):
            loss_fn = lambda params: loss_func(model=self.model, params=params, x=batch, train=True, key=key)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)  # Get loss and gradients for loss
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            return state, loss
        self.train_step = jax.jit(train_step)
        # Eval function
        def eval_step(state, batch, key):
            return mae_loss(model=self.model, params=state.params, x=batch, train=False, key=key)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, train_data, key):
        # Initialize model
        init_key, rng_key, dropout_init_key = jax.random.split(key, 3)
        params = self.model.init({"params": init_key, "dropout": dropout_init_key}, x=self.exmp_imgs, train=True, key=rng_key)["params"]
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

    def train_model(self, train_data, val_data, key, num_epochs=500):
        # Train model for defined number of epochs
        best_eval = 1e6
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            self.train_epoch(train_data=train_data, epoch=epoch_idx, key=key)
            if epoch_idx % 10 == 0:
                eval_loss = self.eval_model(val_data, key)
                print(f"Epoch {epoch_idx}: val_loss={eval_loss:.5f}")
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    self.save_model(step=epoch_idx)
        return self.state.params

    def train_epoch(self, train_data, epoch, key):
        # Train model for one epoch, and log avg loss
        losses = []
        for batch in train_data:
            self.state, loss = self.train_step(self.state, batch, key)
            losses.append(loss)
        losses_np = np.stack(jax.device_get(losses))
        avg_loss = losses_np.mean()
        print(f"Epoch {epoch}: avg_train_loss={avg_loss:.5f}")

    def eval_model(self, data_loader, key):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss = self.eval_step(self.state, batch, key)
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