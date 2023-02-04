# DD2412 Advanced Deep Learning

This project was carried out within the framework of the course DD2412 Advanced Deep Learning at KTH, Stockholm. In this project we aim to reproduce the work described in the paper <a href="https://arxiv.org/abs/2111.06377">Masked Autoencoders Are Scalable Vision Learners</a> with the <a href="https://github.com/google/jax">JAX</a> library in Python.

The necessary libraries to run the code are `Jax`, `Flax` and `Tensorflow-datasets`.

The `main_pretrain.py` file can be used to run the pretraining of the model and save that model at the end. Then, the `main_mae_classification.py` file can be used to start the fine-tuning of a pre-trained MAE model. Finally, the files `test_mae.py` (resp. `test_mae_classification.py`) can be used to run a pre-trained model (resp. fine-tuned model) on a set of images from the train and test sets.

Project members:
<ul>
<li><a href="https://github.com/maxellende">Maxellende Julienne</a></li>
<li><a href="https://github.com/Victor-Manach">Victor Manach</a></li>
<li><a href="https://github.com/SushiQ">David Nordmark</a></li>
</ul>
