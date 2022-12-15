# DD2412 Advanced Deep Learning

This project was carried out within the framework of the course DD2412 Advanced Deep Learning at KTH, Stockholm. In this project we aim to reproduce the work described in the paper <a href="https://arxiv.org/abs/2111.06377">Masked Autoencoders Are Scalable Vision Learners</a> with the <a href="https://github.com/google/jax">JAX</a> library in Python.

The necessary libraries to run the code are `Jax`, `Flax` and `Tensorflow-datasets`.

To run the training phase, use `python main.py`. The model is saved directly after the training.
When the training is complete, the model is ran once on the entire train and test sets and the average loss is returned.
Finally, the model is ran over one image and the results are plotted and save to a .png file.

To run a pretrained model, use `python test_mae.py`. The script loads a pre-trained model and runs it on N images from the train set and then on N images from the test. All the results are plot and saved to .png files.

Project members:
<ul>
<li><a href="https://github.com/maxellende">Maxellende Julienne</a></li>
<li><a href="https://github.com/Victor-Manach">Victor Manach</a></li>
<li><a href="https://github.com/SushiQ">David Nordmark</a></li>
</ul>
