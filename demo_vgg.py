import tensorflow as tf
import numpy as np
import os
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from universal_pert import universal_perturbation

device = '/gpu:0'
num_classes = 1000  # VGG16 pretrained on ImageNet has 1000 output classes
path_train_imagenet = 'data/train'  # Change if using another dataset
path_test_image = 'data/test_img.png'
file_perturbation = 'data/universal_vgg.npy'

with tf.device(device):
    print('>> Loading VGG16 model...')
    model = VGG16(weights='imagenet')
    input_tensor = model.input
    output_tensor = model.output

    def f(image_inp):
        image_inp = preprocess_input(image_inp.copy())
        return model(image_inp, training=False).numpy()

    if not os.path.isfile(file_perturbation):
        print('>> Compiling gradient function...')
        def grad_fs(image_inp, indices):
            image_inp = tf.convert_to_tensor(preprocess_input(image_inp.copy()), dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(image_inp)
                preds = model(image_inp, training=False)
                grads = [tape.gradient(preds[:, i], image_inp) for i in indices]
            grads = [g for g in grads if g is not None]
            return tf.stack(grads).numpy() if grads else None

        datafile = 'data/imagenet_data.npy'
        if not os.path.isfile(datafile):
            print('>> Creating ImageNet batch...')
            X = create_imagenet_npy(path_train_imagenet)
            os.makedirs('data', exist_ok=True)
            np.save(datafile, X)
        else:
            print('>> Loading preprocessed ImageNet data...')
            X = np.load(datafile)

        print('>> Running universal perturbation...')
        v = universal_perturbation(X, f, grad_fs, delta=0.2, num_classes=num_classes)
        np.save(file_perturbation, v)
    else:
        print('>> Found existing universal perturbation')
        v = np.load(file_perturbation)

    print('>> Testing on sample image')
    image_original = preprocess_image_batch([path_test_image], img_size=(256, 256), crop_size=(224, 224))
    label_original = np.argmax(f(image_original), axis=1)[0]

    perturbed = image_original + v
    label_perturbed = np.argmax(f(perturbed), axis=1)[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(undo_image_avg(image_original[0]).astype('uint8'))
    plt.title(f"Original Class: {label_original}")

    plt.subplot(1, 2, 2)
    plt.imshow(undo_image_avg(perturbed[0]).astype('uint8'))
    plt.title(f"Perturbed Class: {label_perturbed}")
    plt.show()
