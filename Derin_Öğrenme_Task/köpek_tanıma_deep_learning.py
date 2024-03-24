
import tensorflow as tf
from pathlib import Path
import os
import scipy
import random
from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import random
from shutil import copy2
from tqdm.auto import tqdm

train_list = scipy.io.loadmat("train_list.mat")
test_list = scipy.io.loadmat("test_list.mat")
file_list = scipy.io.loadmat("file_list.mat")

train_file_list = list([item[0][0] for item in train_list["file_list"]])
test_file_list = list([item[0][0] for item in test_list["file_list"]])
full_file_list = list([item[0][0] for item in file_list["file_list"]])


def count_subfolders(directory_path: str) -> int:
    return len([name for name in Path(directory_path).iterdir() if name.is_dir()])


directory_path = "Annotation"
folder_count = count_subfolders(directory_path)

image_folders = os.listdir("Images")

folder_to_class_name_dict = {}
for folder_name in image_folders:
  class_name = "_".join(folder_name.split("-")[1:]).lower()
  folder_to_class_name_dict[folder_name] = class_name


dog_names = sorted(list(folder_to_class_name_dict.values()))


def count_images_in_subdirs(target_directory: str) -> List[Dict[str, int]]:
    images_dir = Path(target_directory)
    image_class_dirs = [directory for directory in images_dir.iterdir() if directory.is_dir()]
    image_class_counts = []

    for image_class_dir in image_class_dirs:


        class_name = image_class_dir.stem

        image_count = len(list(image_class_dir.rglob("*.jpg")))
        image_class_counts.append({"class_name": class_name,
                                   "image_count": image_count})


    return image_class_counts
image_class_counts = count_images_in_subdirs("Images")
images_split_dir = Path("images_split")

train_dir = images_split_dir / "train"
test_dir = images_split_dir / "test"

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

for dog_name in dog_names:

  train_class_dir = train_dir / dog_name
  train_class_dir.mkdir(parents=True, exist_ok=True)


  test_class_dir = test_dir / dog_name
  test_class_dir.mkdir(parents=True, exist_ok=True)





# 1. Take in a list of source files to copy and a target directory
def copy_files_to_target_dir(file_list: list[str],
                             target_dir: str,
                             images_dir: str = "Images",
                             verbose: bool = False) -> None:
    for file in tqdm(file_list):

      source_file_path = Path(images_dir) / Path(file)


      file_class_name = folder_to_class_name_dict[Path(file).parts[0]]

      file_image_name = Path(file).name

      destination_file_path = Path(target_dir) / file_class_name / file_image_name

      destination_file_path.parent.mkdir(parents=True, exist_ok=True)

      if verbose:
        print(f"[INFO] Copying: {source_file_path} to {destination_file_path}")
      copy2(src=source_file_path, dst=destination_file_path)


copy_files_to_target_dir(file_list=train_file_list,
                         target_dir=train_dir,
                         verbose=False)

copy_files_to_target_dir(file_list=test_file_list,
                         target_dir=test_dir,
                         verbose=False)


train_image_paths = list(train_dir.rglob("*.jpg"))
test_image_paths = list(test_dir.rglob("*.jpg"))




IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42




train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)


test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=test_dir,
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False, # don't need to shuffle the test dataset (this makes evaluations easier)
    seed=SEED
)


image_batch, label_batch = next(iter(train_ds))
image_batch.shape, label_batch.shape

class_names = train_ds.class_names



AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

INPUT_SHAPE = (224,224,3)
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=INPUT_SHAPE,
    pooling="avg", 
    include_preprocessing=True,
)


base_model.trainable = False

count_parameters(model=base_model, print_output=True)
feature_vector_2 = base_model(tf.expand_dims(image_batch[0], axis=0))

tf.random.set_seed(42)
sequential_model = tf.keras.Sequential([base_model, # input and middle layers
                                        tf.keras.layers.Dense(units=len(dog_names), # output layer
                                                              activation="softmax")])

single_image_input = tf.expand_dims(image_batch[0], axis=0)

single_image_output_sequential = sequential_model(single_image_input)

np.sum(single_image_output_sequential)

highest_value_index_sequential_model_output = np.argmax(single_image_output_sequential)
highest_value_sequential_model_output = np.max(single_image_output_sequential)

sequential_model_predicted_label = class_names[tf.argmax(sequential_model(tf.expand_dims(image_batch[0], axis=0)), axis=1).numpy()[0]]

single_image_ground_truth_label = class_names[tf.argmax(label_batch[0])]
inputs = tf.keras.Input(shape=INPUT_SHAPE)
x = base_model(inputs, training=False)


outputs = tf.keras.layers.Dense(units=len(class_names), # one output per class
                                activation="softmax",
                                name="output_layer")(x)
functional_model = tf.keras.Model(inputs=inputs,
                                  outputs=outputs,
                                  name="functional_model")


functional_model.summary()


single_image_output_functional = functional_model(single_image_input)

highest_value_index_functional_model_output = np.argmax(single_image_output_functional)
highest_value_functional_model_output = np.max(single_image_output_functional)

highest_value_index_functional_model_output, highest_value_functional_model_output

def create_model(include_top: bool = False,
                 num_classes: int = 1000,
                 input_shape: tuple[int, int, int] = (224, 224, 3),
                 include_preprocessing: bool = True,
                 trainable: bool = False,
                 dropout: float = 0.2,
                 model_name: str = "model") -> tf.keras.Model:

  base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=include_top,
    weights="imagenet",
    input_shape=input_shape,
    include_preprocessing=include_preprocessing,
    pooling="avg"
  )

  base_model.trainable = trainable
  inputs = tf.keras.Input(shape=input_shape, name="input_layer")
  x = base_model(inputs, training=trainable)
  outputs = tf.keras.layers.Dense(units=num_classes,
                                  activation="softmax",
                                  name="output_layer")(x)
  model = tf.keras.Model(inputs=inputs,
                         outputs=outputs,
                         name=model_name)

  return model


def plot_model_loss_curves(history: tf.keras.callbacks.History) -> None:


  acc = history.history["accuracy"]
  val_acc = history.history["val_accuracy"]


  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  epochs_range = range(len(acc))
  plt.figure(figsize=(14, 7))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label="Training Accuracy")
  plt.plot(epochs_range, val_acc, label="Validation Accuracy")
  plt.legend(loc="lower right")
  plt.title("Training and Validation Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label="Training Loss")
  plt.plot(epochs_range, val_loss, label="Validation Loss")
  plt.legend(loc="upper right")
  plt.title("Training and Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")

  plt.show()

plot_model_loss_curves(history=history_0)


def tahminleri_degerlendir(ytrue, ypred):
  # Hesaplama yapmak için değerlerin float32 biçiminde olduğuna emin olalım.
  ytrue = tf.cast(ytrue, dtype = tf.float32)
  ypred = tf.cast(ypred, dtype = tf.float32)

  # Metrikleri kullanarak hesaplama yapmak

  mae = tf.keras.metrics.mean_absolute_error(ytrue, ypred)
  mse = tf.keras.metrics.mean_squared_error(ytrue, ypred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(ytrue, ypred)
  x = tf.reduce_mean(tf.abs(ytrue[1:] - ytrue[:-1]))
  mase = mae / x

  if mae.ndim > 0:
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)
    rmse = tf.reduce_mean(rmse)
    mape = tf.reduce_mean(mape)
    mase = tf.reduce_mean(mase)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}


model_1 = create_model(num_classes=120,
                       model_name="model_1")

model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"])

epochs=5
history_1 = model_1.fit(x=train_ds,
                        epochs=epochs,
                        validation_data=test_ds)
model_1.save("model.h5")

plot_model_loss_curves(history=history_1)
model_1_results = model_1.evaluate(test_ds)

test_preds = model_1.predict(test_ds)


import numpy as np
test_ds_images = np.concatenate([images for images, labels in test_ds], axis=0)
test_ds_labels = np.concatenate([labels for images, labels in test_ds], axis=0)
test_ds_labels[0], test_ds_images[0]

preds = model_1.predict(test_ds_images)

def tahminleri_degerlendir(ytrue, ypred):
  # Hesaplama yapmak için değerlerin float32 biçiminde olduğuna emin olalım.
  ytrue = tf.cast(ytrue, dtype = tf.float32)
  ypred = tf.cast(ypred, dtype = tf.float32)

  # Metrikleri kullanarak hesaplama yapmak

  mae = tf.keras.metrics.mean_absolute_error(ytrue, ypred)
  mse = tf.keras.metrics.mean_squared_error(ytrue, ypred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(ytrue, ypred)
  x = tf.reduce_mean(tf.abs(ytrue[1:] - ytrue[:-1]))
  mase = mae / x

  if mae.ndim > 0:
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)
    rmse = tf.reduce_mean(rmse)
    mape = tf.reduce_mean(mape)
    mase = tf.reduce_mean(mase)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}
print(tahminleri_degerlendir(test_ds_labels, preds))