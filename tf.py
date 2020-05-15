import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow_datasets as tfds
import pathlib
import random
from sklearn.model_selection import train_test_split

tfds.disable_progress_bar()

data_path = pathlib.Path('./data')
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]  # 所有图片路径的列表
random.shuffle(all_image_paths)  # 打散

image_count = len(all_image_paths)

label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())

label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

train_image_paths =[]
validation_image_paths = []
test_image_paths = []

for i in range(0, 5000):
    train_image_paths.append(all_image_paths[i])

for i in range(5001, 6000):
    validation_image_paths.append(all_image_paths[i])

for i in range(6000, len(all_image_paths)):
    test_image_paths.append(all_image_paths[i])

train_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_image_paths]
validation_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in validation_image_paths]
test_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_image_paths]


raw_train = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
raw_validation = tf.data.Dataset.from_tensor_slices((validation_image_paths, validation_image_labels))
raw_test = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))

# raw_train = tfds.core.ReadInstruction('ds', to=80, unit='%')
# raw_validation = tfds.core.ReadInstruction('ds', from_=80, to=90, unit='%')
# raw_test = tfds.core.ReadInstruction('ds', from_=90, unit='%')


# format the data
IMG_SIZE = 160  # All images will be resized to 160x160

def format_example(path, label):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  # 原始图片大小为(266, 320, 3)，重设为(160, 160)
    image /= 255.0  # 归一化到[0,1]范围
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# imspect a batch of data

for image_batch, label_batch in train_batches.take(1):
   pass

image_batch.shape


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape)

# freeze the convolutional base
base_model.trainable = False
base_model.summary()

# add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(3)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# compile the model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

len(model.trainable_variables)

# Train the model
initial_epochs = 10
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

# learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# unfreeze the top layers of the model
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

# continue training the model
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=validation_batches)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

