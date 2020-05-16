import tensorflow as tf
import random
import pathlib
import matplotlib.pyplot as plt
import os

# Data Sets set up

data_path = pathlib.Path('E:\\AMdata')
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())

label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

raw_train_image = all_image_paths[:round(image_count * 0.8)]
raw_train_label = all_image_labels[:round(image_count * 0.8)]
raw_train = tf.data.Dataset.from_tensor_slices((raw_train_image, raw_train_label))
raw_validation_image = all_image_paths[(round(image_count * 0.8) + 1):round(image_count * 0.9)]
raw_validation_label = all_image_labels[(round(image_count * 0.8) + 1):round(image_count * 0.9)]
raw_validation = tf.data.Dataset.from_tensor_slices((raw_validation_image, raw_validation_label))
raw_test_image = all_image_paths[(round(image_count * 0.9) + 1):]
raw_test_label = all_image_labels[(round(image_count * 0.9) + 1):]
raw_test = tf.data.Dataset.from_tensor_slices((raw_test_image, raw_test_label))

IMG_SIZE = 160  # All images will be resized to 160x160


def format_example(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 400
#
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# Model building and training
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# # Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


prediction_layer = tf.keras.layers.Dense(3, activation=tf.nn.softmax)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

base_learning_rate = 0.0001

initial_epochs = 10

base_model.trainable = True


model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              metrics=['accuracy'])

checkpoint_path = 'cp1.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch', save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches,
                    validation_steps=2,
                    steps_per_epoch=len(raw_train_label) // BATCH_SIZE,
                    callbacks=[cp_callback])

model.load_weights('cp1.ckpt')
loss1, acc = model.evaluate(test_batches, verbose=2)
print("restored model,accuracy: {:5.2f}%".format(100 * acc))

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
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
