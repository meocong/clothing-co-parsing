from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from keras.utils import Sequence
import numpy as np
from online_augment import augment
import glob
import cv2
from sklearn.model_selection import train_test_split

class DataGenerator(Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[
                  idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[
                  idx * self.batch_size:(idx + 1) * self.batch_size]

        # return np.array([
        #     resize(imread(file_name), (200, 200))
        #     for file_name in batch_x]), np.array(batch_y)
        temp = [augment(x,y) for (x,y) in zip(batch_x, batch_y)]
        return np.array([x[0] for x in temp]), np.array([x[1] for x in temp])


model = Unet(backbone_name='resnet18', encoder_weights='imagenet', freeze_encoder=True)
model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

mask_images = glob.glob("./mask")
X = [cv2.imread(x.replace("mask","photos")) for x in mask_images]
y = [cv2.imread(x) for x in mask_images]


X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.1, random_state=42)
batch_size = 1
my_training_batch_generator = DataGenerator(X_train, Y_train, batch_size)
my_validation_batch_generator = DataGenerator(X_val, Y_val, batch_size)

# pretrain model decoder
model.fit_generator(generator=my_training_batch_generator,
                    epochs=2,
                    steps_per_epoch = len(X_train) // batch_size,
                    validation_data=my_validation_batch_generator,
                    verbose=1,
                    validation_steps=len(X_val) // batch_size)
# model.fit(X, y, epochs=2)
model.save('./model/2ndepoch_model.h5')

# release all layers for training
set_trainable(model) # set all layers trainable and recompile model

# continue training
model.fit_generator(generator=my_training_batch_generator,
                    epochs=100,
                    steps_per_epoch = len(X_train) // batch_size,
                    validation_data=my_validation_batch_generator,
                    verbose=1,
                    validation_steps=len(X_val) // batch_size)
# model.fit(X, y, epochs=100)
model.save("./model/102thepoch_model.h5")
