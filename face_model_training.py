from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from glob import glob

train_source = 'data/datasets/raw_datasets'
valid_source = 'data/datasets/process_datasets'

network = VGG16(input_shape=[224, 224] + [3], weights='imagenet', include_top=False)

for layer in network.layers:
  layer.trainable = False 

folders = glob('data/datasets/raw_datasets/*')

flattened = Flatten()(network.output)

prediction = Dense(len(folders), activation='softmax')(flattened)

model = Model(inputs=network.input, outputs=prediction)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

datagenerator_train = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

datagenerator_test = ImageDataGenerator(rescale = 1./255)

training_dataset = datagenerator_train.flow_from_directory(train_source, target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

test_dataset = datagenerator_test.flow_from_directory(valid_source, target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

model.fit_generator(training_dataset, validation_data=test_dataset, epochs=7, steps_per_epoch=len(training_dataset), validation_steps=len(test_dataset))

model.summary()

model.save('trained_face_model.h5')

