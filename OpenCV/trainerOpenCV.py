from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from glob import glob

training_source = 'data/datasets/raw_datasets'
validation_source = 'data/datasets/process_datasets'

network = VGG16(
  input_shape=[224, 224] + [3],
  weights='imagenet',
  include_top=False
)

for l in network.layers:
  l.trainable = False 

folders = glob('data/datasets/raw_datasets/*')

flat = Flatten()(network.output)

prediction = Dense(len(folders), activation='softmax')(flat)

model = Model(
  inputs=network.input, 
  outputs=prediction
)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

generate_train_data = ImageDataGenerator(
  rescale = 1./255, 
  shear_range = 0.2, 
  zoom_range = 0.2, 
  horizontal_flip = True
)

generate_test_data = ImageDataGenerator(rescale = 1./255)

training_dataset = generate_train_data.flow_from_directory(
  training_source, 
  target_size = (224, 224), 
  batch_size = 32, 
  class_mode = 'categorical'
)

test_dataset = generate_test_data.flow_from_directory(
  validation_source, 
  target_size = (224, 224), 
  batch_size = 32, 
  class_mode = 'categorical'
)

model.fit_generator(
  training_dataset, 
  validation_data=test_dataset, 
  epochs=7, 
  steps_per_epoch=len(training_dataset), 
  validation_steps=len(test_dataset)
)

model.summary()

model.save('trained_face_model.h5')

