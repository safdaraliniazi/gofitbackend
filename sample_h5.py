import tensorflow as tf

# Load the MobileNetV2 model with pre-trained ImageNet weights
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Save the model as an .h5 file
model.save('mobilenet_v2.h5')

print("Model saved as 'mobilenet_v2.h5'")
