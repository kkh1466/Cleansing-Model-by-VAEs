import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"GPUs recognized by TensorFlow: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        print(f"GPU {i}: {gpu.name}, Details: {details}")
else:
    print("No GPU recognized by TensorFlow.")


gpu = tf.config.list_physical_devices('GPU')
print(gpu)
