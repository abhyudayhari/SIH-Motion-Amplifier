import tensorflow as tf


def read_and_decode(filename_queue, im_size=(512, 512, 1)):
    writeOpts = tf.io.TFRecordOptions(compression_type='ZLIB')
    reader = tf.TFRecordReader(options=writeOpts)
    _, single_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      single_example,
      features={
        'frameA': tf.io.FixedLenFeature([], tf.int64),
        'frameB': tf.io.FixedLenFeature([], tf.int64),
        'amplified': tf.io.FixedLenFeature([], tf.int64),
        'amplification_factor': tf.io.FixedLenFeature([], tf.float32),
        })
    frameA = tf.decode_raw(features['frameA'], tf.uint8)
    frameB = tf.decode_raw(features['frameB'], tf.uint8)
    frameAmp = tf.decode_raw(features['amplified'], tf.uint8)
    amplification_factor = tf.cast(features['amplification_factor'], tf.float32)

    frameA = tf.reshape(frameA, im_size)
    frameB = tf.reshape(frameB, im_size)
    frameAmp = tf.reshape(frameAmp, im_size)

    # Normalize to -1 to +1
    frameA = tf.to_float(frameA) / 127.5 - 1.0
    frameB = tf.to_float(frameB) / 127.5 - 1.0
    frameAmp = tf.to_float(frameAmp) / 127.5 - 1.0

    return frameA, frameB, frameAmp, amplification_factor

def read_and_decode_3frames(filename_queue, im_size=(512, 512, 1)):
    writeOpts = tf.io.TFRecordOptions(compression_type='ZLIB')
    reader = tf.compat.v1.TFRecordReader(options=writeOpts)
    _, single_example = reader.read(filename_queue)
    features = tf.compat.v1.parse_single_example(
      single_example,
      features={
        'frameA': tf.io.FixedLenFeature([], tf.int64),
        'frameB': tf.io.FixedLenFeature([], tf.int64),
        'frameC': tf.io.FixedLenFeature([], tf.int64),
        'amplified': tf.io.FixedLenFeature([], tf.int64),
        'amplification_factor': tf.io.FixedLenFeature([], tf.float32),
        })
    frameA = tf.image.decode_image(features['frameA'], dtype = tf.uint8)
    frameB = tf.image.decode_image(features['frameB'], dtype = tf.uint8)
    frameC = tf.image.decode_image(features['frameC'], dtype = tf.uint8)
    frameAmp = tf.image.decode_image(features['amplified'], dtype = tf.uint8)
    amplification_factor = tf.cast(features['amplification_factor'], tf.float32)

    frameA = tf.reshape(frameA, im_size)
    frameB = tf.reshape(frameB, im_size)
    frameC = tf.reshape(frameC, im_size)
    frameAmp = tf.reshape(frameAmp, im_size)

    # Normalize to -1 to +1
    frameA = tf.cast(frameA,tf.float32) / 127.5 - 1.0
    frameB = tf.cast(frameB,tf.float32) / 127.5 - 1.0
    frameC = tf.cast(frameC,tf.float32) / 127.5 - 1.0
    frameAmp = tf.cast(frameAmp,tf.float32) / 127.5 - 1.0

    return frameA, frameB, frameC, frameAmp, amplification_factor
