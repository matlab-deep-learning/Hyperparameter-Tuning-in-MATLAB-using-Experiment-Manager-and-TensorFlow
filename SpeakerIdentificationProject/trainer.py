import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train(x_train, y_train, x_test, y_test, params, model_file):
    numEpoch = int(params['numEpoch'])
    miniBatchSize = int(params['miniBatchSize'])
    learnRate = params['learnRate']

    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    train_res = model.fit(x_train, y_train, batch_size=miniBatchSize, epochs=numEpoch, verbose=0)
    test_res = model.evaluate(x_test, y_test, verbose=0)
    model.save(model_file)
    return (train_res, test_res)

def create_model():
    imageinput = keras.Input(shape=(1,3200,1), name="imageinput")
    conv_1 = layers.Conv2D(80, (1,251), name="conv_1")(imageinput)
    batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1")(conv_1)
    leakyrelu_1 = layers.LeakyReLU(alpha=0.200000)(batchnorm_1)
    maxpool_1 = layers.MaxPool2D(pool_size=(1,3), strides=(1,1))(leakyrelu_1)
    conv_2 = layers.Conv2D(60, (1,5), name="conv_2")(maxpool_1)
    batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2")(conv_2)
    leakyrelu_2 = layers.LeakyReLU(alpha=0.200000)(batchnorm_2)
    maxpool_2 = layers.MaxPool2D(pool_size=(1,3), strides=(1,1))(leakyrelu_2)
    conv_3 = layers.Conv2D(60, (1,5), name="conv_3")(maxpool_2)
    batchnorm_3 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_3")(conv_3)
    leakyrelu_3 = layers.LeakyReLU(alpha=0.200000)(batchnorm_3)
    maxpool_3 = layers.MaxPool2D(pool_size=(1,3), strides=(1,1))(leakyrelu_3)
    fc_1 = layers.Reshape((-1,), name="fc_1_preFlatten1")(maxpool_3)
    fc_1 = layers.Dense(256, name="fc_1")(fc_1)
    batchnorm_4 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_4")(fc_1)
    leakyrelu_4 = layers.LeakyReLU(alpha=0.200000)(batchnorm_4)
    fc_2 = layers.Dense(256, name="fc_2")(leakyrelu_4)
    batchnorm_5 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_5")(fc_2)
    leakyrelu_5 = layers.LeakyReLU(alpha=0.200000)(batchnorm_5)
    fc_3 = layers.Dense(256, name="fc_3")(leakyrelu_5)
    batchnorm_6 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_6")(fc_3)
    leakyrelu_6 = layers.LeakyReLU(alpha=0.200000)(batchnorm_6)
    fc_4 = layers.Dense(6, name="fc_4")(leakyrelu_6)
    softmax = layers.Softmax()(fc_4)

    model = keras.Model(inputs=[imageinput], outputs=[softmax])
    return model