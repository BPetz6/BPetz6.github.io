import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import Layer
import os
import gc

gc.enable()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
np.set_printoptions(linewidth=100000)

data_dir = os.path.join(os.getenv('SLURM_TMPDIR'), 'data')
# Construct the full path to the CSV file
csv_file_path = os.path.join(data_dir, 'jp_female_mr1.csv')
df = pd.read_csv(csv_file_path)
data = df

models_dir = os.path.join(os.getenv('SLURM_TMPDIR'), 'models')
#
# path = "/Users/braedonpetz/Downloads/jp_female_mr1.csv"
# df = pd.read_csv(path)
# data = df
# models_dir = "/Users/braedonpetz/Downloads"


def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=31000)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


configure_gpu()

YEARS = 50
AGES = 101
REGIONS = 47

prefectures = [
    "Hokkaido", "Aomori", "Iwate", "Miyagi", "Akita", "Yamagata", "Fukushima",
    "Ibaraki", "Tochigi", "Gunma", "Saitama", "Chiba", "Tokyo", "Kanagawa",
    "Niigata", "Toyama", "Ishikawa", "Fukui", "Yamanashi", "Nagano", "Gifu",
    "Shizuoka", "Aichi", "Mie", "Shiga", "Kyoto", "Osaka", "Hyogo", "Nara",
    "Wakayama", "Tottori", "Shimane", "Okayama", "Hiroshima", "Yamaguchi",
    "Tokushima", "Kagawa", "Ehime", "Kochi", "Fukuoka", "Saga", "Nagasaki",
    "Kumamoto", "Oita", "Miyazaki", "Kagoshima", "Okinawa"
]


def create_year_tuples(dataset):
    """
    Creates a tuple of the form (y-x, z-y) for each year in the dataset.
    Also returns an array containing all the first elements of the tuples, and one containing all the second elements.
    Also returns the original data for the first and second elements of the tuples so we can add the predictions.
    :param dataset: The dataset to create the tuples from
    """
    unique_ages = sorted(dataset['year'].unique())
    tupleSet = []
    inputs = []
    targets = []
    reginputs = []
    regtargets = []
    for k in range(len(unique_ages) - 2):
        x = dataset[dataset['year'] == unique_ages[k]]
        y = dataset[dataset['year'] == unique_ages[k + 1]]
        z = dataset[dataset['year'] == unique_ages[k + 2]]
        x = x['lograte'].values
        x = x.reshape((101,47,1))
        y = y['lograte'].values
        y = y.reshape((101,47,1))
        z = z['lograte'].values
        z = z.reshape((101,47,1))
        tupleSet.append((y-x, z-y))
        inputs.append(y-x)
        targets.append(z-y)
        reginputs.append(y)
        regtargets.append(z)
    return tupleSet, tf.convert_to_tensor(inputs,dtype=tf.float32), tf.convert_to_tensor(targets,dtype=tf.float32), reginputs, regtargets

def loss1(y_true, y_pred):
    '''
    :param y_true: The validaton data
    :param y_pred: the training data
    :return: the average mean squared error
    '''
    spe = tf.reduce_mean(tf.square(x=y_pred - y_true))
    return spe


def loss2(y_true, y_pred):
    '''
    :param y_true: The validaton data
    :param y_pred: the training data
    :return: the average mean squared error
    '''
    spe = tf.reduce_mean(tf.square(x=y_pred - y_true))
    return spe

def penalty(y_true, y_pred):
    '''
    :param y_true: The validaton data
    :param y_pred: the training data
    :return: the average mean squared error, multiplied by the choice of lambda
    '''
    spe = tf.reduce_mean(tf.square(x=y_pred - y_true))
    return 1*spe


class TileLayer(Layer):
    '''
    This layer is used to tile the input tensor so it can be imported into psi as timesteps
    '''
    def call(self, x):
        return tf.tile(tf.expand_dims(x, axis=-3), [1, AGES, 1,1])

class TransformerLayer(Layer):
    '''
    This layer is used to create a transformer layer, which is a building block in Transformer models.
    tfm.nlp.layers.TransformerEncoderBlock would also cover this, but then I would have to mess with the package requirements
    '''
    def __init__(self, embed_dim, num_heads, ff_dim):
        # Call the parent class's initializer
        super(TransformerLayer, self).__init__()

        # Initialize a multi-head self-attention layer
        # num_heads: the number of attention heads
        # key_dim: the dimension of the embedding space (embed_dim)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        # Define a feedforward neural network (FFN) as a sequential model
        # The FFN consists of two dense layers:
        # - The first Dense layer has ff_dim units
        # - The second Dense layer maps the output back to the original embedding dimension (embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="sigmoid"), layers.Dense(embed_dim)]
        )

        # Define two layer normalization layers
        # Layer normalization stabilizes and accelerates training by normalizing the inputs across the features dimension
        # epsilon=1e-6 is a small constant added to the variance to avoid division by zero
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply the multi-head self-attention mechanism to the inputs
        # The inputs attend to themselves to capture relationships within the sequence
        attn_output = self.att(inputs, inputs)

        # Apply a residual connection by adding the attention output to the original inputs
        # Follow this by layer normalization to stabilize the training process
        out1 = self.layernorm1(inputs + attn_output)

        # Pass the normalized output through the feedforward network (FFN)
        ffn_output = self.ffn(out1)

        # Apply another residual connection by adding the FFN output to the previous output (out1)
        # Follow this by a second layer normalization and return the result
        return self.layernorm2(out1 + ffn_output)


batch_size = 32
train_dataset = data[data['year']<=2002]
test_dataset = data[(data['year']>2002) & (data['year']<=2012)]
val_dataset = data[data['year']>2012]
# num_units = int(os.environ.get('UNIT'))
# num_layers = int(os.environ.get('LAYER'))
num_units = 256 # Higher number of units gives more of a performance boost than extra layers
num_layers = 1


# Note that the convLSTM layers are very memory intensive, so layers with a number of filters greater than 128 is likely to cause a memory error, even with 32+ GB of GPU memory
# Batch size is also a factor, but one would have to reduce it significantly to avoid memory errors, which would slow training down to a crawl
if os.path.exists(os.path.join(models_dir, 'phi_model.keras')):
    phi_model = tf.keras.models.load_model(os.path.join(models_dir, 'phi_model.keras'))
else:
    phi_model = tf.keras.Sequential()
    phi_model.add(layers.Input(shape=(AGES,REGIONS,1)))
    for i in range(num_layers):
        phi_model.add(layers.ConvLSTM1D(filters=num_units, kernel_size=1, activation='sigmoid', return_sequences=True, data_format="channels_last")) #using 34 instead of 128
    phi_model.add(layers.ConvLSTM1D(filters=64, kernel_size=1, activation='sigmoid', return_sequences=False, data_format="channels_last"))

if os.path.exists(os.path.join(models_dir, 'psi_model.keras')):
    psi_model = tf.keras.models.load_model(os.path.join(models_dir, 'psi_model.keras'), custom_objects={'TileLayer': TileLayer})
else:
    psi_model = tf.keras.Sequential()
    psi_model.add(keras.Input(shape=(REGIONS, 64)))
    psi_model.add(TileLayer())
    for i in range(num_layers):
        psi_model.add(layers.ConvLSTM1D(filters=num_units, kernel_size=1, activation='sigmoid', return_sequences=True, data_format="channels_last"))
    psi_model.add(layers.ConvLSTM1D(filters=1, kernel_size=1, activation='sigmoid', return_sequences=True, data_format="channels_last"))

if os.path.exists(os.path.join(models_dir, 'gamma_model.keras')):
    gammaModel = tf.keras.models.load_model(os.path.join(models_dir, 'gamma_model.keras'), custom_objects={'TransformerLayer': TransformerLayer})
else:
    gammaModel = keras.Sequential()
    gammaModel.add(keras.Input(shape=(REGIONS,64)))
    gammaModel.add(TransformerLayer(embed_dim=64, num_heads=8, ff_dim=128))
    gammaModel.add(layers.Dense(512, activation='sigmoid'))
    gammaModel.add(layers.Dense(256, activation='sigmoid'))
    gammaModel.add(layers.Dense(128, activation='sigmoid'))
    gammaModel.add(layers.Dense(64, activation='sigmoid'))


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


def train(datas, epochs):
    global phi_model, psi_model, gammaModel, gamma_optimizer
    trainSet, trainInputs, trainTargets, trainX, trainY = create_year_tuples(datas)
    dataset = tf.data.Dataset.from_tensor_slices((trainInputs, trainTargets)).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for k in range(epochs):
        for step, (x_batch, y_batch) in enumerate(dataset):
            loss = TrainStep(x_batch, y_batch)
            if step % 100 == 0:
                print(f"Step {step}: Loss = {np.mean(loss.numpy())}")
        if k % 10 == 0: # save progress periodically while training
            with open(os.path.join(models_dir, 'epoch.txt'), 'w') as f:
                f.write(str(k))
            phi_model.save(os.path.join(models_dir, 'phi_model.keras'))
            psi_model.save(os.path.join(models_dir, 'psi_model.keras'))
            gammaModel.save(os.path.join(models_dir, 'gamma_model.keras'))


def validate(ds):
    global gammaModel, phi_model, psi_model
    total_loss = 0
    valSet, valInputs, valTargets, valX, valY = create_year_tuples(ds)
    valDataset = tf.data.Dataset.from_tensor_slices((valInputs, valTargets)).batch(batch_size)
    valDataset = valDataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for step, (x_batch, y_batch) in enumerate(valDataset):
        valLoss = validate_step(x_batch, y_batch)
        total_loss += np.mean(valLoss.numpy())
    return total_loss


@tf.function
def TrainStep(x, y_true):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass through phi
        z_t = phi_model(x)

        # Forward pass through psi (directly from phi)
        y_pred_direct = psi_model(z_t)

        # calculate the loss from phi -> psi
        loss1st = loss1(x, y_pred_direct)

        # pass z_t through gamma to get z_t+1
        z_t1 = gammaModel(z_t)

        # pass z_t+1 through psi to get the next periods curves
        y_pred_indirect = psi_model(z_t1)

        # Calculate second loss (phi -> gamma -> psi)
        loss2nd = loss2(y_true, y_pred_indirect)


        phi_ftplus1 = phi_model(y_true)

        pen = penalty(z_t1, phi_ftplus1)

        # Total loss
        total_loss = loss1st + loss2nd + pen

    # Compute gradients
    gradients = tape.gradient(total_loss,
                              phi_model.trainable_variables +gammaModel.trainable_variables + psi_model.trainable_variables)
    # apply gradients
    optimizer.apply_gradients(zip(gradients,
            phi_model.trainable_variables + gammaModel.trainable_variables + psi_model.trainable_variables))
    return total_loss


@tf.function
def validate_step(x, y_true):
    # This is the same as the train step but without the gradient computation/application

    z_t = phi_model(x)
    y_pred_direct = psi_model(z_t)
    loss1st = loss1(x, y_pred_direct)

    z_t1 = gammaModel(z_t)
    y_pred_indirect = psi_model(z_t1)
    loss2nd = loss2(y_true, y_pred_indirect)

    phi_ftplus1 = phi_model(y_true)
    pen = penalty(z_t1, phi_ftplus1)

    # Total loss
    total_loss = loss1st + loss2nd + pen
    return total_loss

if os.path.exists(os.path.join(models_dir, 'epoch.txt')):
    with open(os.path.join(models_dir, 'epoch.txt'), 'r') as f:
        epo = int(f.read())
else:
    epo = 0

print(f"epochs already trained: {epo}")
train(train_dataset, epochs=700-epo)
val_loss = validate(test_dataset)
print(f"validation loss: {val_loss}")
