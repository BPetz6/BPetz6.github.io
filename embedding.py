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
distance_file_path = os.path.join(data_dir, 'distance_matrix.npy')
df = pd.read_csv(csv_file_path)
data = df
distance_matrix = np.load(distance_file_path)

models_dir = os.path.join(os.getenv('SLURM_TMPDIR'), 'models')

# path = "/Users/braedonpetz/Downloads/jp_female_mr1.csv"
# df = pd.read_csv(path)
# data = df
# models_dir = "/Users/braedonpetz/Downloads"
# distance_matrix = np.load('/Users/braedonpetz/Cedar/data/distance_matrix.npy')

    
def configure_gpu():
    '''
    This function configures the GPU to have a memory limit of 31000
    '''
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


class DistanceEmbeddingLayer(Layer):
    '''
    This layer is used to embed the distance matrix into the latent space and then concatenate it onto the input.
    We are using a custom layer so the concatenation is included in the layer
    '''
    def __init__(self, distance_matrix, embedding_dim, **kwargs):
        # Call the parent class's initializer
        super(DistanceEmbeddingLayer, self).__init__(**kwargs)

        # Determine the number of regions from the shape of the distance matrix
        self.num_regions = distance_matrix.shape[0]

        # Store the embedding dimension (the size of the latent dimenstion space to be added to the input)
        self.embedding_dim = embedding_dim

        # Convert the distance matrix to a fixed TensorFlow tensor
        self.distance_matrix = tf.convert_to_tensor(distance_matrix, dtype=tf.float32)

        # Define the embedding layer which will create embeddings for each region
        # input_dim is the number of regions, and output_dim is the embedding dimension for the distances
        self.distance_embedding = layers.Embedding(input_dim=self.num_regions, output_dim=embedding_dim)

    def call(self, inputs):
        # Generate embeddings for all regions using the range of region indices
        # This will produce a matrix of shape (num_regions, embedding_dim)
        distance_embeddings = self.distance_embedding(tf.range(self.num_regions))

        # Compute the weighted embeddings by multiplying the distance matrix with the embeddings
        # This essentially applies the distance weights to the embeddings
        weighted_embeddings = tf.matmul(self.distance_matrix, distance_embeddings)

        # Determine the batch size from the inputs; this is used to repeat the embeddings for each input in the batch
        batch_size = tf.shape(inputs)[0]

        # Expand the dimensions of the weighted embeddings to match the batch size
        # This is necessary for concatenation later
        repeated_embeddings = tf.tile(tf.expand_dims(weighted_embeddings, axis=0), [batch_size, 1, 1])

        # Concatenate the original input (latent representation) with the weighted embeddings along the last dimension
        # This combines the input features with the additional distance-based information
        concatenated = tf.concat([inputs, repeated_embeddings], axis=-1)

        # Return the combined output which now includes the original input features and the embedded distance information
        return concatenated

    def get_config(self):
        # Create a configuration dictionary for the layer
        # This is for saving the model and reloading it later
        config = super(DistanceEmbeddingLayer, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,  # Save the embedding dimension
            'distance_matrix': self.distance_matrix.numpy()  # Save the distance matrix as a numpy array
        })
        return config


batch_size = 32
train_dataset = data[data['year']<=2002]
test_dataset = data[(data['year']>2002) & (data['year']<=2012)]
val_dataset = data[data['year']>2012]
# num_units = int(os.environ.get('UNIT'))
# num_layers = int(os.environ.get('LAYER'))
num_units = 256 # Higher number of units gives more of a performance boost than extra layers
num_layers = 1
embedding_dim = 8 # Better than 4, 16, and 32

distance_embedding_layer = DistanceEmbeddingLayer(distance_matrix=distance_matrix, embedding_dim=embedding_dim)

# Note that the convLSTM layers are very memory intensive, so layers with a number of filters greater than 128 is likely to cause a memory error, even with 32+ GB of GPU memory
# Batch size is also a factor, but one would have to reduce it significantly to avoid memory errors, which would slow training down to a crawl
if os.path.exists(os.path.join(models_dir, 'phi_model.keras')):
    phi_model = tf.keras.models.load_model(os.path.join(models_dir, 'phi_model.keras'))
else:
    phi_model = tf.keras.Sequential()
    phi_model.add(layers.Input(shape=(AGES,REGIONS,1)))
    for i in range(num_layers):
        phi_model.add(layers.ConvLSTM1D(filters=num_units, kernel_size=1, activation='sigmoid', return_sequences=True, data_format="channels_last"))
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
    gammaModel = tf.keras.models.load_model(os.path.join(models_dir, 'gamma_model.keras'), custom_objects={'DistanceEmbeddingLayer': DistanceEmbeddingLayer(distance_matrix=distance_matrix, embedding_dim=embedding_dim)})
else:
    gammaModel = keras.Sequential()
    gammaModel.add(keras.Input(shape=(REGIONS,64)))
    gammaModel.add(distance_embedding_layer)
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