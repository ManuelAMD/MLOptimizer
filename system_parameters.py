"""
Parameters class for system modification

Modify this class to change the parameters for training and datasets options

"""
"""
#Rabbit MQ Connections
MASTER_PORT: int = 15672
MASTER_MODEL_PARAMETER_QUEUE: str = 'parameters'
MASTER_MODEL_PERFORMANCE_QUEUE: str = 'results'
MASTER_HOST_URL: str = 'localhost'
MASTER_USER: str = 'guest'
MASTER_PASSWORD: str = 'guest'
MASTER_VIRTUAL_HOST: str = '/'

MASTER_CONNECTION = [MASTER_PORT, MASTER_MODEL_PARAMETER_QUEUE, MASTER_MODEL_PERFORMANCE_QUEUE, MASTER_HOST_URL, MASTER_USER, MASTER_PASSWORD, MASTER_VIRTUAL_HOST]
#For each connection, it counts like it's an slave.
SLAVES_CONNECTIONS = [[15672, 'parameters', 'results', 'localhost', 'guest', 'guest', '/'],[15672, 'parameters', 'results', 'localhost', 'guest', 'guest', '/']]
"""
class SystemParameters:
    #Rabbit MQ Connections
    """INSTANCE_PORT: int = 15672
    INSTANCE_MODEL_PARAMETER_QUEUE: str = 'parameters'
    INSTANCE_MODEL_PERFORMANCE_QUEUE: str = 'results'
    INSTANCE_HOST_URL: str = 'localhost'
    INSTANCE_USER: str = 'guest'
    INSTANCE_PASSWORD: str = 'guest'
    INSTANCE_VIRTUAL_HOST: str = '/'"""
    INSTANCE_PORT: int = 15672
    INSTANCE_MODEL_PARAMETER_QUEUE: str = 'parameters'
    INSTANCE_MODEL_PERFORMANCE_QUEUE: str = 'results'
    INSTANCE_HOST_URL: str = '189.186.75.55'
    INSTANCE_USER: str = 'invitado'
    INSTANCE_PASSWORD: str = 'mcc2021'
    INSTANCE_VIRTUAL_HOST: str = '/'

    INSTANCE_CONNECTION = [INSTANCE_PORT, INSTANCE_MODEL_PARAMETER_QUEUE, INSTANCE_MODEL_PERFORMANCE_QUEUE, INSTANCE_HOST_URL, INSTANCE_USER, INSTANCE_PASSWORD, INSTANCE_VIRTUAL_HOST]


    #Dataset parameters
    #Tensorflow datasets = mnist, fashion_mnist, cifar10, cifar100, horses_or_humans
    DATASET_NAME: str = 'carbon_nanotubes'
    #Types: image = 1, regression = 2, time-series = 3.
    DATASET_TYPE: int = 2
    DATASET_BATCH_SIZE: int = 8
    DATASET_VALIDATION_SPLIT: float = 0.2
    #empty for tensorflowDatasets
    #DATASET_INFO_ROUTE: str = '' 
    DATASET_SHAPE: tuple = (5)
    #Image dataset parameters
    DATASET_CLASSES: int = 10
    #Regression dataset parameters
    DATASET_FEATURES: int = 5
    DATASET_LABELS: int = 3
    #Time Series dataset parameters
    DATASET_WINDOW_SIZE: int = 100
    DATASET_DATA_SIZE: int = 1


    #AutoML parameters
    #specify if the train is with cpu (False) or with gpu (True)
    TRAIN_GPU: bool = True
    TRIALS = 10
    #Exploration parameters 
    EXPLORATION_SIZE: int = 10
    EXPLORATION_EPOCHS: int = 10
    EXPLORATION_EARLY_STOPPING_PATIENCE: int = 3
    #Hall of fame parameters
    HALL_OF_FAME_SIZE: int = 3
    HALL_OF_FAME_EPOCHS: int = 150
    HOF_EARLY_STOPPING_PATIENCE: int = 10


    #Model parameters
    DTYPE: str = 'float32'
    OPTIMIZER: str = 'adam'
    LAYERS_ACTIVATION_FUNCTION: str = 'relu'
    OUTPUT_ACTIVATION_FUNCTION: str = 'linear'
    KERNEL_INITIALIZER: str = 'normal'
    LOSS_FUNCTION: str = 'mse'

    #Image Classification
    #LOSS_FUNCTION: str = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']
    #LAYERS_ACTIVATION_FUNCTION: str = 'relu'
    #OUTPUT_ACTIVATION_FUNCTION: str = 'softmax'
    PADDING: str = 'same'
    #KERNEL_INITIALIZER: str = 'he_uniform'
    WEIGHT_DECAY = 1e-4

    #Regression models
    #LOSS_FUNCTION: str = 'mse'
    #LAYERS_ACTIVATION_FUNCTION: str = 'relu'
    #OUTPUT_ACTIVATION_FUNCTION: str = 'linear'
    #KERNEL_INITIALIZER: str = 'normal'

    #Time-series models
    #LOSS_FUNCTION: str = 'mse'
    LSTM_ACTIVATION_FUNCTION: str = 'tanh'
    #LAYERS_ACTIVATION_FUNCTION: str = 'tanh'
    #OUTPUT_ACTIVATION_FUNCTION: str = 'tanh'
    #KERNEL_INITIALIZER: str = 'normal'
