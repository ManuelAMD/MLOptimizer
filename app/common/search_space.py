from dataclasses import dataclass, field
from enum import Enum
import abc
import optuna
import numpy as np
from dataclasses_json import DataClassJsonMixin
from typing import List
from app.common.socketCommunication import *
from app.common.tools import *

class SearchSpaceType(Enum):
    IMAGE = 'image'
    REGRESSION = 'regression'
    TIME_SERIES = 'time_series'
    IMAGE_TIME_SERIES = 'image_time_series'

class DatasetType(Enum):
    IMAGE = 1
    REGRESSION = 2
    TIME_SERIES = 3
    IMAGE_TIME_SERIES = 4

@dataclass(frozen=True)
class SearchSpace:
	""" A Search Space constants container

    This object defines the ranges of a Search Space used to build model architectures
    """

	@staticmethod
	@abc.abstractmethod
	def get_type() -> SearchSpaceType:
		pass

	@classmethod
	@abc.abstractmethod
	def get_hash(cls) -> str:
		pass

class ModelArchitectureFactory(abc.ABC):
    """ A Model Architecture Factory
    This object contains the logic for generating the model architecture and parameters.
    It is associated to a SearchSpace object for getting architecture ranges and uses
    an optuna Trial object as the recommender model (default is optuna bayessian TPE model)
    """
    @abc.abstractmethod
    def generate_model_params(self, recommender: optuna.Trial):
        pass

    @abc.abstractmethod
    def get_search_space(self) -> SearchSpace:
        pass

@dataclass
class ModelArchitectureParameters(DataClassJsonMixin):
	""" Contains the architecture parameters generated by ModelArchitectureFactory

    This object is used for being encoded to JSON to send models to the workers.
    The worker nodes construct the models based on this object
    """

#Search spaces from each type of search
@dataclass(frozen=True)
class ImageModelSearchSpace(SearchSpace):
    BASE_ARCHITECTURE: tuple = field(default=('cnn','inception',), hash=False)

    #Vanilla CNN vased model search space
    CNN_BLOCKS_N_MIN: int = 1
    CNN_BLOCKS_N_MAX: int = 4
    CNN_BLOCKS_CONV_LAYERS_N_MIN: int = 1
    CNN_BLOCKS_CONV_LAYERS_N_MAX: int = 4
    CNN_BLOCK_CONV_FILTERS_BASE_MULTIPLIER: int = 32
    CNN_BLOCK_CONV_FILTERS_MIN: int = 1
    CNN_BLOCK_CONV_FILTERS_MAX: int = 2
    CNN_BLOCK_CONV_FILTERS_SIZES: tuple = (3,5)
    # CNN_BLOCK_MAX_POOLING_ENABLED: tuple = (False, True)
    CNN_BLOCK_MAX_POOLING_SIZES: tuple = (2,)
    CNN_BLOCK_DROPOUT_VALUES = (0, 0.1, 0.3, 0.5)
    # Inception based model search space
    """The number of input CNN layers is dynamically calculated based on reduction allowed
    by input size and output size (input size of first Inception module)"""
    INCEPTION_STEM_BLOCK_BASE_MULTIPLIER: int = 8
    INCEPTION_STEM_BLOCK_CONV_FILTERS_MIN: int = 1
    INCEPTION_STEM_BLOCK_CONV_FILTERS_MAX: int = 16
    INCEPTION_STEM_BLOCK_CONV_FILTER_SIZES: tuple = (3,5,7)
    INCEPTION_STEM_BLOCK_MAX_POOLING_SIZES: tuple = (2,3)
    INCEPTION_STEM_BLOCK_OUTPUT_SIZE_MIN: int = 8
    INCEPTION_STEM_BLOCK_OUTPUT_SIZE_MAX: int = 32
    INCEPTION_BLOCKS_N_MIN: int = 1
    INCEPTION_BLOCKS_N_MAX: int = 3
    INCEPTION_MODULES_N_MIN: int = 1
    INCEPTION_MODULES_N_MAX: int = 3
    INCEPTION_MODULES_INPUT_SIZE_MIN: int = 4
    INCEPTION_MODULES_INPUT_SIZE_MAX: int = 56
    #16 completed range of googleLeNet (limited by the VRAM de GPUs)
    INCEPTION_MODULES_BASE_MULTIPLIER: int = 4
    INCEPTION_MODULES_CONV1X1_FILTERS_MIN: int = 1
    INCEPTION_MODULES_CONV1X1_FILTERS_MAX: int = 16
    INCEPTION_MODULES_CONV3X3_REDUCE_FILTERS_MIN: int = 1
    INCEPTION_MODULES_CONV3X3_REDUCE_FILTERS_MAX: int = 12
    INCEPTION_MODULES_CONV3X3_FILTERS_MIN: int = 1
    INCEPTION_MODULES_CONV3X3_FILTERS_MAX: int = 24
    INCEPTION_MODULES_CONV5X5_REDUCE_FILTERS_MIN: int = 1
    INCEPTION_MODULES_CONV5X5_REDUCE_FILTERS_MAX: int = 3
    INCEPTION_MODULES_CONV5X5_FILTERS_MIN: int = 1
    INCEPTION_MODULES_CONV5X5_FILTERS_MAX: int = 8
    INCEPTION_MODULES_POOLING_CONV_FILTERS_MIN: int = 1
    INCEPTION_MODULES_POOLING_CONV_FILTERS_MAX: int = 8

    #Classifier layer mode
    CLASSIFIER_LAYER_TYPE: tuple = field(default=('mlp','gap'), hash=False)
    #MLP classifier layers search space
    CLASSIFIER_LAYERS_N_MIN: int = 0
    CLASSIFIER_LAYERS_N_MAX: int = 2
    CLASSIFIER_LAYERS_BASE_MULTIPLIER: int = 16
    CLASSIFIER_LAYERS_UNITS_MIN: int = 1
    CLASSIFIER_LAYERS_UNITS_MAX: int = 32
    CLASSIFIER_DROPOUT_VALUES = (0, 0.1, 0.3, 0.5)

    @staticmethod
    def get_type() -> SearchSpaceType:
        return SearchSpaceType.IMAGE

    @classmethod
    def get_hash(cls) -> str:
        return hash(cls())

@dataclass(frozen=True)
class RegressionModelSearchSpace(SearchSpace):
    BASE_ARCHITECTURE: tuple = field(default=('mlp'), hash=False)
    CLASSIFIER_LAYERS_N_MIN: int = 1
    CLASSIFIER_LAYERS_N_MAX: int = 10
    CLASSIFIER_LAYERS_BASE_MULTIPLIER: int = 16
    CLASSIFIER_LAYERS_UNITS_MIN: int = 1
    CLASSIFIER_LAYERS_UNITS_MAX: int = 32
    CLASSIFIER_DROPOUT_VALUES = (0, 0.1, 0.2, 0.3, 0.5)

    @staticmethod
    def get_type() -> SearchSpaceType:
        return SearchSpaceType.REGRESSION

    @classmethod
    def get_hash(cls) -> str:
        return hash(cls())

@dataclass(frozen=True)
class TimeSeriesModelSearchSpace(SearchSpace):
    BASE_ARCHITECTURE: tuple = field(default=('lstm', 'mlp'), hash=False)
    #LSTM architecture
    LSTM_LAYERS_N_MIN: int = 1
    LSTM_LAYERS_N_MAX: int = 6
    LSTM_LAYERS_BASE_MULTIPLIER: int = 16
    LSTM_LAYERS_UNITS_MIN: int = 1
    LSTM_LAYERS_UNITS_MAX: int = 32

    #MLP based model search space
    CLASSIFIER_LAYER: tuple = field(default=('mlp','gap'), hash=False)
    CLASSIFIER_LAYERS_N_MIN: int = 1
    CLASSIFIER_LAYERS_N_MAX: int = 10
    CLASSIFIER_LAYERS_BASE_MULTIPLIER: int = 16
    CLASSIFIER_LAYERS_UNITS_MIN: int = 1
    CLASSIFIER_LAYERS_UNITS_MAX: int = 32
    CLASSIFIER_DROPOUT_VALUES = (0, 0.1, 0.2, 0.3, 0.5)

    @staticmethod
    def get_type() -> SearchSpaceType:
        return SearchSpaceType.TIME_SERIES

    @classmethod
    def get_hash(cls) -> str:
        return hash(cls())

class ImageTimeSeriesModelSearchSpace(SearchSpace):
    #BASE_ARCHITECTURE: tuple = field(default=(''))
    CODIFIER_LAYERS_N_MIN: int = 2
    CODIFIER_LAYERS_N_MAX: int = 6

    CODIFIER_UNITS_MIN: int = 2
    CODIFIER_UNITS_MAX: int = 16

    CONV_KERNEL: tuple = (3,5,7)

    KERNEL_X_MIN: int = 0
    KERNEL_X_MAX: int = 16
    KERNEL_Y_MIN: int = 0
    KERNEL_Y_MAX: int = 16

    def set_kernel_values(self, val_x, val_y):
        self.KERNEL_X_MIN = 0
        self.KERNEL_X_MAX = len(divisors(val_x))
        self.KERNEL_Y_MIN = 0
        self.KERNEL_Y_MAX = len(divisors(val_x))
        return

    @staticmethod
    def get_type() -> SearchSpaceType:
        return SearchSpaceType.IMAGE_TIME_SERIES
    
    @classmethod
    def get_hash(cls) -> str:
        return hash(cls())


#Architecture parameters part
#Clase capaz de almacenar la información de los parámetros de un modelo a manera de datos.
@dataclass
class ImageModelArchitectureParameters(ModelArchitectureParameters):

    base_architecture: str
    # CNN based architecture
    # All blocks parameters
    cnn_blocks_n: int
    # Per block parameters
    cnn_blocks_conv_layers_n: int
    cnn_block_conv_filters: List[int]
    cnn_block_conv_filter_sizes: List[int]
    # cnn_block_max_pooling_enabled: List[bool]
    cnn_block_max_pooling_sizes: List[int]
    cnn_block_dropout_values: List[int]

    # Inception based architecture
    # Pre-Inception input reduction CNN layers
    inception_stem_blocks_n: int
    inception_stem_block_conv_filters: List[int]
    inception_stem_block_conv_filter_sizes: List[int]
    inception_stem_block_max_pooling_sizes: List[int]
    # Inception modules parameters
    inception_blocks_n: int
    inception_modules_n: int
    inception_modules_conv1x1_filters: List[int]
    inception_modules_conv3x3_reduce_filters: List[int]
    inception_modules_conv3x3_filters: List[int]
    inception_modules_conv5x5_reduce_filters: List[int]
    inception_modules_conv5x5_filters: List[int]
    inception_modules_pooling_conv_filters: List[int]

    # Classifier layers
    classifier_layer_type: str
    classifier_layers_n: int
    classifier_dropouts: List[int]
    classifier_layers_units: List[int]
    learning_rate: float

    @staticmethod
    def new():
        return ImageModelArchitectureParameters(
            base_architecture=None,
            cnn_blocks_n=0,
            cnn_blocks_conv_layers_n=0,
            cnn_block_conv_filters=list(),
            cnn_block_conv_filter_sizes=list(),
            cnn_block_max_pooling_sizes=list(),
            cnn_block_dropout_values=list(),
            inception_stem_blocks_n=0,
            inception_stem_block_conv_filters=list(),
            inception_stem_block_conv_filter_sizes=list(),
            inception_stem_block_max_pooling_sizes=list(),
            inception_blocks_n=0,
            inception_modules_n=0,
            inception_modules_conv1x1_filters=list(),
            inception_modules_conv3x3_reduce_filters=list(),
            inception_modules_conv3x3_filters=list(),
            inception_modules_conv5x5_reduce_filters=list(),
            inception_modules_conv5x5_filters=list(),
            inception_modules_pooling_conv_filters=list(),
            classifier_layer_type=None,
            classifier_layers_n=0,
            classifier_dropouts=list(),
            classifier_layers_units=list(),
            learning_rate=0.0
        )

@dataclass
class RegressionModelArchitectureParameters(ModelArchitectureParameters):
    base_architecture: str
    # Classifier layers
    classifier_layers_n: int
    classifier_dropouts: List[int]
    classifier_layers_units: List[int]
    learning_rate: float

    @staticmethod
    def new():
        return RegressionModelArchitectureParameters(
            base_architecture=None,
            classifier_layers_n=0,
            classifier_dropouts=list(),
            classifier_layers_units=list(),
            learning_rate=0.0
        )

@dataclass
class TimeSeriesModelArchitectureParameters(ModelArchitectureParameters):
    base_architecture: str
    # Lstm layers
    lstm_layers_n: int
    lstm_layers_units: List[int]
    # Classifier layers
    classifier_layer: str
    classifier_layers_n: int
    classifier_dropouts: List[int]
    classifier_layers_units: List[int]
    learning_rate: float

    @staticmethod
    def new():
        return TimeSeriesModelArchitectureParameters(
            base_architecture=None,
            lstm_layers_n=0,
            lstm_layers_units=list(),
            classifier_layer=None,
            classifier_layers_n=0,
            classifier_dropouts=list(),
            classifier_layers_units=list(),
            learning_rate=0.0
        )

@dataclass
class ImageTimeSeriesModelArchitectureParameters(ModelArchitectureParameters):
    codifier_layers_n: int

    codifier_units: List[int]
    conv_kernels: List[int]

    kernels_x: List[int]
    kernels_y: List[int]

    @staticmethod
    def new():
        return ImageTimeSeriesModelArchitectureParameters(
            codifier_layers_n = 0,
            codifier_units = list(),
            conv_kernels = list(),
            kernels_x = list(),
            kernels_y = list()
        )

#Model architectures factories part

class ImageModelArchitectureFactory(ModelArchitectureFactory):
    _inception_stem_blocks_n_range = None
    sp = ImageModelSearchSpace()

    def generate_model_params(self, recommender: optuna.Trial, input_dim: tuple):
        model_params: ImageModelArchitectureParameters = ImageModelArchitectureParameters.new()
        #Base architecture by optuna
        model_params.base_architecture = recommender.suggest_categorical('BASE_ARCHITECTURE', self.sp.BASE_ARCHITECTURE)
        #Define the structure, by the base architecture.
        if model_params.base_architecture == 'cnn':
            self._generate_cnn_based_architecture(input_dim, recommender, model_params)
        elif model_params.base_architecture == 'inception':
            self._generate_inception_based_architecture(input_dim, recommender, model_params)
        #Classifier part by optuna
        model_params.classifier_layer_type = recommender.suggest_categorical('CLASSIFIER_LAYER_TYPE', self.sp.CLASSIFIER_LAYER_TYPE)
        #verify if has or not a classifier layer.
        if model_params.classifier_layer_type == 'mlp':
            model_params = self._generate_mlp_classifier_layers(recommender, model_params)
        print(recommender.params)
        return model_params

    def _generate_cnn_based_architecture(self, input_dim: tuple, recommender: optuna.Trial, model_params: ImageModelArchitectureParameters)->ImageModelArchitectureParameters:
        model_params.cnn_blocks_n = recommender.suggest_int('CNN_BLOCKS_N', self.sp.CNN_BLOCKS_N_MIN, self.sp.CNN_BLOCKS_N_MAX)
        tag = 'CNN_BLOCKS_CONV_LAYERS_N'
        model_params.cnn_blocks_conv_layers_n = recommender.suggest_int(tag, self.sp.CNN_BLOCKS_CONV_LAYERS_N_MIN, self.sp.CNN_BLOCKS_CONV_LAYERS_N_MAX)
        for n in range (0, model_params.cnn_blocks_n):
            #Number of filters
            tag = 'CNN_BLOCKS_CONV_FILTERS_'+str(n)
            conv_filters = round(recommender.suggest_loguniform(tag, self.sp.CNN_BLOCK_CONV_FILTERS_MIN, self.sp.CNN_BLOCK_CONV_FILTERS_MAX))
            conv_filters = conv_filters * self.sp.CNN_BLOCK_CONV_FILTERS_BASE_MULTIPLIER * (2**n)
            model_params.cnn_block_conv_filters.append(conv_filters)
            #Filters sizes
            tag = "CNN_BLOCK_CONV_FILTER_SIZES_" + str(n)
            filter_size = recommender.suggest_categorical(tag, self.sp.CNN_BLOCK_CONV_FILTERS_SIZES)
            model_params.cnn_block_conv_filter_sizes.append(filter_size)
            #Max pooling filter sizes
            tag = "CNN_BLOCK_MAX_POOLING_SIZES_" + str(n)
            max_pool_size = recommender.suggest_categorical(tag, self.sp.CNN_BLOCK_MAX_POOLING_SIZES)
            model_params.cnn_block_max_pooling_sizes.append(max_pool_size)
            #Dropout values
            tag = "CNN_BLOCK_DROPOUT_VALUES_" + str(n)
            dropout_value = recommender.suggest_categorical(tag, self.sp.CNN_BLOCK_DROPOUT_VALUES)
            model_params.cnn_block_dropout_values.append(dropout_value)
        return model_params

    def _generate_inception_based_architecture(self, input_dim: tuple, recommender: optuna.Trial, model_params: ImageModelArchitectureParameters)->ImageModelArchitectureParameters:
        # Vanilla CNN input layers parameters
        inception_stem_blocks_n_range: tuple = self._get_inception_stem_blocks_n_range(input_dim[0])
        model_params.inception_stem_blocks_n = recommender.suggest_int("INCEPTION_STEM_N", inception_stem_blocks_n_range[0], inception_stem_blocks_n_range[1])

        for n in range(0, model_params.inception_stem_blocks_n):
            #Inception filters number
            tag = "INCEPTION_STEM_BLOCK_CONV_FILTERS_" + str(n)
            filters = round(recommender.suggest_loguniform(tag, self.sp.INCEPTION_STEM_BLOCK_CONV_FILTERS_MIN, self.sp.INCEPTION_STEM_BLOCK_CONV_FILTERS_MAX))
            filters = filters * self.sp.INCEPTION_STEM_BLOCK_BASE_MULTIPLIER
            model_params.inception_stem_block_conv_filters.append(filters)

            #Filter sizes for inception blocks
            tag = "INCEPTION_STEM_BLOCK_CONV_FILTER_SIZES_" + str(n)
            conv_size = recommender.suggest_categorical(tag, self.sp.INCEPTION_STEM_BLOCK_CONV_FILTER_SIZES)
            model_params.inception_stem_block_conv_filter_sizes.append(conv_size)

            #Max pooling filter sizes
            tag = "INCEPTION_STEM_BLOCK_MAX_POOLING_SIZES_" + str(n)
            pool_size = recommender.suggest_categorical(tag, self.sp.INCEPTION_STEM_BLOCK_MAX_POOLING_SIZES)
            model_params.inception_stem_block_max_pooling_sizes.append(pool_size)

        # Inception modules parameters
        model_params.inception_blocks_n = recommender.suggest_int("INCEPTION_BLOCKS_N", self.sp.INCEPTION_BLOCKS_N_MIN, self.sp.INCEPTION_BLOCKS_N_MAX)
        model_params.inception_modules_n = recommender.suggest_int("INCEPTION_MODULES_N", self.sp.INCEPTION_MODULES_N_MIN, self.sp.INCEPTION_MODULES_N_MAX)

        conv_1x1_filters_min = self.sp.INCEPTION_MODULES_CONV1X1_FILTERS_MIN
        conv3x3_reduce_filters_min = self.sp.INCEPTION_MODULES_CONV3X3_REDUCE_FILTERS_MIN
        conv3x3_filters_min = self.sp.INCEPTION_MODULES_CONV3X3_FILTERS_MIN
        conv5x5_reduce_filters_min = self.sp.INCEPTION_MODULES_CONV5X5_REDUCE_FILTERS_MIN
        conv5x5_filters_min = self.sp.INCEPTION_MODULES_CONV5X5_FILTERS_MIN
        pooling_conv_filters_min = self.sp.INCEPTION_MODULES_POOLING_CONV_FILTERS_MIN

        #Inception blocks generation
        for n in range(0, model_params.inception_blocks_n):

            tag = "INCEPTION_MODULES_CONV1X1_FILTERS_" + str(n)
            filters = round(recommender.suggest_loguniform(tag, conv_1x1_filters_min, self.sp.INCEPTION_MODULES_CONV1X1_FILTERS_MAX))
            conv_1x1_filters_min = filters
            filters = filters * self.sp.INCEPTION_MODULES_BASE_MULTIPLIER
            model_params.inception_modules_conv1x1_filters.append(filters)

            tag = "INCEPTION_MODULES_CONV3X3_REDUCE_FILTERS_" + str(n)
            filters = round(recommender.suggest_loguniform(tag, conv3x3_reduce_filters_min, self.sp.INCEPTION_MODULES_CONV3X3_REDUCE_FILTERS_MAX))
            conv3x3_reduce_filters_min = filters
            filters = filters * self.sp.INCEPTION_MODULES_BASE_MULTIPLIER
            model_params.inception_modules_conv3x3_reduce_filters.append(filters)

            tag = "INCEPTION_MODULES_CONV3X3_FILTERS_" + str(n)
            min_filters = max(conv3x3_reduce_filters_min, conv3x3_filters_min)
            filters = round(recommender.suggest_loguniform( tag, min_filters, self.sp.INCEPTION_MODULES_CONV3X3_FILTERS_MAX))
            conv3x3_filters_min = filters
            filters = filters * self.sp.INCEPTION_MODULES_BASE_MULTIPLIER
            model_params.inception_modules_conv3x3_filters.append(filters)

            tag = "INCEPTION_MODULES_CONV5X5_REDUCE_FILTERS_" + str(n)
            filters = round(recommender.suggest_loguniform(tag, conv5x5_reduce_filters_min, self.sp.INCEPTION_MODULES_CONV5X5_REDUCE_FILTERS_MAX))
            conv5x5_reduce_filters_min = filters
            filters = filters * self.sp.INCEPTION_MODULES_BASE_MULTIPLIER
            model_params.inception_modules_conv5x5_reduce_filters.append(filters)

            tag = "INCEPTION_MODULES_CONV5X5_FILTERS_" + str(n)
            min_filters = max(conv5x5_reduce_filters_min, conv5x5_filters_min)
            filters = round(recommender.suggest_loguniform(tag, min_filters, self.sp.INCEPTION_MODULES_CONV5X5_FILTERS_MAX))
            conv5x5_filters_min = filters
            filters = filters * self.sp.INCEPTION_MODULES_BASE_MULTIPLIER
            model_params.inception_modules_conv5x5_filters.append(filters)

            tag = "INCEPTION_MODULES_POOLING_CONV_FILTERS_" + str(n)
            filters = round(recommender.suggest_loguniform(tag, pooling_conv_filters_min, self.sp.INCEPTION_MODULES_POOLING_CONV_FILTERS_MAX))
            pooling_conv_filters_min = filters
            filters = filters * self.sp.INCEPTION_MODULES_BASE_MULTIPLIER
            model_params.inception_modules_pooling_conv_filters.append(filters)
        return model_params

    def _generate_mlp_classifier_layers(self, recommender: optuna.Trial, model_params: ImageModelArchitectureParameters) -> ImageModelArchitectureParameters:
        model_params.classifier_layers_n = recommender.suggest_int("CLASSIFIER_LAYERS_N", self.sp.CLASSIFIER_LAYERS_N_MIN, self.sp.CLASSIFIER_LAYERS_N_MAX)

        for n in range(0, model_params.classifier_layers_n):
            tag = "CLASSIFIER_DROPOUTS_" + str(n)
            dropout = recommender.suggest_categorical(tag, self.sp.CLASSIFIER_DROPOUT_VALUES)
            model_params.classifier_dropouts.append(dropout)

            tag = "CLASSIFIER_LAYERS_UNITS_" + str(n)
            units = round(recommender.suggest_loguniform(tag, self.sp.CLASSIFIER_LAYERS_UNITS_MIN, self.sp.CLASSIFIER_LAYERS_UNITS_MAX))
            units = units * self.sp.CLASSIFIER_LAYERS_BASE_MULTIPLIER
            model_params.classifier_layers_units.append(units)

        return model_params

    def _get_inception_stem_blocks_n_range(self, input_size):

        if self._inception_stem_blocks_n_range is not None:
            return self._inception_stem_blocks_n_range

        min_result_size = self.sp.INCEPTION_MODULES_INPUT_SIZE_MIN
        max_result_size = self.sp.INCEPTION_MODULES_INPUT_SIZE_MAX

        cad = "Acceptable size range before Inception modules is " + str(min_result_size) + " to " + str(max_result_size) + " px"
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': "Calculating reduction modules acceptable range..."})

        # Calculate minimum reduction modules for input
        # Given the smallest conv filter size and smallest pooling reduction
        min_conv_filter_size = min(self.sp.INCEPTION_STEM_BLOCK_CONV_FILTER_SIZES)
        min_pooling_size = min(self.sp.INCEPTION_STEM_BLOCK_MAX_POOLING_SIZES)

        reduced_size = input_size
        modules_n_min = 0

        while reduced_size > max_result_size:
            conv_output_size = reduced_size - min_conv_filter_size + 1
            pool_output_size = int(np.floor(conv_output_size / min_pooling_size))
            reduced_size = pool_output_size
            modules_n_min = modules_n_min + 1
            print("Max size after ", modules_n_min, " modules: ", reduced_size)

        # Calculate maximum reduction modules allowed for input size
        # Given the largest conv filter size and largest pooling reduction
        max_conv_filter_size = max(self.sp.INCEPTION_STEM_BLOCK_CONV_FILTER_SIZES)
        max_pooling_size = max(self.sp.INCEPTION_STEM_BLOCK_MAX_POOLING_SIZES)

        reduced_size = input_size
        modules_n_max = 0

        while reduced_size > min_result_size:
            conv_output_size = reduced_size - max_conv_filter_size + 1
            pool_output_size = int(np.floor(conv_output_size / max_pooling_size))
            reduced_size = pool_output_size
            if reduced_size > min_result_size:
                modules_n_max = modules_n_max + 1
                print("Min size after ", modules_n_max, " modules: ", reduced_size)
            else:
                break
        module_range = (modules_n_min, modules_n_max)

        cad = "Initial modules number range for input " + str(input_size) + " is " + str(module_range)
        SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})

        self._inception_stem_blocks_n_range = module_range

        return module_range

    def get_search_space(self) -> SearchSpace:
        return self.sp

class RegressionModelArchitectureFactory(ModelArchitectureFactory):
    sp = RegressionModelSearchSpace()

    def generate_model_params(self, recommender: optuna.Trial, input_dim: tuple):
        model_params: RegressionModelArchitectureParameters = RegressionModelArchitectureParameters.new()

        model_params.base_architecture = 'mlp'#recommender.suggest_categorical('BASE_ARCHITECTURE', self.sp.BASE_ARCHITECTURE)

        if model_params.base_architecture == 'mlp':
            model_params = self._generate_mlp_classifier_layers(recommender, model_params)

        print(recommender.params)
        return model_params

    def _generate_mlp_classifier_layers(self, recommender: optuna.Trial, model_params: RegressionModelArchitectureParameters) -> RegressionModelArchitectureParameters:
        model_params.classifier_layers_n = recommender.suggest_int('CLASSIFIER_LAYERS_N', self.sp.CLASSIFIER_LAYERS_N_MIN, self.sp.CLASSIFIER_LAYERS_N_MAX)

        for n in range(0, model_params.classifier_layers_n):
            tag = 'CLASSIFIER_DROPOUTS_'+str(n)
            dropout = recommender.suggest_categorical(tag, self.sp.CLASSIFIER_DROPOUT_VALUES)
            model_params.classifier_dropouts.append(dropout)

            tag = "CLASSIFIER_LAYERS_UNITS_" + str(n)
            units = round(recommender.suggest_loguniform(tag, self.sp.CLASSIFIER_LAYERS_UNITS_MIN, self.sp.CLASSIFIER_LAYERS_UNITS_MAX))
            units = units * self.sp.CLASSIFIER_LAYERS_BASE_MULTIPLIER
            model_params.classifier_layers_units.append(units)

        return model_params

    def get_search_space(self) -> SearchSpace:
        return self.sp

class TimeSeriesModelArchitectureFactory(ModelArchitectureFactory):
    sp = TimeSeriesModelSearchSpace()

    def generate_model_params(self, recommender: optuna.Trial, input_dim: tuple):
        model_params: TimeSeriesModelArchitectureParameters = TimeSeriesModelArchitectureParameters.new()
        model_params.base_architecture = recommender.suggest_categorical("BASE_ARCHITECTURE", self.sp.BASE_ARCHITECTURE)
        if model_params.base_architecture == "lstm":
            model_params = self._generate_lstm_layers(recommender, model_params)

        model_params.classifier_layer = recommender.suggest_categorical("CLASSIFIER_LAYER", self.sp.CLASSIFIER_LAYER)

        if model_params.base_architecture == 'mlp' or model_params.classifier_layer == 'mlp':
            model_params = self._generate_mlp_classifier_layers(recommender, model_params)

        print(recommender.params)
        return model_params

    def _generate_lstm_layers(self, recommender:optuna.Trial, model_params: TimeSeriesModelArchitectureParameters) -> TimeSeriesModelArchitectureParameters:
        model_params.lstm_layers_n = recommender.suggest_int("LSTM_LAYERS_N", self.sp.LSTM_LAYERS_N_MIN, self.sp.LSTM_LAYERS_N_MAX)

        for n in range(0, model_params.lstm_layers_n):
            tag = "LSTM_LAYERS_UNITS_" + str(n)
            units = round(recommender.suggest_loguniform(tag, self.sp.LSTM_LAYERS_UNITS_MIN, self.sp.LSTM_LAYERS_UNITS_MAX))
            units = units * self.sp.LSTM_LAYERS_BASE_MULTIPLIER
            model_params.lstm_layers_units.append(units)

        return model_params

    def _generate_mlp_classifier_layers(self, recommender: optuna.Trial, model_params: TimeSeriesModelArchitectureParameters) -> TimeSeriesModelArchitectureParameters:
        model_params.classifier_layers_n = recommender.suggest_int("CLASSIFIER_LAYERS_N", self.sp.CLASSIFIER_LAYERS_N_MIN, self.sp.CLASSIFIER_LAYERS_N_MAX)

        for n in range(0, model_params.classifier_layers_n):
            tag = "CLASSIFIER_DROPOUTS_" + str(n)
            dropout = recommender.suggest_categorical(tag, self.sp.CLASSIFIER_DROPOUT_VALUES)
            model_params.classifier_dropouts.append(dropout)

            tag = "CLASSIFIER_LAYERS_UNITS_" + str(n)
            units = round(recommender.suggest_loguniform(tag, self.sp.CLASSIFIER_LAYERS_UNITS_MIN, self.sp.CLASSIFIER_LAYERS_UNITS_MAX))
            units = units * self.sp.CLASSIFIER_LAYERS_BASE_MULTIPLIER
            model_params.classifier_layers_units.append(units)

        return model_params

    def get_search_space(self) -> SearchSpace:
        return self.sp

class ImageTimeSeriesModelArchitectureFactory(ModelArchitectureFactory):
    sp = ImageTimeSeriesModelSearchSpace()

    def generate_model_params(self, recommender: optuna.Trial, input_dim: tuple):
        if self.sp.KERNEL_X_MIN == 0:
            self.sp.set_kernel_values(input_dim[0], input_dim[1])
        x = input_dim[0]
        y = input_dim[1]
        div_x = divisors(x)
        div_x = div_x[:int(len(div_x))]
        div_y = divisors(y)
        div_y = div_y[:int(len(div_y))]
        #self.sp.set_divisors(div_x, div_y)
        model_params: ImageTimeSeriesModelArchitectureParameters = ImageTimeSeriesModelArchitectureParameters.new()

        model_params.codifier_layers_n = recommender.suggest_int('CODIFIER_LAYERS_N', self.sp.CODIFIER_LAYERS_N_MIN, self.sp.CODIFIER_LAYERS_N_MAX)
        for n in range(0, model_params.codifier_layers_n):
            tag = 'KERNELS_X_'+str(n)
            tag_2 = 'KERNELS_Y_'+str(n)
            #-1 for list index
            #actual_kernel_x = 
            #actual_kernel_y = 
            tam_x = len(div_x)
            if tam_x == 1:
                pos_x = 0
            else:
                pos_x = recommender.suggest_int(tag, self.sp.KERNEL_X_MIN, self.sp.KERNEL_X_MAX)
                while pos_x >= tam_x:
                    pos_x -= tam_x
            tam_y = len(div_y)
            if tam_y == 1:
                pos_y = 0
            else:
                pos_y = recommender.suggest_int(tag_2, self.sp.KERNEL_Y_MIN, self.sp.KERNEL_Y_MAX)
                while pos_y >= tam_y:
                    pos_y -= tam_y
            #print("Posición actual x", pos_x, "tam:", tam_x,"Posición actual y", pos_y, "tam_y:",tam_y)
            x = x / div_x[pos_x]
            y = y / div_y[pos_y]
            new_kernel_x = divisors(x)
            new_kernel_y = divisors(y)
            if len(new_kernel_x) == 0:
                print("kernel_x vacio", new_kernel_x)
                new_kernel_x = [1]
            if len(new_kernel_y) == 0:
                print("kernel_y vacio", new_kernel_y)
                new_kernel_y = [1]
            print("Se actualizan los kernels", new_kernel_x, new_kernel_y)
            model_params.kernels_x.append(div_x[pos_x])
            model_params.kernels_y.append(div_y[pos_y])
            if len(new_kernel_x) == 1:
                div_x = new_kernel_x
            else:
                div_x = new_kernel_x[:int(len(new_kernel_x)/2)]
                #div_x = new_kernel_x[:len(new_kernel_x)]
            if len(new_kernel_y) == 1:
                div_y = new_kernel_y
            else:
                div_y = new_kernel_y[:int(len(new_kernel_y)/2)]
                #div_y = new_kernel_y[:len(new_kernel_y)]
            model_params.codifier_units.append(recommender.suggest_int('CODIFIER_UNITS_'+str(n), self.sp.CODIFIER_UNITS_MIN, self.sp.CODIFIER_UNITS_MAX))
            tag_3 = 'CONV_FILTER_SIZE_' + str(n)
            kernel_size = recommender.suggest_categorical(tag_3, self.sp.CONV_KERNEL)
            model_params.conv_kernels.append((kernel_size, kernel_size))
        print(recommender.params)
        return model_params

    def get_search_space(self) -> SearchSpace:
        return self.sp