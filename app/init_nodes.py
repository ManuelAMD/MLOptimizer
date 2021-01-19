from app.common.dataset import *
from app.common.search_space import *
from system_parameters import *

def master():
	model_architecture_factory: ModelArchitectureFactory = get_model_architecture()
	dataset: Dataset = get_dataset()
	print(model_architecture_factory)
	from app.master_node.optimization_job import OptimizationJob
	optimization_job = OptimizationJob(dataset, model_architecture_factory)
	optimization_job.start_optimization(trials=TRIALS)

def slave():
	model_architecture_factory: ModelArchitectureFactory = get_model_architecture()
	from app.slave_node.training_slave import TrainingSlave
	training_slave = TrainingSlave(model_architecture_factory)
	training_slave.start_slave()

def get_model_architecture() -> ModelArchitectureFactory:
	if DATASET_TYPE == 1:
		return ImageModelArchitectureFactory()
	elif DATASET_TYPE == 2:
		return RegressionModelArchitectureFactory()
	elif DATASET_TYPE == 3:
		return TimeSeriesModelArchitectureFactory()
	else:
		print("Please enter a valid dataset type")
		return

def get_dataset() -> Dataset:
	if DATASET_TYPE == 1:
		return ImageClassificationBenchmarkDataset(DATASET_NAME, DATASET_SHAPE, DATASET_CLASSES, DATASET_BATCH_SIZE, DATASET_VALIDATION_SPLIT)
	elif DATASET_TYPE == 2:
		return RegressionBenchmarkDataset(DATASET_NAME, DATASET_SHAPE, DATASET_FEATURES, DATASET_LABELS, DATASET_BATCH_SIZE, DATASET_VALIDATION_SPLIT)
	elif DATASET_TYPE == 3:
		return TimeSeriesBenchmarkDataset(DATASET_NAME, DATASET_WINDOW_SIZE, DATASET_DATA_SIZE, DATASET_BATCH_SIZE, DATASET_VALIDATION_SPLIT)
	else:
		print("Please enter a valid dataset type")
		return