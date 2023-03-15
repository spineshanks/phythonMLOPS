# General imports
from datetime import datetime
from typing import NamedTuple

# Kubeflow imports
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component

# GCP imports
from google_cloud_pipeline_components import aiplatform as gcc_aip
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.experimental import (
    hyperparameter_tuning_job, custom_job)
from google_cloud_pipeline_components.aiplatform import (
    AutoMLTabularTrainingJobRunOp, TabularDatasetCreateOp)

# Azure imports
from azureml.core import (
    Workspace,
    Experiment,
    Dataset,
    Datastore,
    ComputeTarget,
    Environment,
    ScriptRunConfig
)
from azureml.data import OutputFileDatasetConfig
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline

# AWS imports


from config import (DawConfig, daw_config, TrainingType, AutoMlConfig, 
                    CustomTrainingConfig)

# Hack for dsl.pipeline decorator
config = daw_config #DawConfig.from_dict({})

@component(packages_to_install=["google-cloud-aiplatform"])
def model_dir(base_output_directory: str, best_trial: str) -> str:
    from google.cloud.aiplatform_v1.types import study

    trial_proto = study.Trial.from_json(best_trial)
    model_id = trial_proto.id
    return f"{base_output_directory}/{model_id}/model"


@dsl.pipeline(
    name=config.pipeline_name,
    description=config.pipeline_description,
    pipeline_root="gs://{}/vertex_porchlight/".format(config.staging_bucket)
)
def pipeline():
    
    from daw_component import get_or_create_component_provider
    
    component_provider = get_or_create_component_provider(config)
    
    endpoint_creation_task = component_provider.get_or_create_endpoint_creation_component()
    
    dataset_creation_task = component_provider.get_or_create_dataset_creation_component()
    
    preprocessing_task = component_provider.get_or_create_preprocessing_component()
    
    if preprocessing_task and dataset_creation_task:
        preprocessing_task.after(dataset_creation_task)
    
    training_task = component_provider.get_or_create_training_component()
    model_dir_task = component_provider.get_or_create_model_dir_task()
    
    if preprocessing_task:
        training_task.after(preprocessing_task)
    
    model_deployment_task = component_provider.get_or_create_model_deployment_component()
        
def compile_pipeline():
    """
    Note that we can't pass in config here -- complex types are not supported
    by kubeflow:
    
    The pipeline argument "config" is viewed as an artifact due to its type
    "DawConfig". And we currently do not support passing artifacts as pipeline
    inputs. Consider type annotating the argument with a primitive type, such as
    "str", "int", "float", "bool", "dict", and "list".
    """
    compiler.Compiler().compile(
        pipeline_func = pipeline,
        package_path = config.ml_pipeline_config_file
    )
