General imports

from datetime import datetime
from typing import NamedTuple

Kubeflow imports

from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component

REPLACED: GCP imports

WITH: Azure imports

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

AWS imports

from config import (DawConfig, daw_config, TrainingType, AutoMlConfig, 
                    CustomTrainingConfig)

Hack for dsl.pipeline decorator

config = dawconfig #DawConfig.fromdict({})

@component(packagestoinstall=["azureml-sdk", "azureml-pipeline-core"])
def modeldir(baseoutputdirectory: str, besttrial: str) -> str:
    from azureml.core.experiment import Experiment

trialproto = Experiment.get(besttrial)
modelid = trialproto.id
return f"{baseoutputdirectory}/{model_id}/model"

@dsl.pipeline(
    name=config.pipelinename,
    description=config.pipelinedescription,
    pipelineroot="azureml://{}/vertexporchlight/".format(config.staging_bucket)
)
def pipeline():

from dawcomponent import getorcreatecomponent_provider

componentprovider = getorcreatecomponent_provider(config)

endpointcreationtask = componentprovider.getorcreateendpointcreationcomponent()

datasetcreationtask = componentprovider.getorcreatedatasetcreationcomponent()

preprocessingtask = componentprovider.getorcreatepreprocessingcomponent()

if preprocessingtask and datasetcreationtask:
    preprocessingtask.after(datasetcreationtask)

CHANGED: replaced GCP AutoML TabularTrainingJobRunOp with Azure TabularDataSetCreateOp

trainingtask = componentprovider.getorcreatetrainingcomponent(TabularDatasetCreateOp)

CHANGED: added modeldirtask for Azure

modeldirtask = componentprovider.getorcreatemodeldirtask()

if preprocessingtask:
    trainingtask.after(preprocessing_task)

modeldeploymenttask = componentprovider.getorcreatemodeldeploymentcomponent()

def compile_pipeline():
    """
    Note that we can't pass in config here -- complex types are not supported
    by kubeflow:

The pipeline argument "config" is viewed as an artifact due to its type
"DawConfig". And we currently do not support passing artifacts as pipeline
inputs. Consider type annotating the argument with a primitive type, such as
"str", "int", "float", "bool", "dict", and "list".
"""

CHANGED: Replaced GCP AIPlatform arguments with Azure ML arguments

compiler.Compiler().compile(
    pipelinefunc = pipeline,
    packagepath = config.mlpipelineconfigfile,
    useazuremlpythonsdk=True
