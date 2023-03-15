# Kubeflow imports
from kfp import dsl
from kfp.v2.dsl import component

# GCP imports
from google_cloud_pipeline_components.experimental import (
    hyperparameter_tuning_job, custom_job)
from google_cloud_pipeline_components import aiplatform as gcc_aip
from google_cloud_pipeline_components.aiplatform import (
    AutoMLTabularTrainingJobRunOp, TabularDatasetCreateOp)
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Script imports
from config import (DawConfig, AutoMlConfig, CustomTrainingConfig, InputDataType)


# Custom lightweight components
@component(packages_to_install=["google-cloud-aiplatform"])
def model_dir(base_output_directory: str, best_trial: str) -> str:
    from google.cloud.aiplatform_v1.types import study

    trial_proto = study.Trial.from_json(best_trial)
    model_id = trial_proto.id
    return f"{base_output_directory}/{model_id}/model"

"""
@component(packages_to_install=["google-cloud-aiplatform",
                                "slack-sdk==3.17.2",
                                "google-cloud-pubsub==1.7.0"])
def slackbot_model_deploy(best_trial: str) -> bool:
    from google.cloud.aiplatform_v1.types import study
    from google.cloud import pubsub_v1
    import json
    import time
    
    trial_proto = study.Trial.from_json(best_trial)
    model_id = trial_proto.id
    
    print(trial_proto)
    
    project_id = "seabrook-ai"
    topic_id = "slack-messages"
    subscription_id = "subscription-testing"
    
    test_payload = {
        "model_name": "My Model",
        "metrics": {
            "rmse": "0.9",
            "accuracy": "0.9"
        },
        "number_of_trials": 3,
        "training_job_link": "https://example.com/stuff",
        "data_source_type": "table",
        "data_source_format": "BigQuery",
        "data_source_path": "my-project:mydataset.mytable",
        "data_source_last_updated": "2022-07-10"
    }
    
    bytes_payload = json.dumps(test_payload).encode("UTF-8")
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    future = publisher.publish(topic_path, bytes_payload)
    
    subscriber = pubsub_v1.SubscriberClient()
    #request = pubsub_v1.types.PullRequest(
    #    subscription=subscription_id,
    #    max_messages=1
    #)
    
    result = None
    
    while not result:
        response = subscriber.pull(subscription=subscriber.subscription_path(project_id, subscription_id), max_messages=1)
        if len(response.received_messages) == 0:
            time.sleep(5)
            continue
            
        result = json.loads(response.received_messages[0].message.data.decode("UTF-8"))["approved"]
        
    return result
"""

class ComponentProvider:
    
    _daw_config: DawConfig = None
    _endpoint_creation_component = None
    _dataset_creation_component = None
    _preprocessing_component = None
    _training_component = None
    _model_deployment_component = None
    _model_upload_component = None
    _model_dir_component = None
    
    
    def __init__(
        self,
        daw_config: DawConfig,
        endpoint_creation_component = None,
        preprocessing_component = None,
        training_component = None,
        model_deployment_component = None,
        model_upload_component = None,
        model_dir_component = None
    ):
        self._daw_config = daw_config
        self._endpoint_creation_component = endpoint_creation_component
        self._preprocessing_component = preprocessing_component
        self._training_component = training_component
        self._model_deployment_component = model_deployment_component
        self._model_upload_component = model_upload_component
        self._model_dir_component = model_dir_component
        
    
    def get_or_create_dataset_creation_component(
        self,
        dataset_creation_component = None
    ):
        if not self._dataset_creation_component:
            if dataset_creation_component:
                self._dataset_creation_component = dataset_creation_component
            else:
                if self._daw_config.input_data_config.create_input_dataset:
                    input_data_type = self._daw_config.input_data_config.type
                    input_data_uri = self._daw_config.input_data_config.input_data_uri
                    
                    if input_data_type == InputDataType.CSV.value:
                        self._dataset_creation_component = TabularDatasetCreateOp(
                            project=self._daw_config.gcp_project,
                            location=self._daw_config.region,
                            display_name=self._daw_config.input_data_config.dataset_name,
                            gcs_source=input_data_uri
                        )
                    elif input_data_type == InputDataType.BQ.value:
                        self._dataset_creation_component = TabularDatasetCreateOp(
                            project=self._daw_config.gcp_project,
                            location=self._daw_config.region,
                            display_name=self._daw_config.input_data_config.dataset_name,
                            bq_source=input_data_uri
                        )
                    else:
                        raise ValueError(f"Datasets for type {input_data_type} "
                                         "are currently unsupported.")
                else:
                    # Don't create a dataset if we don't need to.
                    pass
        return self._dataset_creation_component
        
        
    def get_or_create_endpoint_creation_component(
        self,
        endpoint_creation_component = None
    ):
        if not self._endpoint_creation_component:
            if endpoint_creation_component:
                self._endpoint_creation_component = endpoint_creation_component
            else:
                self._endpoint_creation_component = gcc_aip.EndpointCreateOp(
                    project=self._daw_config.gcp_project,
                    display_name=self._daw_config.serving_config.endpoint_name,
                    description=self._daw_config.serving_config.endpoint_description
                )
                
        return self._endpoint_creation_component

    def get_or_create_preprocessing_component(
        self,
        preprocessing_component = None
    ):
        if self._preprocessing_component is None:
            if preprocessing_component:
                self._preprocessing_component = preprocessing_component
            else:
                if self._daw_config.preprocessing_config.enabled:
                    if type(self._daw_config.training_config) == AutoMlConfig:
                        self._preprocessing_component = TabularDatasetCreateOp(
                            project=config.gcp_project,
                            location=config.region,
                            display_name="test",
                            gcs_source=self._daw_config.input_data_config.input_data_uri
                        )
                    else:
                        self._preprocessing_component = custom_job.CustomTrainingJobOp(
                            project=self._daw_config.gcp_project,
                            location=self._daw_config.region,
                            display_name=self._daw_config.preprocessing_config.job_name,
                            worker_pool_specs=[{
                                "machine_spec": {
                                    "machine_type": 
                                        self._daw_config.preprocessing_config.machine_type,
                                    "accelerator_type": 
                                        self._daw_config.preprocessing_config.accelerator_type,
                                    "accelerator_count": 
                                        self._daw_config.preprocessing_config.accelerator_count
                                },
                                "replica_count": 
                                    self._daw_config.preprocessing_config.machine_count,
                                "container_spec": {
                                    "image_uri": 
                                        self._daw_config.preprocessing_config.image,
                                    "args": [
                                        '--input-data', 
                                        self._daw_config.input_data_config.input_data_uri
                                    ]
                                }
                            }]
                        )
        return self._preprocessing_component

    def get_or_create_training_component(
        self,
        training_component = None
    ):
        if self._training_component is None:
            if training_component:
                self._training_component = training_component
            else:
                if self._daw_config.training_config.enabled:
                    if (type(self._daw_config.training_config) == AutoMlConfig):
                        self._training_component = AutoMLTabularTrainingJobRunOp(
                            project=self._daw_config.gcp_project,
                            optimization_prediction_type=self._daw_config.training_config.problem_type,
                            display_name=self._daw_config.training_config.job_name,
                            dataset=self._dataset_creation_component.outputs['dataset'],
                            target_column=self._daw_config.training_config.target_column,
                            budget_milli_node_hours=self._daw_config.training_config.budget_node_hours,
                            export_evaluated_data_items=False
                        )
                    else:
                        print(self._daw_config.training_config)
                        
                        hyperparameter_specs = {
                            param: hpt.DoubleParameterSpec(min=val["range"]["min"], 
                                                           max=val["range"]["max"], 
                                                           scale=val["range"]["scale"]) for param, val in  self._daw_config.training_config.hyperparameter_tuning["hyperparameters"].items()}
                        self._training_component = hyperparameter_tuning_job.HyperparameterTuningJobRunOp(
                            display_name=self._daw_config.training_config.job_name,
                            project=self._daw_config.gcp_project,
                            location=self._daw_config.region,
                            worker_pool_specs = [{
                                "machine_spec": {
                                    "machine_type": self._daw_config.training_config.machine_type,
                                    "accelerator_type": self._daw_config.training_config.accelerator_type,
                                    "accelerator_count": self._daw_config.training_config.accelerator_count
                                },
                                "replica_count": self._daw_config.training_config.machine_count,
                                "container_spec": {
                                    "image_uri": self._daw_config.training_config.image,
                                    "args": [
                                        "--training-dataset-path", self._daw_config.preprocessing_config.training_data_output_path,
                                        "--test-dataset-path", self._daw_config.preprocessing_config.test_data_output_path]
                                }
                            }],
                            study_spec_metrics=self._daw_config.training_config.hyperparameter_tuning["metrics"],
                            study_spec_parameters=hyperparameter_tuning_job.serialize_parameters(hyperparameter_specs),
                            max_trial_count=self._daw_config.training_config.hyperparameter_tuning["number_of_trials"], # orig 15, can change back after testing (too slow!)
                            parallel_trial_count=self._daw_config.training_config.hyperparameter_tuning["parallel_trial_count"],
                            base_output_directory=self._daw_config.training_config.model_output_path
                        ) 

        return self._training_component
    
    def get_or_create_model_dir_task(
        self,
        model_dir_component = None
    ):
        if self._model_dir_component is None:
            if model_dir_component:
                self._model_dir_component = model_dir_component
            else:
                if (type(self._daw_config.training_config) == CustomTrainingConfig):
                    trials_task = self._training_component.GetTrialsOp(
                        gcp_resources=training_task.outputs['gcp_resources']
                    )

                    best_trial_task = self._training_component.GetBestTrialOp(
                        trials=trials_task.output, 
                        study_spec_metrics=self._daw_config.training_config.hyperparameter_tuning["metrics"]
                    )

                    self._model_dir_component = model_dir(MODEL_OUTPUT_PATH, best_trial_task.output).after(best_trial_task)
        return self._model_dir_component
    
    def get_or_create_model_upload_component(
        self,
        model_upload_component = None
    ):
        if self._model_upload_component is None:
            if model_upload_component:
                self._model_upload_component = model_upload_component
            else:
                if (type(self._daw_config.training_config) == AutoMlConfig):
                    return None
                else:
                    gcc_aip.ModelUploadOp(
                        project=config.gcp_project,
                        display_name="upload_model",
                        artifact_uri=model_dir_op.output,
                        serving_container_image_uri=self._daw_config.serving_config.image,
                    )

                    
    def get_or_create_model_deployment_component(
        self,
        model_deployment_component = None
    ):
        if self._model_deployment_component is None:
            if model_deployment_component:
                self._model_deployment_component = model_deployment_component
            else:
                if (type(self._daw_config.training_config) == AutoMlConfig):
                    self._model_deployment_component = gcc_aip.ModelDeployOp(
                        model = self._training_component.outputs["model"],
                        endpoint = self._endpoint_creation_component.outputs["endpoint"],
                        traffic_split = {
                            "0": 100
                        },
                        dedicated_resources_machine_type = self._daw_config.serving_config.machine_type,
                        dedicated_resources_min_replica_count = self._daw_config.serving_config.min_replica_count,
                        dedicated_resources_max_replica_count = self._daw_config.serving_config.max_replica_count
                        
                    )
                else:
                    self._model_deployment_component = gcc_aip.ModelDeployOp(
                        model = self._model_upload_component.outputs["model"],
                        endpoint = self._endpoint_creation_component.outputs["endpoint"],
                        traffic_split = {
                            "0": 100
                        }
                    ),
                    dedicated_resources_machine_type = self._daw_config.serving_config.machine_type,
                    dedicated_resources_min_replica_count = self._daw_config.serving_config.min_replica_count,
                    dedicated_resources_max_replica_count = self._daw_config.serving_config.max_replica_count
                

        return self._model_deployment_component
                    

                    
# Component provider singleton
_component_provider: DawConfig = None
                    
def get_or_create_component_provider(
    daw_config: DawConfig,
    endpoint_creation_component = None,
    preprocessing_component = None,
    training_component = None,
    model_deployment_component = None
) -> ComponentProvider:
    global _component_provider
    
    if not _component_provider:
        _component_provider = ComponentProvider(
            daw_config,
            endpoint_creation_component,
            preprocessing_component,
            training_component,
            model_deployment_component
        )
    
    return _component_provider
    
                    
                    
                    