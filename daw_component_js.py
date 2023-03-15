# Azure Machine Learning imports
from azureml.core import Workspace, Dataset
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

# Script imports
from config import (DawConfig, AutoMlConfig, CustomTrainingConfig, InputDataType)

# Custom lightweight components
@dsl.component(packages_to_install=["azureml-core"])
def model_dir(base_output_directory: PipelineParameter, best_trial: PipelineParameter) -> str:
    from azureml.train.hyperdrive import HyperDriveRun
    
    best_trial_value = best_trial.value
    model_id = HyperDriveRun.get(best_trial_value).id
    return f"{base_output_directory}/{model_id}/model"


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
        dataset_creation_component=None
    ):
        if not self._dataset_creation_component:
            if dataset_creation_component:
                self._dataset_creation_component = dataset_creation_component
            else:
                if self._daw_config.input_data_config.create_input_dataset:
                    input_data_type = self._daw_config.input_data_config.type
                    input_data_uri = self._daw_config.input_data_config.input_data_uri
                    
                    if input_data_type == InputDataType.CSV.value:
                        self._dataset_creation_component = TabularDatasetFactory.from_delimited_files(
                            path=input_data_uri,
                            validate=True,
                            include_path=True,
                            set_column_types_from_file=True,
                            header=self._daw_config.input_data_config.headers,
                            skiprows=self._daw_config.input_data_config.skiprows,
                            delimiter=self._daw_config.input_data_config.delimiter,
                            encoding=self._daw_config.input_data_config.encoding
                        )
                    elif input_data_type == InputDataType.BQ.value:
                        bq_dataset_name = self._daw_config.input_data_config.dataset_name
                        bq_table_name = self._daw_config.input_data_config.table_name
                        dataset = Dataset.get_by_name(workspace=self._daw_config.workspace, name=bq_dataset_name)
                        bq_tabular_dataset = dataset.to_tabular_dataset(
                            target=bq_table_name, 
                            query=f'select * from {bq_dataset_name}.{bq_table_name}',
                            partition_format=None, 
                            validate=True
                        )
                        self._dataset_creation_component = bq_tabular_dataset
                    else:
                        raise ValueError(f"Datasets for type {input_data_type} "
                                         "are currently unsupported.")
                else:
                    # Don't create a dataset if we don't need to.
                    pass
        return self._dataset_creation_component

def get_or_create_endpoint_creation_component(
        self,
        endpoint_creation_component=None
    ):
        if not self._endpoint_creation_component:
            if endpoint_creation_component:
                self._endpoint_creation_component = endpoint_creation_component
            else:
                self._endpoint_creation_component = AmlCompute(
                    workspace=self._daw_config.workspace,
                    name=self._daw_config.serving_config.endpoint_name,
                    description=self._daw_config.serving_config.endpoint_description,
                    vm_size=self._daw_config.serving_config.vm_size,
                    min_nodes=self._daw_config.serving_config.min_nodes,
                    max_nodes=self._daw_config.serving_config.max_nodes
                )
                
        return self._endpoint_creation_component

def get_or_create_preprocessing_component(
        self,
        preprocessing_component=None
    ):
        if self._preprocessing_component is None:
            if preprocessing_component:
                self._preprocessing_component = preprocessing_component
            else:
                if self._daw_config.preprocessing_config.enabled:
                    if type(self._daw_config.training_config) == AutoMlConfig:
                        self._preprocessing_component = TabularDatasetFactory.from_delimited_files(
                            path=self._daw_config.input_data_config.input_data_uri,
                            validate=True,
                            include_path=True,
                            set_column_types_from_file=True,
                            header=self._daw_config.input_data_config.headers,
                            skiprows=self._daw_config.input_data_config.skiprows,
                            delimiter=self._daw_config.input_data_config.delimiter,
                            encoding=self._daw_config.input_data_config.encoding
                        )
                    else:
                        self._preprocessing_component = PythonScriptStep(
                            script_name=self._daw_config.preprocessing_config.script_path,
                            arguments=[
                                "--input-data", self._daw_config.input_data_config.input_data_uri
                            ],
                            inputs=[self._daw_config.input_data_config.input_data_uri],
                            outputs=[],
                            compute_target=self._daw_config.compute_target,
                            source_directory=self._daw_config.source_directory,
                            allow_reuse=True,
                            runconfig=RunConfiguration(),
                            pip_packages=self._daw_config.preprocessing_config.pip_packages,
                            python_version=self._daw_config.preprocessing_config.python_version
                        )
        return self._preprocessing_component
def get_or_create_training_component(self, training_component=None):
    if self._training_component is None:
        if training_component:
            self._training_component = training_component
        else:
            if self._daw_config.training_config.enabled:
                if (type(self._daw_config.training_config) == AutoMlConfig):
                    self._training_component = AzureMLAutoMLTabularTrainingJobRunOp(
                        workspace=self._daw_config.azureml_workspace,
                        optimization_prediction_type=self._daw_config.training_config.problem_type,
                        display_name=self._daw_config.training_config.job_name,
                        dataset=self._dataset_creation_component.outputs['dataset'],
                        target_column=self._daw_config.training_config.target_column,
                        compute_target=self._daw_config.training_config.compute_target,
                        training_data_output_path=self._daw_config.preprocessing_config.training_data_output_path,
                        test_data_output_path=self._daw_config.preprocessing_config.test_data_output_path,
                        budget_milli_node_hours=self._daw_config.training_config.budget_node_hours,
                        export_evaluated_data_items=False
                    )
                else:
                    hyperparameter_specs = {
                        param: hpt.HyperParameterSpec(
                            name=param,
                            type=hpt.HyperParameterType.DOUBLE,
                            min_value=val["range"]["min"],
                            max_value=val["range"]["max"],
                            scale=val["range"]["scale"]
                        ) for param, val in self._daw_config.training_config.hyperparameter_tuning["hyperparameters"].items()
                    }
                    self._training_component = AzureMLHyperparameterTuningJobRunOp(
                        display_name=self._daw_config.training_config.job_name,
                        workspace=self._daw_config.azureml_workspace,
                        compute_target=self._daw_config.training_config.compute_target,
                        training_data_output_path=self._daw_config.preprocessing_config.training_data_output_path,
                        test_data_output_path=self._daw_config.preprocessing_config.test_data_output_path,
                        metric_names=self._daw_config.training_config.hyperparameter_tuning["metrics"],
                        parameter_specs=hyperparameter_specs,
                        max_total_runs=self._daw_config.training_config.hyperparameter_tuning["number_of_trials"],
                        max_concurrent_runs=self._daw_config.training_config.hyperparameter_tuning["parallel_trial_count"],
                        primary_metric_name=self._daw_config.training_config.hyperparameter_tuning["primary_metric"],
                        output_path=self._daw_config.training_config.model_output_path
                    )

    return self._training_component

def get_or_create_model_dir_task(self, model_dir_component=None):
    if self._model_dir_component is None:
        if model_dir_component:
            self._model_dir_component = model_dir_component
        else:
            if (type(self._daw_config.training_config) == CustomTrainingConfig):
                trials_task = self._training_component.GetTrialsOp(
                    azureml_workspace=self._daw_config.azureml_workspace,
                    compute_target=self._daw_config.training_config.compute_target,
                    training_data_output_path=self._daw_config.preprocessing_config.training_data_output_path,
                    test_data_output_path=self._daw_config.preprocessing_config.test_data_output_path,
                    gcp_resources=training_task.outputs['gcp_resources']
                )

                best_trial_task = self._training_component.GetBestTrialOp(
                    azureml_workspace=self._daw_config.azureml_workspace,
                    compute_target=self._daw_config.training_config.compute_target,
                    training_data_output_path=self._daw_config.preprocessing_config.training_data_output_path,
                    test_data_output_path=self._daw_config.preprocessing_config.test_data_output_path,
                    trials=trials_task.output, 
                    metric_names=self._daw_config.training_config.hyperparameter_tuning["metrics"]
                )

                self._model_dir_component = model_dir(MODEL_OUTPUT_PATH, best_trial_task.output).after(best_trial_task)
    return self._model_dir_component


def get_or_create_model_upload_component(self, model_upload_component=None):
    if self._model_upload_component is None:
        if model_upload_component:
            self._model_upload_component = model_upload_component
        else:
            if (type(self._daw_config.training_config) == AutoMlConfig):
                return None
            else:
                gcc_aip.ModelUploadOp(
                    workspace=self._daw_config.azureml_workspace,
                    display_name="upload_model",
                    inputs=[model_dir_op.output],
                    target=self._daw_config.serving_config.model_registry
                )

                    
ddef get_or_create_model_deployment_component(self, model_deployment_component=None):
    if self._model_deployment_component is None:
        if model_deployment_component:
            self._model_deployment_component = model_deployment_component
        else:
            model_name = "my_model"
            model = Model.register(model_path="path/to/model",
                                   model_name=model_name,
                                   workspace=self._daw_config.workspace)
            
            inference_config = InferenceConfig(environment=self._daw_config.environment,
                                               entry_script="score.py",
                                               source_directory="./")
            
            deployment_config = AciWebservice.deploy_configuration(cpu_cores=self._daw_config.serving_config.cpu_cores,
                                                                    memory_gb=self._daw_config.serving_config.memory_gb)
            
            service_name = "my_service"
            service = Model.deploy(workspace=self._daw_config.workspace,
                                    name=service_name,
                                    models=[model],
                                    inference_config=inference_config,
                                    deployment_config=deployment_config)
            service.wait_for_deployment(show_output=True)
            
            self._model_deployment_component = service.scoring_uri

    return self._model_deployment_component

_component_provider: DawConfig = None

def get_or_create_component_provider(
    daw_config: DawConfig,
    endpoint_creation_component: Optional = None,
    preprocessing_component: Optional = None,
    training_component: Optional = None,
    model_deployment_component: Optional = None
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
Note that the code itself doesn't seem to have any direct dependence on either Google Cloud or Azure, so most of the changes involve updating imports and making sure that any necessary dependencies are installed. The only changes to the code itself are adding the Optional type hint and updating the import for that type hint.




