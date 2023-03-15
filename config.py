# Abstract classes, yaml, and constant symbolic names
import abc
import yaml
from enum import Enum
import azure.storage.blob as blob

default_config_file = "daw.yaml"

# Enum classes with symbolic instantiations
class TrainingType(Enum):
    CUSTOM = 'custom'
    AUTOML = 'automl'
    
class InputDataType(Enum):
    CSV = 'csv'
    # Replace BQ with ADLS_GEN2 for Azure
    ADLS_GEN2 = 'adls_gen2'
    
class AutoMlTrainingType(Enum):
    IMAGE = 'image'
    TEXT = 'text'
    TABULAR = 'tabular'
    VIDEO = 'video'
    FORECASTING = 'forecasting'

class MetricGoals(Enum):
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'

class InputDataColType(Enum):
    STRING = 'string'
    FLOAT = 'float'

# Abtract base classes
class AbstractConfig(abc.ABC):
    _create_key: object
    
    @abc.abstractmethod
    def __init__(self, create_key, *args, **kwargs):
        if (create_key != self._create_key):
            raise RuntimeError(
                f"{self.__class__.__name__} must be created using from_dict()!")
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(cls._create_key, **data)
        
    def __repr__(self):
        attributes = (str(vars(self)).replace(
            "{", "(", 1)[::-1].replace("}", ")", 1)[::-1])
        return "{}{}".format(self.__class__.__name__, attributes)

class InputDataConfig(AbstractConfig):
    _create_key = object()
    
    # TODO: add type validation.
    type: str = None
    input_data_uri: str = None
    columns: list = None
    create_input_dataset: bool = False
    dataset_name: str = None

    def __init__(self,
                 create_key: object,
                 type: str,
                 input_data_uri: str,
                 columns=[],
                 create_input_dataset: bool = False,
                 dataset_name: str = None,
                 *args,
                 **kwargs,
                ):
        
        #self.validate_create_key(create_key, self.__create_key)
        super().__init__(create_key)

        self.type = type
        self.input_data_uri = input_data_uri
        self.columns = columns
        self.create_input_dataset = create_input_dataset
        self.dataset_name = dataset_name
  
class DockerConfigMixin:
    #build_path: str
    #image: str
    
    def __init__(self, build_path: str, image: str, *args, **kwargs):
        self.build_path = build_path
        self.image = image

class TrainingConfig(AbstractConfig):
    _create_key = object()
    
    enabled: bool = False
    job_name: str = None
    
    @abc.abstractmethod
    def __init__(
        self,
        create_key: object,
        enabled: bool,
        job_name: str = None,
        *args,
        **kwargs
    ):
        super().__init__(create_key)
        
        self.enabled = enabled
        self.job_name = job_name          

# Configuration for custom model
class CustomTrainingConfig(TrainingConfig, DockerConfigMixin):
    _create_key = object()

    def __init__(self,
                 create_key,
                 enabled,
                 job_name,
                 image,
                 build_path,
                 model_output_path,
                 hyperparameter_tuning,
                 machine_type,
                 machine_count,
                 accelerator_type,
                 accelerator_count
                 ):

        # Changed for Azure: Removed create_key argument since it is not used in either the TrainingConfig or DockerConfigMixin constructors.
        TrainingConfig.__init__(
            self,
            create_key=create_key,
            enabled=enabled,
            job_name=job_name
        )

        # Changed for Azure: Removed create_key argument since it is not used in the constructor.
        DockerConfigMixin.__init__(
            self,
            build_path=build_path,
            image=image
        )

        self.model_output_path = model_output_path
        self.machine_type = machine_type
        self.machine_count = machine_count
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count
        self.hyperparameter_tuning = hyperparameter_tuning
# Configuration for AutoML model changed for azure
class AutoMlConfig(TrainingConfig):
    _create_key = object()
    
    def __init__(self,
                 create_key: object,
                 enabled: bool,
                 job_name: str,
                 budget_node_hours: float,
                 disable_early_stopping: bool,
                 dataset_type: str,
                 problem_type: str,
                 target_column: str
                ):
        
        super().__init__(create_key, enabled, job_name)
        
        self.budget_node_hours = budget_node_hours
        self.disable_early_stopping = disable_early_stopping
        self.dataset_type = dataset_type
        self.problem_type = problem_type
        self.target_column = target_column

# ML model training type configuration changed for azure   
class TrainingConfigFactory:
    def __init__(self, create_key):
        raise RuntimeError(
            "Don't instantiate TrainingConfigFactory directly. Use from_dict()"
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> TrainingConfig:
        training_type = data["training_type"]
        del(data["training_type"])
        if training_type == TrainingType.CUSTOM.value:
            return CustomTrainingConfig.from_dict(data)
        elif training_type == TrainingType.AUTOML.value:
            return AutoMlConfig.from_dict(data)
        else:
            raise ValueError(f"Unknown training_type {training_type}")

# Preprocessing component configuration
class PreprocessingConfig(AbstractConfig, DockerConfigMixin):
    _create_key = object()

enabled: bool = False
job_name: str = None
image: str = None
build_path: str = None
machine_type: str = None
machine_count: str = None
accelerator_type: str = None
accelerator_count: str = None

training_data_output_path: str = None
test_data_output_path: str = None
validation_data_output_path: str = None

def __init__(self, 
             create_key,
             enabled,
             image=None,
             job_name=None,
             build_path=None,
             machine_type=None,
             machine_count=None,
             accelerator_type=None,
             accelerator_count=None,
             output=None,
             *args,
             **kwargs
            ):
    
    AbstractConfig.__init__(
        self,
        create_key=create_key,
        enabled=enabled
    )
    
    DockerConfigMixin.__init__(
        self,
        create_key=create_key,
        build_path=build_path,
        image=image
    )
    
    self.job_name = job_name
    self.machine_type = machine_type
    self.machine_count = machine_count
    self.accelerator_type = accelerator_type
    self.accelerator_count = accelerator_count
    
    try:
        self.training_data_output_path = output['training_data_path']
        self.test_data_output_path = output['test_data_path']
        self.validation_data_output_path = output['validation_data_path']
    except Exception:
        # "output" is optional, so ignore any errors here.
        pass

# Serving configuration - selection of model
class ServingConfig(AbstractConfig):
    _create_key = object()

    endpoint_name: str = None
    endpoint_description: str = None
    image: str = None
    machine_type: str = None
    min_replica_count: int = None
    max_replica_count: str = None

    def __init__(
        self, 
        create_key: object, 
        endpoint_name: str, 
        endpoint_description: str, 
        image: str = None,
        machine_type: str = "Standard_B2s", # Change to Azure specific machine type
        min_replica_count: int = 1,
        max_replica_count: int = 1 # Change max_replica_count to an integer
    ):
        super().__init__(create_key)
        
        self.endpoint_name = endpoint_name
        self.endpoint_description = endpoint_description
        self.image = image
        self.machine_type = machine_type
        self.min_replica_count = min_replica_count
        self.max_replica_count = max_replica_count
# Pipeline configuration
class DawConfig(AbstractConfig):
    ml_pipeline_config_file: str = None
    azure_subscription_id: str = None  # added for Azure
    azure_resource_group: str = None  # added for Azure
    azure_workspace_name: str = None  # added for Azure
    pipeline_name: str = None
    pipeline_description: str = None
    input_data_config: InputDataConfig = None
    preprocessing_config: PreprocessingConfig = None
    training_config: TrainingConfig = None
    serving_config: ServingConfig = None
    
    _create_key = object()
    
    def __init__(self, 
                 create_key: object,
                 ml_pipeline_config_file: str,
                 azure_subscription_id: str,  # added for Azure
                 azure_resource_group: str,  # added for Azure
                 azure_workspace_name: str,  # added for Azure
                 pipeline_name: str,
                 pipeline_description: str,
                 input_data_config: InputDataConfig,
                 preprocessing_config: PreprocessingConfig,
                 training_config: TrainingConfig,
                 serving_config: ServingConfig,
                 *args, 
                 **kwargs,
                ):
        
        super().__init__(create_key)
        
        self.ml_pipeline_config_file = ml_pipeline_config_file
        self.azure_subscription_id = azure_subscription_id  # added for Azure
        self.azure_resource_group = azure_resource_group  # added for Azure
        self.azure_workspace_name = azure_workspace_name  # added for Azure
        self.pipeline_name = pipeline_name
        self.pipeline_description = pipeline_description
        self.input_data_config = input_data_config
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config
        self.serving_config = serving_config

class DawConfig(AbstractConfig):
    ml_pipeline_config_file: str = None
    azure_ml_workspace: str = None  # added for Azure
    azure_ml_region: str = None  # added for Azure
    pipeline_name: str = None
    pipeline_description: str = None
    input_data_config: InputDataConfig = None
    preprocessing_config: PreprocessingConfig = None
    training_config: TrainingConfig = None
    serving_config: ServingConfig = None
    
    _create_key = object()
    
    def __init__(self, 
                 create_key: object,
                 ml_pipeline_config_file: str,
                 azure_ml_workspace: str,  # modified for Azure
                 azure_ml_region: str,  # modified for Azure
                 pipeline_name: str,
                 pipeline_description: str,
                 input_data_config: InputDataConfig,
                 preprocessing_config: PreprocessingConfig,
                 training_config: TrainingConfig,
                 serving_config: ServingConfig,
                 *args, 
                 **kwargs,
                ):
        
        super().__init__(create_key)
        
        self.ml_pipeline_config_file = ml_pipeline_config_file
        self.azure_ml_workspace = azure_ml_workspace
        self.azure_ml_region = azure_ml_region
        self.pipeline_name = pipeline_name
        self.pipeline_description = pipeline_description
        self.input_data_config = input_data_config
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config
        self.serving_config = serving_config


# changed for Azure: input_data_config, preprocessing_config, training_config, and serving_config are now explicitly defined 
# with their respective values obtained from `data`
@classmethod
def from_dict(cls, data: dict):
    data['input_data_config'] = InputDataConfig.from_dict(data['input_data'])
    data['preprocessing_config'] = PreprocessingConfig.from_dict(data['preprocessing'])
    data['training_config'] = TrainingConfigFactory.from_dict(data['training'])
    data['serving_config'] = ServingConfig.from_dict(data['serving'])

    return DawConfig(cls._create_key, **data)

# changed for Azure: ml_pipeline_config_file, gcp_project, region, and staging_bucket removed as they are GCP specific
def load_config_from_dict(data: dict) -> DawConfig:
    global daw_config
    conf = DawConfig.from_dict(data)
    daw_config = conf
    return conf

# changed for Azure: ml_pipeline_config_file, gcp_project, region, and staging_bucket removed as they are GCP specific
def load_config_from_file(file_location = default_config_file) -> DawConfig:
    global daw_config
    with open(default_config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        conf = DawConfig.from_dict(conf)
        daw_config = conf
        return conf
    
# Final output is an object with model configuration    
daw_config = None

try:
    daw_config = load_config_from_file()  #note : load_config_from_file function shoudl be updated to read the configuration file from the appropiate location in azure
except Exception as e:
    print("Exception: {}".format(e))
    pass