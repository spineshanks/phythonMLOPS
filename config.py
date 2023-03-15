# Abstract classes, yaml, and constant symbolic names
import abc
import yaml
from enum import Enum

default_config_file = "daw.yaml"

# Enum classes with symbolic instantiations
class TrainingType(Enum):
    CUSTOM = 'custom'
    AUTOML = 'automl'
    
class InputDataType(Enum):
    CSV = 'csv'
    BQ = 'bigquery'
    
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
        
        TrainingConfig.__init__(
            self,
            create_key=create_key,
            enabled=enabled,
            job_name=job_name
        )
        
        DockerConfigMixin.__init__(
            self,
            create_key=create_key,
            build_path=build_path,
            image=image
        )

        self.model_output_path = model_output_path
        self.machine_type = machine_type
        self.machine_count = machine_count
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count
        self.hyperparameter_tuning = hyperparameter_tuning

# Configuration for AutoML model     
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

# ML model training type configuration    
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
        machine_type: str = "n1-standard-2", # matches default used by Vertex itself: https://cloud.google.com/vertex-ai/docs/predictions/model-co-hosting
        min_replica_count: int = 1,
        max_replica_count: int = 1
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
    gcp_project: str = None
    region: str = None
    staging_bucket: str = None
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
                 gcp_project: str, 
                 region: str,
                 staging_bucket: str,
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
        self.gcp_project = gcp_project
        self.region = region
        self.staging_bucket = staging_bucket
        self.pipeline_name = pipeline_name
        self.pipeline_description = pipeline_description
        self.input_data_config = input_data_config
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config
        self.serving_config = serving_config
    
    @classmethod
    def from_dict(cls, data: dict):
        #ml_pipeline_config_file = data["ml_pipeline_config_file"] if 'ml_pipeline_config_file' in data else None
        #gcp_project = data["gcp_project"] if 'gcp_project' in data else None
        #region = data["region"] if 'region' in data else None
        #endregion
        #staging_bucket = data["staging_bucket"] if 'staging_bucket' in data else None
        #pipeline_name = data["pipeline_name"] if 'pipeline_name' in data else "test"
        #pipeline_description = data["pipeline_description"] if 'pipeline_description' in data else None
        data['input_data_config'] = InputDataConfig.from_dict(data['input_data'])
        data['preprocessing_config'] = PreprocessingConfig.from_dict(data['preprocessing'])
        data['training_config'] = TrainingConfigFactory.from_dict(data['training'])
        data['serving_config'] = ServingConfig.from_dict(data['serving'])

        #input_data_config = InputDataConfig.from_dict(data['input_data']) if 'input_data' in data else None
        #preprocessing_config = PreprocessingConfig.from_dict(data['preprocessing']) if 'preprocessing' in data else None
        #training_config = TrainingConfig.from_dict(data['training']) if 'training' in data else None
        
        return DawConfig(cls._create_key, **data)

def load_config_from_dict(data: dict) -> DawConfig:
    global daw_config
    conf = DawConfig.from_dict(data)
    daw_config = conf
    return conf

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
    daw_config = load_config_from_file()
except Exception as e:
    print("Exception: {}".format(e))
    pass
