import pytest
import config

class TestAbstractConfig:
    
    def test_abstract_config_instantiation(self):
        # We shouldn't be able to directly instantiate an abstract class.
        with pytest.raises(TypeError):
            config.AbstractConfig("my key")

class TestInputDataConfig:
    def test_input_data_config_instantiation(self):
        with pytest.raises(RuntimeError):
            config.InputDataConfig(object(), "type", "input_uri", ["test"])
            
    def test_input_data_config_from_dict_with_bad_type(self):
        config.InputDataConfig.from_dict({
            "type": "csv",
            "input_data_uri": "my_uri"
        })
        
class TestAbstractTrainingConfig:
    def test_abstract_training_config_instantiation(self):
        # We shouldn't be able to directly instantiate an abstract class.
        with pytest.raises(TypeError):
            config.TrainingConfig(object(), False, "test")
        
    def test_abstract_training_config_bad_training_type(self):
        with pytest.raises(ValueError):
            config.TrainingConfigFactory.from_dict({
                "training_type": "bad"
            })
            
    def test_abstract_training_config_automl(self):
        automl = config.TrainingConfigFactory.from_dict({
            "training_type": "automl",
            "enabled": True,
            "job_name": "example-job",
            "budget_node_hours": 3,
            "disable_early_stopping": False,
            "dataset_type": "test",
            "problem_type": "type",
        })
        
        assert type(automl) == config.AutoMlConfig
        
    def test_abstract_training_config_custom(self):
        automl = config.TrainingConfigFactory.from_dict({
            "training_type": "custom",
            "enabled": True,
            "job_name": "example-job",
            "image": "gcr.io/my-image:latest",
            "build_path": "./training_app/",
            "model_output_path": "gs://my-bucket/models",
            "machine_type": "n1-standard1",
            "machine_count": 3,
            "accelerator_type": "nvidia-something-rather",
            "accelerator_count": 2,
            "hyperparameter_tuning": {
                "number_of_trials": 1,
                "parallel_trial_count": 1,
                "metrics": {
                    "mse": "minimize"
                },
                "hyperparameters": {
                    "learning_rate": {
                        "type": "double",
                        "range": {
                            "min": 0.001,
                            "max": 1,
                            "scale": "log"
                        }
                },
                    "subsample": {
                        "type": "double",
                        "range": {
                            "min": 0.001,
                            "max": 1,
                            "scale": "log"
                        }
            }
        }}})
        
        assert type(automl) == config.CustomTrainingConfig
        
class TestDawConfig:
    
    def test_daw_config_instantiation(self):
        with pytest.raises(RuntimeError):
            config.DawConfig(
                create_key="my key", 
                ml_pipeline_config_file="test_file.yaml", 
                gcp_project="my-project",
                region="us-central1",
                staging_bucket="my-bucket",
                pipeline_name="my-pipeline", 
                pipeline_description="my description", 
                input_data_config=config.InputDataConfig.from_dict(
                     {"type": "csv", "input_data_uri": "test"}
                ),
                preprocessing_config=config.PreprocessingConfig.from_dict({
                     "enabled": False
                }),
                training_config=config.TrainingConfigFactory.from_dict({
                     "training_type": "automl",
                     "enabled": True,
                     "job_name": "example-job",
                     "budget_node_hours": 3,
                     "disable_early_stopping": False,
                     "dataset_type": "test",
                     "problem_type": "type",
                 }),
                serving_config=config.ServingConfig.from_dict({
                    "endpoint_name": "test",
                    "endpoint_description": "test description",
                })
            )
            
    def test_daw_config_instantiation_from_dict(self):
        daw_config = config.DawConfig.from_dict({
            "ml_pipeline_config_file": "test_file.yaml", 
            "gcp_project": "my-project",
            "region": "us-central1",
            "staging_bucket": "my-bucket",
            "pipeline_name": "my-pipeline", 
            "pipeline_description": "my description",
            "input_data": {"type": "csv", "input_data_uri": "test"},
            "preprocessing": {"enabled": False},
            "training": {
                 "training_type": "automl",
                 "enabled": True,
                 "job_name": "example-job",
                 "budget_node_hours": 3,
                 "disable_early_stopping": False,
                 "dataset_type": "test",
                 "problem_type": "type",
            },
            "serving": {
                "endpoint_name": "test",
                "endpoint_description": "An endpoint for testing",
            }
        }
        )
        
        assert type(daw_config) == config.DawConfig
