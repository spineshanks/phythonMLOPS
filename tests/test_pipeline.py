import pytest
import config
import sys


class TestPipeline:
    
    @pytest.fixture(autouse=True)
    def unimport_components(self):
        try:
            sys.modules.pop('pipeline')
            sys.modules.pop('daw_component')
        except KeyError:
            pass
        
    
    def test_automl_pipeline(self):
        config.load_config_from_dict({
            "ml_pipeline_config_file": "test_file.json", 
            "gcp_project": "my-project",
            "region": "us-central1",
            "staging_bucket": "my-bucket",
            "pipeline_name": "my-pipeline", 
            "pipeline_description": "my description",
            "input_data": {"type": "csv", "input_data_uri": "test", "create_input_dataset": True, "dataset_name": "example-dataset"},
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
        })
        
        import pipeline
        pipeline.compile_pipeline()
        
    def test_custom_training_pipeline(self):
        config.load_config_from_dict({
            "ml_pipeline_config_file": "test_file.json", 
            "gcp_project": "my-project",
            "region": "us-central1",
            "staging_bucket": "my-bucket",
            "pipeline_name": "my-pipeline", 
            "pipeline_description": "my description",
            "input_data": {"type": "csv", "input_data_uri": "test", "create_input_dataset": True, "dataset_name": "example-dataset"},
            "preprocessing": {
                "enabled": True,
                "job_name": "example-job",
                "image": "gcr.io/my-image:latest",
                "build_path": "./training_app/",
                "model_output_path": "gs://my-bucket/models",
                "machine_type": "n1-standard1",
                "machine_count": 3,
                "accelerator_type": "nvidia-something-rather",
                "accelerator_count": 2
            },
            "training": {
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
        }}
            },
            "serving": {
                "endpoint_name": "test",
                "endpoint_description": "An endpoint for testing",
            }
        })
        
        import pipeline
        pipeline.compile_pipeline()
        