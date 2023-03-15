from invoke import task
from datetime import datetime
import google.cloud.aiplatform as aip
import config
import pipeline
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

conf = config.load_config_from_file()
#training_conf = config.TrainingConfig.from_dict(conf)
#preprocessing_conf = config.PreprocessingConfig.from_dict(conf)

# inv build-training-image

@task
def build_training_image(c):
    logger.info("Building training image...")
    c.run(f"docker build {conf.training_config.build_path} -t {conf.training_config.image}")
    c.run(f"docker push {conf.training_config.image}")
    
@task
def build_preprocessing_image(c):
    logger.info("Building preprocessing image...")
    c.run(f"docker build {conf.preprocessing_config.build_path} -t {conf.preprocessing_config.image}")
    c.run(f"docker push {conf.preprocessing_config.image}")

@task
def build_vertex_pipeline(c):
    logger.info("Building Vertex AI pipeline...")
    #try:
    pipeline.compile_pipeline()
    logger.info("Done!")
    #except Exception as e:
    #    logger.error("There's an issue: {}".format(e))
    #    pass
    
@task
def submit_vertex_pipeline(c):
    logger.info("Submitting Vertex AI pipeline...")
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    DISPLAY_NAME = conf.pipeline_name + "_" + TIMESTAMP
    BUCKET_NAME = conf.staging_bucket
    #BUCKET_URI = f"gs://{BUCKET_NAME}"
    PIPELINE_ROOT = "gs://{}/vertex_porchlight/".format(conf.staging_bucket)

    job = aip.PipelineJob(
        project=conf.gcp_project,
        display_name=DISPLAY_NAME,
        template_path=conf.ml_pipeline_config_file,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=True #Change to false.
    )
    
    logger.info(f"job: {job}")
    job.run(sync=True)


@task
def run_unit_tests(c):
    logger.info("Running unit tests...")
    c.run("python3 -m pytest --cov=. --disable-pytest-warnings")
    