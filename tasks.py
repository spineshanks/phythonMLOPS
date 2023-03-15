import logging
import datetime
from azureml.core import Experiment, Run
from invoke import task
import config
import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf = config.load_config_from_file()
# training_conf = config.TrainingConfig.from_dict(conf)
# preprocessing_conf = config.PreprocessingConfig.from_dict(conf)


@task
def build_training_image(c):
    logger.info("Building training image...")
    c.run(f"docker build {conf.training_config.build_path} -t {conf.training_config.image}")
    c.run(f"az acr login --name {conf.container_registry_name}.azurecr.io")
    c.run(f"docker push {conf.training_config.image}")


@task
def build_preprocessing_image(c):
    logger.info("Building preprocessing image...")
    c.run(f"docker build {conf.preprocessing_config.build_path} -t {conf.preprocessing_config.image}")
    c.run(f"az acr login --name {conf.container_registry_name}.azurecr.io")
    c.run(f"docker push {conf.preprocessing_config.image}")


@task
def build_aml_pipeline(c):
    logger.info("Building Azure Machine Learning pipeline...")
    #try:
    pipeline.compile_pipeline()
    logger.info("Done!")
    # except Exception as e:
    #     logger.error("There's an issue: {}".format(e))
    #     pass


@task
def submit_aml_pipeline(c):
    logger.info("Submitting Azure Machine Learning pipeline...")

    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    DISPLAY_NAME = conf.pipeline_name + "_" + TIMESTAMP

    # create and start an experiment
    experiment = Experiment(workspace=conf.workspace, name=DISPLAY_NAME)
    run = experiment.start_logging()

    # upload artifacts for AML steps
    run.upload_folder(name='codes', path='./codes')
    run.upload_folder(name='scripts', path='./scripts')

    # submit pipeline
    submitted_pipeline = experiment.submit(pipeline=pipeline, tags={"type": "automl"}, )
    logger.info("Submitted pipeline run: {}".format(submitted_pipeline.id))

    # set completed status and end time
    run.complete()


@task
def run_unit_tests(c):
    logger.info("Running unit tests...")
    c.run("python3 -m pytest --cov=. --disable-pytest-warnings")

