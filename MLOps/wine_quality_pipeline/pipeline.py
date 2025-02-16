import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.sklearn import SKLearn

def create_pipeline(
        role,
        bucket,
        pipeline_name,
        base_job_prefix,
        boto3_session  # Add this parameter
):
    # Create SageMaker session using the provided boto3 session
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)

    # Create SKLearn processor with the explicit session
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/sklearn-preprocess",
        sagemaker_session=sagemaker_session  # Add this parameter
    )

    # The rest of your pipeline creation code remains the same, but add sagemaker_session to other components
    sklearn_estimator = SKLearn(
        entry_point="training.py",
        source_dir="scripts",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        framework_version="0.23-1",
        base_job_name=f"{base_job_prefix}/sklearn-train",
        sagemaker_session=sagemaker_session  # Add this parameter
    )

    # Modify create_model_monitor call to pass the session
    model_monitor = create_model_monitor(role, bucket, base_job_prefix, sagemaker_session)

    # Create pipeline with session
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[],
        steps=[preprocessing_step, training_step, evaluation_step],
        sagemaker_session=sagemaker_session  # Add this parameter
    )

    return pipeline, model_monitor

def create_model_monitor(role, bucket, base_job_prefix, sagemaker_session):
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=f"s3://{bucket}/wine-quality/monitoring",
        capture_options=["REQUEST", "RESPONSE"],
        csv_content_types=["text/csv"],
        json_content_types=["application/json"]
    )

    model_monitor = sagemaker.model_monitor.ModelMonitor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        base_job_name=f"{base_job_prefix}/monitor",
        data_capture_config=data_capture_config,
        sagemaker_session=sagemaker_session  # Add this parameter
    )

    return model_monitor

# Execute the pipeline
if __name__ == "__main__":
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    default_bucket = sagemaker_session.default_bucket()

    pipeline, model_monitor = create_pipeline(
        role=role,
        bucket=default_bucket,
        pipeline_name="WineQualityPipeline",
        base_job_prefix="wine-quality"
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()