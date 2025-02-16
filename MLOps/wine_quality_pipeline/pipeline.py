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

def create_model_monitor(role, bucket, base_job_prefix):
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
        data_capture_config=data_capture_config
    )

    return model_monitor

def create_pipeline(
        role,
        bucket,
        pipeline_name,
        base_job_prefix
):
    # Create SKLearn processor for data preprocessing
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/sklearn-preprocess"
    )

    # Define preprocessing step
    preprocessing_step = ProcessingStep(
        name="PreprocessWineData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/wine-quality/raw/winequality.csv",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/wine-quality/processed/train"
            ),
            ProcessingOutput(
                output_name="test_data",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/wine-quality/processed/test"
            )
        ],
        code="scripts/preprocessing.py"
    )

    # Define training step
    sklearn_estimator = SKLearn(
        entry_point="training.py",
        source_dir="scripts",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        framework_version="0.23-1",
        base_job_name=f"{base_job_prefix}/sklearn-train"
    )

    training_step = TrainingStep(
        name="TrainWineQualityModel",
        estimator=sklearn_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    # Define evaluation step
    evaluation_step = ProcessingStep(
        name="EvaluateWineQualityModel",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "test_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/wine-quality/evaluation"
            )
        ],
        code="scripts/evaluation.py"
    )

    # Create model monitor
    model_monitor = create_model_monitor(role, bucket, base_job_prefix)

    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[],
        steps=[preprocessing_step, training_step, evaluation_step]
    )

    return pipeline, model_monitor

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