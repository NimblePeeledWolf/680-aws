import sagemaker
from sagemaker.estimator import Estimator

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::6190-7134-9087:role/AmazonSageMaker-ExecutionRole-20240831T214595'

#define ecr image uri
image_uri = '619071349087.dkr.ecr.us-east-2.amazonaws.com/aws-docker:latest'

s3_input_train='s3://sagemaker-studio-619071349087-hpoc4txw2wl/train-data'
s3_input_test='s3://sagemaker-studio-619071349087-hpoc4txw2wl/test-data'


#define the sagemaker estimator

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size=30,
    max_run=3600,
    output_path='s3://your-bucket-name/path/to/output', 
    sagemaker_session=sagemaker_session
)

estimator.fit({'train': s3_input_train, 'test': s3_input_test})