from sagemaker.huggingface.model import HuggingFaceModel

HF_MODEL_ID   = "BAAI/bge-reranker-v2-m3"
HF_TASK       = "text-classification"   # cross-encoder = classification
ROLE_ARN      = "arn:aws:iam::084128132181:role/sagemaker-role"
INSTANCE_TYPE = "ml.g5.2xlarge"

hub_env = {
    "HF_MODEL_ID": HF_MODEL_ID,
    "HF_TASK"    : HF_TASK,
}

hf_model = HuggingFaceModel(
    env                   = hub_env,
    role                  = ROLE_ARN,
    transformers_version  = "4.26.0",
    pytorch_version       = "1.13.1",
    py_version            = "py39",
)

predictor = hf_model.deploy(
    initial_instance_count = 1,
    instance_type          = INSTANCE_TYPE,
    endpoint_name          = "bge-reranker-endpoint"
)

print("Endpoint deployed:", predictor.endpoint_name)
