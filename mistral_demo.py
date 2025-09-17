# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# tokenizer = AutoTokenizer.from_pretrained(
#     model_id,
#     use_auth_token=True,
#     trust_remote_code=True
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     use_auth_token=True,
#     trust_remote_code=True
# )

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)


from transformers import AutoModelForCausalLM, AutoTokenizer

local_path = r"E:\StudyMate\mistral-7b"   # absolute or relative path

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(local_path)
