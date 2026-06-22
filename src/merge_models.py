!pip install -q mergekit huggingface_hub

yaml_config = """
models:
  - model: kanishkav/qwen2.5-1.5b-merged
    parameters:
      weight: 0.45
      density: 0.5
  - model: kanishkav/qwen2.5-1.5b-merged-FLANCOT
    parameters:
      weight: 0.55
      density: 0.5
merge_method: dare_ties
base_model: Qwen/Qwen2.5-1.5B
parameters:
  int8_mask: true
dtype: bfloat16
"""


# Write out the config file locally
config_path = "merge.yaml"
with open(config_path, "w") as f:
    f.write(yaml_config.strip())

print(f"Configuration file written successfully to {config_path}")

output_dir = "./qwen2.5-1.5b-strategyqa-merge"

# Execute mergekit via CLI command
!mergekit-yaml merge.yaml {output_dir} --copy-tokenizer --allow-crimes