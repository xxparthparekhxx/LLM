import json

with open("configs/T4_safe_model.json", "r") as f:
    config = json.load(f)

print(config)

#testout if config is valid
assert config["model"]["n_embd"] % config["model"]["n_heads"] == 0