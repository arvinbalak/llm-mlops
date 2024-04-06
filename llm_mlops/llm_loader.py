import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMLoader:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["llm"]["model_path"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["llm"]["tokenizer_name"]
        )

    def generate_text(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0])
