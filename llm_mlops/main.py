from llm_mlops.llm_loader import LLMLoader


def main():
    llm_loader = LLMLoader("config/llm_config.yaml")
    prompt = "Once upon a time"
    generated_text = llm_loader.generate_text(prompt)
    print(generated_text)


if __name__ == "__main__":
    main()
