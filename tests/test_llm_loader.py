import unittest

from llm_mlops.llm_loader import LLMLoader


class TestLLMLoader(unittest.TestCase):
    def test_generate_text(self):
        llm_loader = LLMLoader("config/llm_config.yaml")
        prompt = "Hello, how are you?"
        generated_text = llm_loader.generate_text(prompt)
        self.assertIsInstance(generated_text, str)
        self.assertTrue(len(generated_text) > 0)


if __name__ == "__main__":
    unittest.main()
