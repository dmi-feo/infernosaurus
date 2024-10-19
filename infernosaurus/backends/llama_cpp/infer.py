import argparse
import json
import sys

from llama_cpp import Llama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-column")
    parser.add_argument("--output-column")
    parser.add_argument("--prompt")
    parser.add_argument("--model-path")

    args = parser.parse_args()

    llm = Llama(model_path=args.model_path)

    for line in sys.stdin:
        data = json.loads(line)
        input_row = str(data[args.input_column])
        prepared_prompt = args.prompt.replace("{{value}}", input_row)
        processed_row = llm.create_completion(prepared_prompt, max_tokens=256, echo=True)["choices"][0]["text"]
        data[args.output_column] = processed_row
        sys.stdout.write(json.dumps(data))


if __name__ == "__main__":
    main()
