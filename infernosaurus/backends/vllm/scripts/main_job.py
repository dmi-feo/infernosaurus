import argparse
import json
import sys

from vllm import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-column")
    parser.add_argument("--output-column")
    parser.add_argument("--prompt")
    parser.add_argument("--model-path")

    args = parser.parse_args()

    llm = LLM(args.model_path)

    for line in sys.stdin:
        data = json.loads(line)
        input_row = str(data[args.input_column])
        prepared_prompt = args.prompt.replace("{{value}}", input_row)
        processed_row = llm.generate(prepared_prompt)[0].outputs[0].text
        data[args.output_column] = processed_row
        sys.stdout.write(json.dumps(data))


if __name__ == "__main__":
    main()
