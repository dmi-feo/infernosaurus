import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-column")
    parser.add_argument("--output-column")
    parser.add_argument("--prompt")
    parser.add_argument("--model-path")

    args = parser.parse_args()

    for line in sys.stdin:
        data = json.loads(line)
        input_row = str(data[args.input_column])

        processed_row = subprocess.check_output(
            ["/llama/bin/llama-cli", "-m", args.model_path, "-p", args.prompt.replace("{{value}}", input_row), "-n", "64"],
        )

        data[args.output_column] = processed_row.decode()
        sys.stdout.write(json.dumps(data))


if __name__ == "__main__":
    main()
