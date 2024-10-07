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
        data[args.output_column] = data[args.input_column]
        sys.stdout.write(json.dumps(data))


if __name__ == "__main__":
    main()
