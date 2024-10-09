import yt.wrapper as yt

from infernosaurus.inference_operator import OnlineInferenceOperator, OfflineInferenceOperator
from infernosaurus import const as c
from infernosaurus import models
from infernosaurus.backends.llama_cpp.backend import LlamaCppOffline
from infernosaurus.models import OfflineInferenceRequest


def test_start_and_stop(yt_with_model):
    llm = OnlineInferenceOperator(
        runtime_config=models.OnlineInferenceRuntimeConfig(
            yt_settings=models.YtSettings(
                proxy_url=yt_with_model.proxy_url_http, token="topsecret",
            ),
            server_resources=models.Resources(mem=10 * c.GiB, cpu=1),
            model_path="//tmp/the-model.gguf",
            operation_title="llama's ass"
        ),
        backend_type="llama_cpp",
    )
    try:
        llm.start()

        yt_cli: yt.YtClient = yt_with_model.get_client(token="topsecret")

        ops = yt_cli.list_operations(state="running")["operations"]
        assert len(ops) == 1
        op = ops[0]
        assert op["brief_spec"]["title"] == "llama's ass"
    finally:
        try:
            llm.stop()
        except Exception:
            raise


def test_server_only(yt_with_model):
    with OnlineInferenceOperator(
        backend_type="llama_cpp",
        runtime_config=models.OnlineInferenceRuntimeConfig(
            yt_settings=models.YtSettings(
                proxy_url=yt_with_model.proxy_url_http, token="topsecret",
            ),
            server_resources=models.Resources(mem=10 * c.GiB, cpu=1),
            model_path="//tmp/the-model.gguf",
        )
    ) as llm:
        openai_client = llm.get_openai_client()

        chat_completion = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Write a recipe of an apple pie",
                }
            ],
            model="the-model.gguf",
        )
        content = chat_completion.choices[0].message.content

        chat_completion = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"What is the following text about?\n{content}",
                }
            ],
            model="the-model.gguf",
        )
        content = chat_completion.choices[0].message.content

        assert "apple pie" in content.lower()


def test_with_workers(yt_with_model):
    with OnlineInferenceOperator(
        backend_type="llama_cpp",
        runtime_config=models.OnlineInferenceRuntimeConfig(
            yt_settings=models.YtSettings(
                proxy_url=yt_with_model.proxy_url_http, token="topsecret",
            ),
            server_resources=models.Resources(mem=3 * c.GiB, cpu=1),
            worker_num=3,
            worker_resources=models.Resources(mem=3 * c.GiB, cpu=1),
            model_path="//tmp/the-model.gguf",
        )
    ) as llm:
        openai_client = llm.get_openai_client()

        chat_completion = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of the Netherlands?",
                }
            ],
            model="the-model.gguf",
        )
        content = chat_completion.choices[0].message.content
        assert "Amsterdam" in content


def test_offline(yt_with_model):
    yt_cli: yt.YtClient = yt_with_model.get_client(token="topsecret")
    yt_cli.create("table", "//tmp/my_table")
    yt_cli.write_table(
        "//tmp/my_table",
        [
            {"number": "one", "country": "Germany"},
            {"number": "two", "country": "Italy"},
            {"number": "three", "country": "Spain"},
            {"number": "four", "country": "France"},
            {"number": "five", "country": "Armenia"},
            {"number": "six", "country": "Serbia"},
        ]
    )

    llm = OfflineInferenceOperator(
        backend_type="llama_cpp",
        runtime_config=models.OfflineInferenceRuntimeConfig(
            yt_settings=models.YtSettings(
                proxy_url=yt_with_model.proxy_url_http, token="topsecret",
            ),
            worker_num=2,
            worker_resources=models.Resources(cpu=4, mem=8 * c.GiB),
        )
    )
    llm.process(models.OfflineInferenceRequest(
        input_table="//tmp/my_table", input_column="country",
        output_table="//tmp/new_table", output_column="capital",
        prompt="What is the capital of {{value}}?",
        model_path="//tmp/the-model.gguf",
    ))

    data = list(yt_cli.read_table("//tmp/new_table"))

    true_answers = ("Berlin", "Rome", "Madrid", "Paris", "Yerevan", "Belgrade")
    for idx, row in enumerate(data):
        assert true_answers[idx] in row["capital"]
