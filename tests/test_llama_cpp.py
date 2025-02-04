import pytest
import yt.wrapper as yt

from infernosaurus.inference_operator import OnlineInferenceOperator, OfflineInferenceOperator
from infernosaurus import const as c
from infernosaurus import models
from infernosaurus.utils import get_random_string


CLIENT_CONFIG_PATCH = {"is_local_mode": True, "proxy": {"enable_proxy_discovery": False}}


def test_start_and_stop(yt_with_model):
    llm = OnlineInferenceOperator(
        runtime_config=models.OnlineInferenceRuntimeConfig(
            yt_settings=models.YtSettings(
                proxy_url=yt_with_model.proxy_url_http, token="topsecret",
                client_config_patch=CLIENT_CONFIG_PATCH,
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
                client_config_patch=CLIENT_CONFIG_PATCH,
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

        assert "pie" in content.lower()


def test_with_workers(yt_with_model):
    with OnlineInferenceOperator(
        backend_type="llama_cpp",
        runtime_config=models.OnlineInferenceRuntimeConfig(
            yt_settings=models.YtSettings(
                proxy_url=yt_with_model.proxy_url_http, token="topsecret",
                client_config_patch=CLIENT_CONFIG_PATCH,
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


@pytest.mark.parametrize("model_workers", [(0, None), (2, models.Resources(cpu=1, mem=3 * c.GiB))])
def test_offline(yt_with_model, model_workers):
    yt_cli: yt.YtClient = yt_with_model.get_client(token="topsecret")
    table_path = f"//tmp/{get_random_string(13)}"
    yt_cli.create("table", table_path)
    yt_cli.write_table(
        table_path,
        [
            {"number": "one", "country": "Germany", "true_answer": "Berlin"},
            {"number": "two", "country": "Italy", "true_answer": "Rome"},
            {"number": "three", "country": "Spain", "true_answer": "Madrid"},
            {"number": "four", "country": "France", "true_answer": "Paris"},
            {"number": "five", "country": "Armenia", "true_answer": "Yerevan"},
            {"number": "six", "country": "Serbia", "true_answer": "Belgrade"},
        ]
    )

    llm = OfflineInferenceOperator(
        backend_type="llama_cpp",
        runtime_config=models.OfflineInferenceRuntimeConfig(
            yt_settings=models.YtSettings(
                proxy_url=yt_with_model.proxy_url_http, token="topsecret",
                client_config_patch=CLIENT_CONFIG_PATCH,
            ),
            worker_num=2,
            worker_resources=models.Resources(cpu=4, mem=8 * c.GiB),
            model_worker_num=model_workers[0],
            model_worker_resources=model_workers[1],
        )
    )
    llm.process(models.OfflineInferenceRequest(
        input_table=table_path, input_column="country",
        output_table=table_path, output_column="answer",
        prompt="Question: What is the capital of {{value}}? Answer:",
        model_path="//tmp/the-model.gguf", echo=True,
        inference_parameters=models.InferenceParameters(max_tokens=64, temperature=0.2),
    ))

    data = list(yt_cli.read_table(table_path))

    for idx, row in enumerate(data):
        assert row["true_answer"] in row["answer"], row
