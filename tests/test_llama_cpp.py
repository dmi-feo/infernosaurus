import yt.wrapper as yt

from infernosaurus.llm_operator import LLMOperator
from infernosaurus import const as c
from infernosaurus import models
from infernosaurus.models import LLMRequest
from infernosaurus.backends.llama_cpp.backend import LlamaCppOffline


def test_start_and_stop(yt_with_model):
    llm = LLMOperator(
        request=LLMRequest(
            yt_proxy=yt_with_model.proxy_url_http, yt_token="topsecret",
            resources=models.Resources(server_mem=10 * c.GiB, server_cpu=1, worker_num=0),
            model_path="//tmp/the-model.gguf", operation_title="llama's ass"
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
    with LLMOperator(
        backend_type="llama_cpp",
        request=models.LLMRequest(
            yt_proxy=yt_with_model.proxy_url_http,
            yt_token="topsecret",
            resources=models.Resources(
                server_mem=10 * c.GiB,
                server_cpu=1,
                worker_num=0,
            ),
            model_path="//tmp/the-model.gguf",
        )
    ).server() as llm:
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
    with LLMOperator(
            backend_type="llama_cpp",
            request=models.LLMRequest(
                yt_proxy=yt_with_model.proxy_url_http,
                yt_token="topsecret",
                resources=models.Resources(
                    server_mem=3 * c.GiB,
                    server_cpu=1,
                    worker_num=3,
                    worker_cpu=1,
                    worker_mem=3 * c.GiB,
                ),
                model_path="//tmp/the-model.gguf",
            )
    ).server() as llm:
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
        ]
    )

    model = LlamaCppOffline(model_path="//tmp/the-model.gguf", resources=models.Resources(
        server_mem=0, server_cpu=0, worker_num=3, worker_cpu=4, worker_mem=8 * c.GiB,
    ), yt_proxy=yt_with_model.proxy_url_http, yt_token="topsecret")

    model.process(
        input_table="//tmp/my_table", input_column="country",
        output_table="//tmp/new_table", output_column="capital",
        prompt="What is the capital of {{value}}?",
    )

    data = list(yt_cli.read_table("//tmp/new_table"))
    assert "Berlin" in data[0]["capital"]
    assert "Rome" in data[1]["capital"]
    assert "Madrid" in data[2]["capital"]