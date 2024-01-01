# import pytest

from autogen import config_list_openai_aoai
from autogen.oai.utils.config_list import get_config_list
from test.oai.test_utils import KEY_LOC

from autogen.oai.custom_client import OpenAIWrapper

TOOL_ENABLED = False
try:
    from openai.types.chat import ChatCompletionMessage
except ImportError:
    skip = True
else:
    skip = False
    import openai

    if openai.__version__ >= "1.1.0":
        TOOL_ENABLED = True


# @pytest.mark.skipif(skip, reason="openai>=1 not installed")
def _test_completion():
    config_list = get_config_list()
    client = OpenAIWrapper(config_list=config_list)
    response = client.create(prompt="3+3=", model="gpt-3.5-turbo")
    print(response)
    print(client.extract_text_or_completion_object(response))


# @pytest.mark.skipif(skip, reason="openai>=1 not installed")
def _test_chat_completion():
    config_list = get_config_list()
    client = OpenAIWrapper(config_list=config_list)
    response = client.create(messages=[{"role": "user", "content": "5+5="}])
    print(response)
    print(client.extract_text_or_completion_object(response))


def _test_cost(cache_seed, model):
    config_list = get_config_list()
    client = OpenAIWrapper(config_list=config_list, cache_seed=cache_seed)
    response = client.create(prompt="1+3=", model=model)
    print(response.cost)


if __name__ == "__main__":
    # _test_completion()
    # _test_chat_completion()
    cache_seed = 42
    model = "gpt-3.5-turbo-instruct"
    _test_cost(cache_seed, model)

