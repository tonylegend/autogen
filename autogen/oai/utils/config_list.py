from typing import List

from aix.data.enums import ChatGPTModel
from aix.data.model_config import OpenAIModelConfig
from autogen.oai.openai_utils import get_config, config_list_from_json

from aix.utils.paths import get_config_file_path

model_config = OpenAIModelConfig(model_type=ChatGPTModel.GPT_3_5_TURBO, stream=True)
conversations: List[str | None] = [None]

config_conversation_map = {
    model_config: conversations
}


# def get_config_list(model_config_conversation_map: dict = config_conversation_map, config_path: str = None,
#                     filter_dict: dict = None) -> List[dict]:
#     if config_path is not None:
#         return config_list_from_json(config_path, filter_dict=filter_dict)
#     config_list = []
#     for mc, convs in model_config_conversation_map.items():
#         for conv in convs:
#             config_dict = get_config(api_key="OPENAI_API_KEY")
#             config_dict.update(model=mc.model_type.value, session_token=mc.chat_token,
#                                conversation_id=conv, stream=mc.stream)
#             config_list.append(config_dict)
#     return config_list


def get_config_list(config_path: str = None, filter_dict: dict = None) -> List[dict]:
    config_path = config_path or get_config_file_path("config_list.py")
    return config_list_from_json(config_path, filter_dict=filter_dict)


def retrieve_assistants_by_name(client, name) -> str:
    """
    Return the assistants with the given name from OAI GPT assistant list
    """
    assistants = client.beta.assistants.list()
    candidate_assistants = []
    for assistant in assistants.data:
        if assistant.name == name:
            candidate_assistants.append(assistant)
    return candidate_assistants