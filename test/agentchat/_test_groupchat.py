from typing import List

import autogen
from aix.data.enums import ChatGPTModel
from aix.data.model_config import OpenAIModelConfig
from autogen.oai.utils.config_list import get_config_list


def _test_groupchat():
    model_config = OpenAIModelConfig(model_type=ChatGPTModel.GPT_4, stream=True)
    conversations: List[str | None] = [None]

    config_conversation_map = {
        model_config: conversations
    }
    config_list = get_config_list(model_config_conversation_map=config_conversation_map)
    llm_config = {"config_list": config_list, "cache_seed": 42}
    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
        human_input_mode="TERMINATE"
    )
    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
    )
    pm = autogen.AssistantAgent(
        name="Product_manager",
        system_message="Creative in software product ideas.",
        llm_config=llm_config,
    )
    pm.client.clear_cache()
    groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    user_proxy.initiate_chat(manager,
                             message="Find a latest paper about gpt-4 on arxiv and find its potential applications in software.")
                             # message="Find a latest paper about metaverse on arxiv and find its potential applications of improving user experiences.")
    pass

if __name__ == "__main__":
    _test_groupchat()
    pass
