import json
import os

import autogen
from aix.utils.paths import get_config_file_path
from autogen import UserProxyAgent
from autogen.agentchat.contrib.local_agent_builder import AgentBuilder
from autogen.oai.utils.config_list import get_config_list

here = os.path.abspath(os.path.dirname(__file__))
config_path = get_config_file_path('config_list.json')


def _test_build():
    builder = AgentBuilder(builder_model="gpt-4", agent_model="gpt-4")
    building_task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )
    coding = None
    builder.build(
        building_task=building_task,
        default_llm_config={"temperature": 0},
        code_execution_config={
            "last_n_messages": 2,
            "work_dir": f"{here}/test_agent_scripts",
            "timeout": 60,
            "use_docker": "python:3",
        },
        coding=coding,
    )

    # saved_files = builder.save(get_config_file_path("example_save_agent_builder_config.json"))
    #
    # # check config file path
    # assert os.path.isfile(saved_files)
    #
    # saved_configs = json.load(open(saved_files))

    # check config format
    # assert saved_configs.get("building_task", None) is not None
    # assert saved_configs.get("agent_configs", None) is not None
    # assert saved_configs.get("coding", None) is not None
    # assert saved_configs.get("default_llm_config", None) is not None

    # check number of agents
    assert len(builder.agents) <= builder.max_agents + 1

    # check system message
    for agent in builder.agents:
        if not isinstance(agent, UserProxyAgent):
            assert "TERMINATE" in agent.system_message

    return builder.agents


def start_task(execution_task: str, agent_list: list | set, llm_config: dict):
    config_list = autogen.config_list_from_json(config_path, filter_dict={"model": ["gpt-4"]})

    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}
    )
    agent_list[0].initiate_chat(manager, message=execution_task)


def _test_load():
    builder = AgentBuilder(builder_model="gpt-4", agent_model="gpt-4")

    config_save_path = get_config_file_path("example_save_agent_builder_config.json")
    configs = json.load(open(config_save_path))
    agent_configs = {
        e["name"]: {"model": e["model"], "system_message": e["system_message"]} for e in configs["agent_configs"]
    }

    agent_list, loaded_agent_configs = builder.load(
        config_save_path,
        code_execution_config={
            "last_n_messages": 2,
            "work_dir": f"{here}/test_agent_scripts",
            "timeout": 60,
            "use_docker": "python:3",
        },
    )

    # check config loading
    assert loaded_agent_configs["coding"] == configs["coding"]
    if loaded_agent_configs["coding"] is True:
        assert isinstance(agent_list[0], UserProxyAgent)
        agent_list = agent_list[1:]
    for agent in agent_list:
        agent_name = agent.name
        assert agent_configs.get(agent_name, None) is not None
        assert agent_configs[agent_name]["model"] == agent.llm_config["model"]
        assert agent_configs[agent_name]["system_message"] == agent.system_message
    pass


if __name__ == "__main__":
    default_llm_config = {
        'temperature': 0
    }
    agents = _test_build()
    if agents:
        agents[0].client.clear_cache()
    start_task(
        execution_task="Find a recent paper about gpt-4 on arxiv and find its potential applications in software.",
        agent_list=agents,
        llm_config=default_llm_config
    )
    # _test_load()
