from unittest import mock
import builtins
import json

import autogen
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.oai.utils.config_list import get_config_list


def _test_2_agents():
    config_list = get_config_list()
    # create an AssistantAgent named "assistant"
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={
            "cache_seed": 42,  # seed for caching and reproducibility
            "config_list": config_list,  # a list of OpenAI API configurations
            "temperature": 0,  # temperature for sampling
        },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
    )
    assistant.client.clear_cache()
    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # set to True or image name like "python:3" to use docker
        },
    )
    # the assistant receives a message from the user_proxy, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message="""What date is today? Write a function to compare the year-to-date gain for META and TESLA.""",
    )


def _test_math_user_proxy_agent():
    from autogen.agentchat.assistant_agent import AssistantAgent

    conversations = {}
    # autogen.ChatCompletion.start_logging(conversations)

    config_list = get_config_list()
    assistant = AssistantAgent(
        "assistant",
        system_message="You are a helpful assistant.",
        llm_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
        },
    )
    assistant.client.clear_cache()

    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")
    assistant.reset()

    math_problem = "$x^3=125$. What is x?"
    # assistant.receive(
    #     message=mathproxyagent.generate_init_message(math_problem),
    #     sender=mathproxyagent,
    # )
    mathproxyagent.initiate_chat(assistant, problem=math_problem)
    print(conversations)


def _test_execute_one_python_code():
    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")

    # no output found 1
    code = "x=3"
    assert mathproxyagent.execute_one_python_code(code)[0] == "No output found. Make sure you print the results."

    # no output found 2
    code = "if 4 > 5:\n\tprint('True')"

    assert mathproxyagent.execute_one_python_code(code)[0] == "No output found."

    # return error
    code = "2+'2'"
    assert "Error:" in mathproxyagent.execute_one_python_code(code)[0]

    # save previous status
    mathproxyagent.execute_one_python_code("x=3\ny=x*2")
    assert mathproxyagent.execute_one_python_code("print(y)")[0].strip() == "6"

    code = "print('*'*2001)"
    assert (
        mathproxyagent.execute_one_python_code(code)[0]
        == "Your requested query response is too long. You might have made a mistake. Please revise your reasoning and query."
    )


def _test_chat_manager():
    agent1 = autogen.ConversableAgent(
        "alice",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice speaking.",
        function_map={"test_func": lambda x: x},
    )
    agent2 = autogen.ConversableAgent(
        "bob",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob speaking.",
        function_map={"test_func": lambda x: x},
    )
    groupchat = autogen.GroupChat(agents=[agent1, agent2], messages=[], max_round=2)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=False)
    agent1.initiate_chat(group_chat_manager, message="hello")

    assert len(agent1.chat_messages[group_chat_manager]) == 2
    assert len(groupchat.messages) == 2

    group_chat_manager.reset()
    assert len(groupchat.messages) == 0
    agent1.reset()
    agent2.reset()
    agent2.initiate_chat(group_chat_manager, message="hello")
    assert len(groupchat.messages) == 2

    try:
        agent2.initiate_chat(group_chat_manager, message={"function_call": {"name": "test_func", "arguments": '{"x": 1}'}})
    except Exception as e:
        print(str(e))
        raise


def _test_func_call_groupchat():
    agent1 = autogen.ConversableAgent(
        "alice",
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice speaking.",
    )
    agent2 = autogen.ConversableAgent(
        "bob",
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob speaking.",
        function_map={"test_func": lambda x: x},
    )
    groupchat = autogen.GroupChat(agents=[agent1, agent2], messages=[], max_round=3)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=False)
    agent2.initiate_chat(group_chat_manager, message={"function_call": {"name": "test_func", "arguments": '{"x": 1}'}})

    assert len(groupchat.messages) == 3
    assert (
        groupchat.messages[-2]["role"] == "function"
        and groupchat.messages[-2]["name"] == "test_func"
        and groupchat.messages[-2]["content"] == "1"
    )
    assert groupchat.messages[-1]["name"] == "alice"

    agent3 = autogen.ConversableAgent(
        "carol",
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is carol speaking.",
        function_map={"test_func": lambda x: x + 1},
    )
    groupchat = autogen.GroupChat(agents=[agent1, agent2, agent3], messages=[], max_round=3)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=False)
    agent3.initiate_chat(group_chat_manager, message={"function_call": {"name": "test_func", "arguments": '{"x": 1}'}})

    assert (
        groupchat.messages[-2]["role"] == "function"
        and groupchat.messages[-2]["name"] == "test_func"
        and groupchat.messages[-2]["content"] == "1"
    )
    assert groupchat.messages[-1]["name"] == "carol"

    agent2.initiate_chat(group_chat_manager, message={"function_call": {"name": "test_func", "arguments": '{"x": 1}'}})


def __test_selection_method(method: str):
    agent1 = autogen.ConversableAgent(
        "alice",
        max_consecutive_auto_reply=10,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice speaking.",
    )
    agent2 = autogen.ConversableAgent(
        "bob",
        max_consecutive_auto_reply=10,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob speaking.",
    )
    agent3 = autogen.ConversableAgent(
        "charlie",
        max_consecutive_auto_reply=10,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is charlie speaking.",
    )

    groupchat = autogen.GroupChat(
        agents=[agent1, agent2, agent3],
        messages=[],
        max_round=6,
        speaker_selection_method=method,
        allow_repeat_speaker=False if method == "manual" else True,
    )
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=False)

    if method == "round_robin":
        agent1.initiate_chat(group_chat_manager, message="This is alice speaking.")
        assert len(agent1.chat_messages[group_chat_manager]) == 6
        assert len(groupchat.messages) == 6
        assert [msg["content"] for msg in agent1.chat_messages[group_chat_manager]] == [
            "This is alice speaking.",
            "This is bob speaking.",
            "This is charlie speaking.",
        ] * 2
    elif method == "auto":
        agent1.initiate_chat(group_chat_manager, message="This is alice speaking.")
        assert len(agent1.chat_messages[group_chat_manager]) == 6
        assert len(groupchat.messages) == 6
    elif method == "random":
        agent1.initiate_chat(group_chat_manager, message="This is alice speaking.")
        assert len(agent1.chat_messages[group_chat_manager]) == 6
        assert len(groupchat.messages) == 6
    elif method == "manual":
        for user_input in ["", "q", "x", "1", "10"]:
            with mock.patch.object(builtins, "input", lambda _: user_input):
                group_chat_manager.reset()
                agent1.reset()
                agent2.reset()
                agent3.reset()
                agent1.initiate_chat(group_chat_manager, message="This is alice speaking.")
                if user_input == "1":
                    assert len(agent1.chat_messages[group_chat_manager]) == 6
                    assert len(groupchat.messages) == 6
                    assert [msg["content"] for msg in agent1.chat_messages[group_chat_manager]] == [
                        "This is alice speaking.",
                        "This is bob speaking.",
                        "This is alice speaking.",
                        "This is bob speaking.",
                        "This is alice speaking.",
                        "This is bob speaking.",
                    ]
                else:
                    assert len(agent1.chat_messages[group_chat_manager]) == 6
                    assert len(groupchat.messages) == 6
    elif method == "wrong":
        try:
            agent1.initiate_chat(group_chat_manager, message="This is alice speaking.")
        except Exception as e:
            print(str(e))


def _test_speaker_selection_method():
    for method in ["auto", "round_robin", "random", "manual", "wrong", "RounD_roBin"]:
        __test_selection_method(method)


def _test_agent_mentions():
    agent1 = autogen.ConversableAgent(
        "alice",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice speaking.",
    )
    agent2 = autogen.ConversableAgent(
        "bob",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob speaking.",
    )
    agent3 = autogen.ConversableAgent(
        "sam",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is sam speaking.",
    )
    groupchat = autogen.GroupChat(agents=[agent1, agent2, agent3], messages=[], max_round=2)

    # Basic counting
    assert json.dumps(groupchat._mentioned_agents("", [agent1, agent2, agent3]), sort_keys=True) == "{}"
    assert json.dumps(groupchat._mentioned_agents("alice", [agent1, agent2, agent3]), sort_keys=True) == '{"alice": 1}'
    assert (
        json.dumps(groupchat._mentioned_agents("alice bob alice", [agent1, agent2, agent3]), sort_keys=True)
        == '{"alice": 2, "bob": 1}'
    )
    assert (
        json.dumps(groupchat._mentioned_agents("alice bob alice sam", [agent1, agent2, agent3]), sort_keys=True)
        == '{"alice": 2, "bob": 1, "sam": 1}'
    )
    assert (
        json.dumps(groupchat._mentioned_agents("alice bob alice sam robert", [agent1, agent2, agent3]), sort_keys=True)
        == '{"alice": 2, "bob": 1, "sam": 1}'
    )

    # Substring
    assert (
        json.dumps(groupchat._mentioned_agents("sam samantha basam asami", [agent1, agent2, agent3]), sort_keys=True)
        == '{"sam": 1}'
    )

    # Word boundaries
    assert (
        json.dumps(groupchat._mentioned_agents("alice! .alice. .alice", [agent1, agent2, agent3]), sort_keys=True)
        == '{"alice": 3}'
    )

    # Special characters in agent names
    agent4 = autogen.ConversableAgent(
        ".*",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="Match everything.",
    )

    groupchat = autogen.GroupChat(agents=[agent1, agent2, agent3, agent4], messages=[], max_round=2)
    assert (
        json.dumps(
            groupchat._mentioned_agents("alice bob alice sam robert .*", [agent1, agent2, agent3, agent4]),
            sort_keys=True,
        )
        == '{".*": 1, "alice": 2, "bob": 1, "sam": 1}'
    )


if __name__ == "__main__":
    # _test_2_agents()
    # _test_execute_one_python_code()
    # _test_math_user_proxy_agent()
    # _test_chat_manager()
    # _test_func_call_groupchat()
    # _test_speaker_selection_method()
    _test_agent_mentions()
    pass
