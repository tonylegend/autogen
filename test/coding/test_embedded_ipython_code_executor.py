import os
import tempfile
from typing import Dict, Union
import uuid
import pytest
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.coding.base import CodeBlock, CodeExecutor
from autogen.coding.factory import CodeExecutorFactory
from autogen.oai.openai_utils import config_list_from_json
from conftest import skip_openai  # noqa: E402

try:
    from autogen.coding.embedded_ipython_code_executor import EmbeddedIPythonCodeExecutor

    skip = False
    skip_reason = ""
except ImportError:
    skip = True
    skip_reason = "Dependencies for EmbeddedIPythonCodeExecutor not installed."


@pytest.mark.skipif(skip, reason=skip_reason)
def test_create() -> None:
    config: Dict[str, Union[str, CodeExecutor]] = {"executor": "ipython-embedded"}
    executor = CodeExecutorFactory.create(config)
    assert isinstance(executor, EmbeddedIPythonCodeExecutor)

    config = {"executor": EmbeddedIPythonCodeExecutor()}
    executor = CodeExecutorFactory.create(config)
    assert executor is config["executor"]


@pytest.mark.skipif(skip, reason=skip_reason)
def test_init() -> None:
    executor = EmbeddedIPythonCodeExecutor(timeout=10, kernel_name="python3", output_dir=".")
    assert executor.timeout == 10 and executor.kernel_name == "python3" and executor.output_dir == "."

    # Try invalid output directory.
    with pytest.raises(ValueError, match="Output directory .* does not exist."):
        executor = EmbeddedIPythonCodeExecutor(timeout=111, kernel_name="python3", output_dir="/invalid/directory")

    # Try invalid kernel name.
    with pytest.raises(ValueError, match="Kernel .* is not installed."):
        executor = EmbeddedIPythonCodeExecutor(timeout=111, kernel_name="invalid_kernel_name", output_dir=".")


@pytest.mark.skipif(skip, reason=skip_reason)
def test_execute_code_single_code_block() -> None:
    executor = EmbeddedIPythonCodeExecutor()
    code_blocks = [CodeBlock(code="import sys\nprint('hello world!')", language="python")]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code == 0 and "hello world!" in code_result.output


@pytest.mark.skipif(skip, reason=skip_reason)
def test_execute_code_multiple_code_blocks() -> None:
    executor = EmbeddedIPythonCodeExecutor()
    code_blocks = [
        CodeBlock(code="import sys\na = 123 + 123\n", language="python"),
        CodeBlock(code="print(a)", language="python"),
    ]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code == 0 and "246" in code_result.output

    msg = """
def test_function(a, b):
    return a + b
"""
    code_blocks = [
        CodeBlock(code=msg, language="python"),
        CodeBlock(code="test_function(431, 423)", language="python"),
    ]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code == 0 and "854" in code_result.output


@pytest.mark.skipif(skip, reason=skip_reason)
def test_execute_code_bash_script() -> None:
    executor = EmbeddedIPythonCodeExecutor()
    # Test bash script.
    code_blocks = [CodeBlock(code='!echo "hello world!"', language="bash")]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code == 0 and "hello world!" in code_result.output


@pytest.mark.skipif(skip, reason=skip_reason)
def test_timeout() -> None:
    executor = EmbeddedIPythonCodeExecutor(timeout=1)
    code_blocks = [CodeBlock(code="import time; time.sleep(10); print('hello world!')", language="python")]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code and "Timeout" in code_result.output


@pytest.mark.skipif(skip, reason=skip_reason)
def test_silent_pip_install() -> None:
    executor = EmbeddedIPythonCodeExecutor()
    code_blocks = [CodeBlock(code="!pip install matplotlib numpy", language="python")]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code == 0 and code_result.output.strip() == ""

    none_existing_package = uuid.uuid4().hex
    code_blocks = [CodeBlock(code=f"!pip install matplotlib_{none_existing_package}", language="python")]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code == 0 and "ERROR: " in code_result.output


@pytest.mark.skipif(skip, reason=skip_reason)
def test_restart() -> None:
    executor = EmbeddedIPythonCodeExecutor()
    code_blocks = [CodeBlock(code="x = 123", language="python")]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code == 0 and code_result.output.strip() == ""

    executor.restart()
    code_blocks = [CodeBlock(code="print(x)", language="python")]
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code and "NameError" in code_result.output


@pytest.mark.skipif(skip, reason=skip_reason)
def test_save_image() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = EmbeddedIPythonCodeExecutor(output_dir=temp_dir)
        # Install matplotlib.
        code_blocks = [CodeBlock(code="!pip install matplotlib", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and code_result.output.strip() == ""

        # Test saving image.
        code_blocks = [
            CodeBlock(code="import matplotlib.pyplot as plt\nplt.plot([1, 2, 3, 4])\nplt.show()", language="python")
        ]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0
        assert os.path.exists(code_result.output_files[0])
        assert f"Image data saved to {code_result.output_files[0]}" in code_result.output


@pytest.mark.skipif(skip, reason=skip_reason)
def test_save_html() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = EmbeddedIPythonCodeExecutor(output_dir=temp_dir)
        # Test saving html.
        code_blocks = [
            CodeBlock(code="from IPython.display import HTML\nHTML('<h1>Hello, world!</h1>')", language="python")
        ]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0
        assert os.path.exists(code_result.output_files[0])
        assert f"HTML data saved to {code_result.output_files[0]}" in code_result.output


@pytest.mark.skipif(skip, reason=skip_reason)
@pytest.mark.skipif(skip_openai, reason="openai not installed OR requested to skip")
def test_conversable_agent_capability() -> None:
    KEY_LOC = "notebook"
    OAI_CONFIG_LIST = "OAI_CONFIG_LIST"
    config_list = config_list_from_json(
        OAI_CONFIG_LIST,
        file_location=KEY_LOC,
        filter_dict={
            "model": {
                "gpt-3.5-turbo",
                "gpt-35-turbo",
            },
        },
    )
    llm_config = {"config_list": config_list}
    agent = ConversableAgent(
        "coding_agent",
        llm_config=llm_config,
        code_execution_config=False,
    )
    executor = EmbeddedIPythonCodeExecutor()
    executor.user_capability.add_to_agent(agent)

    # Test updated system prompt.
    assert executor.DEFAULT_SYSTEM_MESSAGE_UPDATE in agent.system_message

    # Test code generation.
    reply = agent.generate_reply(
        [{"role": "user", "content": "print 'hello world' to the console in a single python code block"}],
        sender=ConversableAgent("user", llm_config=False, code_execution_config=False),
    )

    # Test code extraction.
    code_blocks = executor.code_extractor.extract_code_blocks(reply)  # type: ignore[arg-type]
    assert len(code_blocks) == 1 and code_blocks[0].language == "python"

    # Test code execution.
    code_result = executor.execute_code_blocks(code_blocks)
    assert code_result.exit_code == 0 and "hello world" in code_result.output.lower()


@pytest.mark.skipif(skip, reason=skip_reason)
def test_conversable_agent_code_execution() -> None:
    agent = ConversableAgent(
        "user_proxy",
        llm_config=False,
        code_execution_config={"executor": "ipython-embedded"},
    )
    msg = """
Run this code:
```python
def test_function(a, b):
    return a * b
```
And then this:
```python
print(test_function(123, 4))
```
"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENAI_API_KEY", "mock")
        reply = agent.generate_reply(
            [{"role": "user", "content": msg}],
            sender=ConversableAgent("user", llm_config=False, code_execution_config=False),
        )
        assert "492" in reply  # type: ignore[operator]
