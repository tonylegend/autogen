import logging
import os
import json
import time
from queue import Queue

from aix.utils.paths import get_project_root
from .datamodel import AgentWorkFlowConfig, Message
from .utils import extract_successful_code_blocks, get_default_agent_config, get_modified_files
from .workflowmanager import AutoGenWorkFlowManager

_logger = logging.getLogger(__name__)


class ChatManager:
    def __init__(self, max_flows: int = 5) -> None:
        super().__init__()
        from aix.utils.misc import StatusAwareThreadPoolExecutor
        self.executor = StatusAwareThreadPoolExecutor(max_workers=max_flows)

    def start_chat(self, message: Message, history: list = None, flow_config: AgentWorkFlowConfig = None,
                   q: Queue = None, **kwargs):
        task_id = self.executor.submit(self.chat, message=message, history=history, flow_config=flow_config,
                                       q=q, **kwargs)
        return task_id

    @staticmethod
    def chat(message: Message, history: list = None, flow_config: AgentWorkFlowConfig = None, q: Queue = None, **kwargs):
        work_dir = kwargs.get("work_dir", '')
        scratch_dir = os.path.join(get_project_root(), work_dir, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)

        # if no flow config is provided, use the default
        if flow_config is None:
            flow_config = get_default_agent_config(scratch_dir)

        # print("Flow config: ", flow_config)
        flow = AutoGenWorkFlowManager(config=flow_config, history=history or [], work_dir=scratch_dir, q=q)
        message_text = message.content.strip()

        start_time = time.time()

        metadata = {}
        flow.run(message=f"{message_text}", clear_history=False)

        metadata["messages"] = flow.agent_history

        output = ""

        if flow_config.summary_method == "last":
            successful_code_blocks = extract_successful_code_blocks(flow.agent_history)
            last_message = flow.agent_history[-1]["message"]["content"]
            successful_code_blocks = "\n\n".join(successful_code_blocks)
            output = (last_message + "\n" + successful_code_blocks) if successful_code_blocks else last_message
        elif flow_config.summary_method == "llm":
            output = ""
        elif flow_config.summary_method == "none":
            output = ""

        metadata["code"] = ""
        end_time = time.time()
        metadata["time"] = end_time - start_time
        modified_files = get_modified_files(start_time, end_time, scratch_dir, dest_dir=work_dir)
        metadata["files"] = modified_files

        print("Modified files: ", len(modified_files))

        output_message = Message(
            user_id=message.user_id,
            root_msg_id=message.root_msg_id,
            role="assistant",
            content=output,
            metadata=json.dumps(metadata),
            session_id=message.session_id,
        )

        return output_message

    def get_chat_status(self, task_id):
        return self.executor.get_task_status(task_id)

    def get_chat_result(self, task_id):
        try:
            return self.executor.get_task_result(task_id)
        except Exception as e:
            _logger.error(f"Error retrieving chat result: {e}")
            return None

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)

    def monitor_tasks(self):
        self.executor.pretty_print_task_statuses()
