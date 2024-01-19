from __future__ import annotations

import logging
import time
from typing import Optional, List, Dict, Literal, Union, Iterator
import asyncio
from typing import TYPE_CHECKING

import httpx
from openai.types.chat import ChatCompletionMessageParam

from aix.data.completion import Completion
from aix.data.enums import ChatGPTModel

if TYPE_CHECKING:
    from autogen.oai.oai_client import OpenAI, AsyncOpenAI
logger = logging.getLogger(__name__)


class SyncAPIResource:
    _client: OpenAI

    def __init__(self, client: OpenAI) -> None:
        self._client = client
        # self._get = client.get
        # self._post = client.post
        # self._patch = client.patch
        # self._put = client.put
        # self._delete = client.delete
        # self._get_api_list = client.get_api_list

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)


class AsyncAPIResource:
    _client: AsyncOpenAI

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client
        # self._get = client.get
        # self._post = client.post
        # self._patch = client.patch
        # self._put = client.put
        # self._delete = client.delete
        # self._get_api_list = client.get_api_list

    @staticmethod
    async def _sleep(seconds: float) -> None:
        await asyncio.sleep(seconds)


class Completions(SyncAPIResource):
    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)

    def create(
        self,
        *,
        model: Union[
            str,
            Literal[
                "babbage-002",
                "davinci-002",
                "gpt-3.5-turbo-instruct",
                "gpt-3.5-turbo",
                "text-davinci-003",
                "text-davinci-002",
                "text-davinci-001",
                "code-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
        ] = None,
        prompt: str = None,
        messages: List[ChatCompletionMessageParam] = None,
        best_of: Optional[int] = None,
        echo: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str], None] = None,
        stream: Optional[Literal[False]] | Literal[True] = False,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: str = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers=None,
        extra_query=None,
        extra_body=None,
        timeout: float | httpx.Timeout | None = None,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        **kwargs
    ) -> Completion | Iterator[Completion]:
        assert (prompt and not messages) or (not prompt and messages)
        prompt_or_messages = prompt or messages
        # chatbot = self._client.get_chatbot(conversation_id=conversation_id)
        _model = ChatGPTModel(model) if model else None
        chatbot = self._client.get_chatbot_1(id=bot_id, model=_model, **kwargs)
        return chatbot.send_message(prompt_or_messages=prompt_or_messages, model=_model,
                                    stream=stream, timeout=timeout)


class Chat(SyncAPIResource):
    completions: Completions

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self.completions = Completions(client)
