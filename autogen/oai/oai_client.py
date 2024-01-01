from __future__ import annotations

import inspect
import os
from typing import Any, Union, Mapping, Dict

import httpx
from openai import APIStatusError, OpenAIError
from typing_extensions import Self

from aix.data.llm.unlimited_gpt.UnlimitedGPT import ChatGPT
from aix.utils.pages import get_chatbot
from autogen.oai.oai_completion import Completions, Chat

DEFAULT_MAX_RETRIES = 2
__version__ = "1.5.0"


class OpenAI:
    # client options
    api_key: str
    organization: str | None
    chatgpt_kwargs = set(inspect.getfullargspec(ChatGPT.__init__).kwonlyargs)
    completions: Completions
    chat: Chat
    chatbots: Dict[str | None, ChatGPT]

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        timeout: Union[float, None] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client=None,
        _strict_response_validation: bool = False,
        **kwargs,
    ) -> None:
        """Construct a new synchronous openai client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        """
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        self.api_key = api_key

        if organization is None:
            organization = os.environ.get("OPENAI_ORG_ID", None)
        self.organization = organization

        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url is None:
            base_url = f"https://api.openai.com/v1"

        self._version = __version__
        self._base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._custom_headers = default_headers or {}
        self._custom_query = default_query or {}
        self._strict_response_validation = _strict_response_validation
        self._idempotency_header = None
        self._http_client = http_client
        self._strict_response_validation = _strict_response_validation,

        # self._default_stream_cls = Stream
        #
        self.completions = Completions(self)
        self.chat = Chat(self)
        # self.edits = resources.Edits(self)
        # self.embeddings = resources.Embeddings(self)
        # self.files = resources.Files(self)
        # self.images = resources.Images(self)
        # self.audio = resources.Audio(self)
        # self.moderations = resources.Moderations(self)
        # self.models = resources.Models(self)
        # self.fine_tuning = resources.FineTuning(self)
        # self.fine_tunes = resources.FineTunes(self)
        # self.beta = resources.Beta(self)
        # self.with_raw_response = OpenAIWithRawResponse(self)

        self._chatbot_kwargs = kwargs
        self.chatbots = {}
        self._chatbots: Dict[str, ChatGPT] = {}

    def get_chatbot(self, conversation_id: str = None) -> ChatGPT:
        chatbot = self.chatbots.get(conversation_id, None)
        if chatbot is None:
            chatbot = get_chatbot(**(self._chatbot_kwargs | {"conversation_id": conversation_id}))
            if conversation_id is not None:
                self.chatbots[conversation_id] = chatbot
        return chatbot

    def get_chatbot_1(self, id: str = None, **kwargs) -> ChatGPT:
        chatbot = self._chatbots.get(id, None)
        if chatbot is None:
            chatbot = get_chatbot(**(self._chatbot_kwargs | kwargs))
            if chatbot:
                self._chatbots[chatbot.id] = chatbot
            else:
                raise RuntimeError(f"Failed to create chatbot {id}")
        return chatbot

    def copy(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client=None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._http_client
        return self.__class__(
            api_key=api_key or self.api_key,
            organization=organization or self.organization,
            base_url=base_url or self._base_url,
            timeout= timeout or self.timeout,
            http_client=http_client,
            max_retries=max_retries or self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @staticmethod
    def _make_status_error(
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        data = body.get("error", body) if isinstance(body, Mapping) else body
        return APIStatusError(err_msg, response=response, body=data)


class AsyncOpenAI(ChatGPT):
    # client options
    api_key: str
    organization: str | None
    chatgpt_kwargs = set(inspect.getfullargspec(ChatGPT.__init__).kwonlyargs)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, None] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client=None,
        _strict_response_validation: bool = False,
        **kwargs,
    ) -> None:
        """Construct a new async openai client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        """
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise OpenAIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )
        self.api_key = api_key

        if organization is None:
            organization = os.environ.get("OPENAI_ORG_ID")
        self.organization = organization

        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url is None:
            base_url = f"https://api.openai.com/v1"

        self._version = __version__
        self._base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._custom_headers = default_headers or {}
        self._custom_query = default_query or {}
        self._strict_response_validation = _strict_response_validation
        self._idempotency_header = None
        self._http_client = http_client
        self._strict_response_validation = _strict_response_validation,

        super().__init__(**kwargs)

    def copy(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._http_client
        return self.__class__(
            api_key=api_key or self.api_key,
            organization=organization or self.organization,
            base_url=base_url or self._base_url,
            timeout=timeout or self.timeout,
            http_client=http_client,
            max_retries=max_retries or self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @staticmethod
    def _make_status_error(
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        data = body.get("error", body) if isinstance(body, Mapping) else body
        return APIStatusError(err_msg, response=response, body=data)


Client = OpenAI

AsyncClient = AsyncOpenAI
