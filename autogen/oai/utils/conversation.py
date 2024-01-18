from typing import Any

from pydantic import BaseModel, field_validator


class Conversation(BaseModel):
    sender: Any = None
    recipient: Any = None
    conversation_id: str | None = None
    bot_id: str = None

    @field_validator('sender', 'recipient')
    @classmethod
    def validate_agent(cls, v):
        from autogen import Agent
        if v is not None and not isinstance(v, (Agent, str)):
            raise TypeError("sender or recipient must be of type Agent or None.")
        return v

    def __hash__(self):
        return hash((self.sender, self.recipient))
