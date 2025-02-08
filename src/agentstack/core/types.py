from enum import Enum
from typing import  Dict,Any, Optional
from pydantic import BaseModel
from datetime import datetime

class AgentType(Enum):
    ASSISTANT = "assistant"
    USER_PROXY = "user_proxy"
    CODE = "code"
    CUSTOM = "custom"


class Message(BaseModel):
    content: str
    role:str
    timestamp:datetime = datetime.now()
    metadata: Dict[str,Any] = {}
    

class AgentConfig(BaseModel):
    name:str
    type:AgentType
    system_message:str
    auto_replay: bool =True
    max_consecutive_replies: int = 10
    metadata: Dict[str, Any] = {}


