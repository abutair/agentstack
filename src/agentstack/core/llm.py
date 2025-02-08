from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import openai
from .config import AgentStackConfig,LLMConfig

from .types import Message

class LLMResponse(BaseModel):
    content: str
    model: str
    usage:Dict[str,int]
    metadata: Dict[str,Any]={}


class BaseLLM(ABC):
    """Base class for LLM Providers"""

    @abstractmethod
    async def generate(self, messages:List[str,str],**kwargs)->LLMResponse:
        """Generate response from the llm."""
        pass

    @abstractmethod
    async def stream(self,messages: List[Dict[str, str]],**kwargs):    
        """Stream responses from the LLM."""
        pass
    

class OpenAILLM(BaseLLM):    
    def __init__(self, config: LLMConfig):
        self.config = config
        openai.api_key = config.api_key
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        try:
            response = await openai.ChatCompletion.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=False,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                }
            )
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}")
    
    async def stream( self,messages: List[Dict[str, str]],**kwargs):
        try:
            async for chunk in await openai.ChatCompletion.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **kwargs
            ):
                if chunk and chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"OpenAI API streaming error: {str(e)}")
        

class LLMError(Exception):
    """Base excepption for LLM-related errors."""
    pass

class LLMFactory:
    """Factory for creating LLM instances."""
    provoders ={
        "openai": OpenAILLM
    }

    @classmethod
    def create(cls, provoders:str=None, **kwargs) ->BaseLLM:
        config = AgentStackConfig()
        provider = provider or config.llm.provider
        
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        return cls._providers[provider](config.llm)
    

    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new LLM provider."""
        cls._providers[name] = provider_class


    def convert_messages_to_llm_format(messages: List[Message]) -> List[Dict[str, str]]:
        """Convert internal message format to LLM-compatible format."""
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]