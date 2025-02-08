from typing import Dict,Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel,Field
import os

class LLMConfig(BaseModel):
    provider:str 
    model:str 
    temperature:float
    max_tokens :int
    api_key:str

class RuntimeConfig(BaseModel):
    workspace_dir: Path = Path("./workspace")
    allow_code_execution: bool = False
    max_iterations: int = 10
    memory_backend: str = "vector"

class ToolConfig(BaseModel):
    enabled_tools: list[str] = ["web_search", "code_execution"]
    tool_configs: Dict[str, Any] = Field(default_factory=dict)

class AgentStackConfig:
    _instance = None 

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.llm = LLMConfig()
            self.runtime = RuntimeConfig()
            self.tools = ToolConfig()
            self.initialized = True
    
    @classmethod
    def load(cls, config_path: str) -> 'AgentStackConfig':
        instance = cls()
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        if 'llm' in config_data:
            instance.llm = LLMConfig(**config_data['llm'])
        if 'runtime' in config_data:
            instance.runtime = RuntimeConfig(**config_data['runtime'])
        if 'tools' in config_data:
            instance.tools = ToolConfig(**config_data['tools'])
        
        return instance
    
    def save(self, config_path: str) -> None:
        config_data = {
            'llm': self.llm.model_dump(),
            'runtime': self.runtime.model_dump(),
            'tools': self.tools.model_dump()
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)


example_config = """
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 2000

runtime:
  workspace_dir: ./workspace
  allow_code_execution: false
  max_iterations: 10
  memory_backend: vector

tools:
  enabled_tools:
    - web_search
    - code_execution
  tool_configs:
    web_search:
      api_key: your_api_key
    code_execution:
      timeout: 30
"""