"""
Configuration management for AI providers and MCP servers.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AIProviderConfig:
    """Configuration for AI providers"""
    name: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60


@dataclass
class MCPConfig:
    """Configuration for MCP servers"""
    name: str
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        self.ai_provider = os.getenv("AI_PROVIDER", "claude").lower()
        self._load_ai_provider_config()
        self._load_mcp_config()
    
    def _load_ai_provider_config(self):
        """Load AI provider specific configuration"""
        if self.ai_provider == "openai":
            self.ai_config = AIProviderConfig(
                name="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                timeout=int(os.getenv("OPENAI_TIMEOUT", "60"))
            )
        elif self.ai_provider == "claude":
            self.ai_config = AIProviderConfig(
                name="claude",
                # Claude Code doesn't need API key in the same way
                timeout=int(os.getenv("CLAUDE_TIMEOUT", "60"))
            )
        else:
            raise ValueError(f"Unsupported AI provider: {self.ai_provider}")
    
    def _load_mcp_config(self):
        """Load MCP server configurations"""
        self.mcp_configs = {
            "github": MCPConfig(
                name="github",
                url="https://api.githubcopilot.com/mcp/",
                headers={"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"}
            ),
            "github_extended": MCPConfig(
                name="github_extended",
                command="uv",
                args=["run", "mcps/github_extended.py"],
                env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}
            ),
            "panther": MCPConfig(
                name="panther",
                command="docker",
                args=[
                    "run", "-i", "-e", "PANTHER_INSTANCE_URL", "-e", "PANTHER_API_TOKEN",
                    "--rm", "ghcr.io/panther-labs/mcp-panther"
                ],
                env={
                    "PANTHER_INSTANCE_URL": os.getenv("PANTHER_INSTANCE_URL"),
                    "PANTHER_API_TOKEN": os.getenv("PANTHER_API_TOKEN"),
                }
            ),
            "slack": MCPConfig(
                name="slack",
                command="docker",
                args=[
                    "run", "-i", "--rm", "-e", "SLACK_BOT_TOKEN", "-e", "SLACK_TEAM_ID",
                    "-e", "SLACK_CHANNEL_IDS", "mcp/slack"
                ],
                env={
                    "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
                    "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID"),
                    "SLACK_CHANNEL_IDS": os.getenv("SLACK_CHANNEL_IDS"),
                }
            ),
            "linear": MCPConfig(
                name="linear",
                command="npx",
                args=["-y", "mcp-remote", "https://mcp.linear.app/sse"]
            ),
            "virustotal": MCPConfig(
                name="virustotal",
                command="npx",
                args=["@burtthecoder/mcp-virustotal"],
                env={"VIRUSTOTAL_API_KEY": os.getenv("VIRUSTOTAL_API_KEY")}
            ),
        }
    
    def get_mcp_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get MCP server configuration in the format expected by the agent"""
        if server_name not in self.mcp_configs:
            return None
        
        config = self.mcp_configs[server_name]
        
        if config.url:
            return {
                "type": "http",
                "url": config.url,
                "headers": config.headers or {}
            }
        else:
            return {
                "command": config.command,
                "args": config.args or [],
                "env": config.env or {}
            }
    
    def get_mcp_servers_config(self, server_names: list[str]) -> Dict[str, Dict]:
        """Get MCP servers configuration for multiple servers"""
        configs = {}
        for server_name in server_names:
            config = self.get_mcp_config(server_name)
            if config:
                configs[server_name] = config
        return configs
    
    def validate_config(self) -> list[str]:
        """Validate configuration and return list of missing required variables"""
        errors = []
        
        # Validate AI provider configuration
        if self.ai_provider == "openai" and not self.ai_config.api_key:
            errors.append("OPENAI_API_KEY is required for OpenAI provider")
        
        # Validate MCP server configurations
        for server_name, config in self.mcp_configs.items():
            if config.env:
                for env_var, value in config.env.items():
                    if not value:
                        errors.append(f"{env_var} is required for {server_name} MCP server")
        
        return errors


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def validate_environment() -> bool:
    """Validate that all required environment variables are set"""
    errors = config.validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    return True 