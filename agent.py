from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncGenerator
import os
from dotenv import load_dotenv

# If .env exists, load it
if os.path.exists(".env"):
    load_dotenv()

# Typical MCP config
# https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-mcp-understanding-config.html
mcp_config = {
    "mcpServers": {
        "panther": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "-e",
                "PANTHER_INSTANCE_URL",
                "-e",
                "PANTHER_API_TOKEN",
                "--rm",
                "ghcr.io/panther-labs/mcp-panther",
            ],
            "env": {
                "PANTHER_INSTANCE_URL": os.getenv("PANTHER_INSTANCE_URL"),
                "PANTHER_API_TOKEN": os.getenv("PANTHER_API_TOKEN"),
            },
        },
        "github": {
            "type": "http",
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
        },
        "github_extended": {
            "command": "uv",
            "args": ["run", "mcps/github_extended.py"],
            "env": {"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")},
        },
        "slack": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e",
                "SLACK_BOT_TOKEN",
                "-e",
                "SLACK_TEAM_ID",
                "-e",
                "SLACK_CHANNEL_IDS",
                "mcp/slack",
            ],
            "env": {
                "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
                "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID"),
                "SLACK_CHANNEL_IDS": os.getenv("SLACK_CHANNEL_IDS"),
            },
        },
        "linear": {
            "command": "npx",
            "args": ["-y", "mcp-remote", "https://mcp.linear.app/sse"],
        },
        "virustotal": {
            "command": "npx",
            "args": ["@burtthecoder/mcp-virustotal"],
            "env": {
                "VIRUSTOTAL_API_KEY": os.getenv("VIRUSTOTAL_API_KEY"),
            },
        },
    }
}


class Message:
    """Generic message class for AI provider responses"""
    def __init__(self, content: str, role: str = "assistant", **kwargs):
        self.content = content
        self.role = role
        self.metadata = kwargs

    def __str__(self):
        return f"{self.role}: {self.content}"


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    async def query(
        self, 
        prompt: str, 
        system_prompt: str,
        mcp_servers: Dict[str, Dict],
        allowed_tools: Optional[List[str]] = None
    ) -> AsyncGenerator[Message, None]:
        """Query the AI provider with the given prompt and configuration"""
        pass


class ClaudeCodeProvider(AIProvider):
    """Claude Code AI provider implementation"""
    
    def __init__(self):
        try:
            from claude_code_sdk import query as claude_query, ClaudeCodeOptions
            self.claude_query = claude_query
            self.ClaudeCodeOptions = ClaudeCodeOptions
        except ImportError:
            raise ImportError("claude-code-sdk is required for ClaudeCodeProvider. Install with: pip install claude-code-sdk")

    async def query(
        self, 
        prompt: str, 
        system_prompt: str,
        mcp_servers: Dict[str, Dict],
        allowed_tools: Optional[List[str]] = None
    ) -> AsyncGenerator[Message, None]:
        options = self.ClaudeCodeOptions(
            permission_mode="bypassPermissions",
            mcp_servers=mcp_servers,
            mcp_tools=allowed_tools,
            system_prompt=system_prompt,
        )
        
        async for message in self.claude_query(prompt=prompt, options=options):
            # Convert Claude Code message to our generic Message format
            yield Message(
                content=str(message),
                role="assistant",
                original_message=message
            )


class OpenAIProvider(AIProvider):
    """OpenAI AI provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
        except ImportError:
            raise ImportError("openai is required for OpenAIProvider. Install with: pip install openai")
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    async def query(
        self, 
        prompt: str, 
        system_prompt: str,
        mcp_servers: Dict[str, Dict],
        allowed_tools: Optional[List[str]] = None
    ) -> AsyncGenerator[Message, None]:
        # Note: OpenAI doesn't support MCP servers directly like Claude Code
        # This is a simplified implementation that ignores MCP servers
        # For full MCP support with OpenAI, you'd need additional tool calling logic
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield Message(
                        content=chunk.choices[0].delta.content,
                        role="assistant"
                    )
        except Exception as e:
            yield Message(
                content=f"Error querying OpenAI: {str(e)}",
                role="assistant",
                error=True
            )


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        prompt: str,
        mcp_servers: List[str],
        allowed_tools: Optional[List[str]] = None,
        ai_provider: Optional[AIProvider] = None,
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.mcp_servers: Dict[str, Dict] = {
            server: mcp_config["mcpServers"][server]
            for server in mcp_servers
            if server in mcp_config["mcpServers"]
        }
        self.allowed_tools: Optional[List[str]] = allowed_tools
        
        # Use provided AI provider or default to Claude Code
        self.ai_provider = ai_provider or ClaudeCodeProvider()

    async def run(self) -> List[Message]:
        messages = []
        async for message in self.ai_provider.query(
            prompt=self.prompt,
            system_prompt=self.system_prompt,
            mcp_servers=self.mcp_servers,
            allowed_tools=self.allowed_tools
        ):
            print(message)
            messages.append(message)
        return messages


# Convenience functions for creating agents with different providers
def create_claude_agent(
    name: str,
    description: str,
    system_prompt: str,
    prompt: str,
    mcp_servers: List[str],
    allowed_tools: Optional[List[str]] = None,
) -> Agent:
    """Create an agent using Claude Code as the AI provider"""
    return Agent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        prompt=prompt,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        ai_provider=ClaudeCodeProvider()
    )


def create_openai_agent(
    name: str,
    description: str,
    system_prompt: str,
    prompt: str,
    mcp_servers: List[str],
    allowed_tools: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o"
) -> Agent:
    """Create an agent using OpenAI as the AI provider"""
    return Agent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        prompt=prompt,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        ai_provider=OpenAIProvider(api_key=api_key, model=model)
    )
