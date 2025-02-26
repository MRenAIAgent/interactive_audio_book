import os
import json
import aiohttp
from typing import Optional, Type

from pydantic.v1 import BaseModel, Field

from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionConfig, ActionInput, ActionOutput


class SerperSearchVocodeActionConfig(ActionConfig, type="serper_search"):  # type: ignore
    """Configuration for the SerperSearch action"""
    api_key: Optional[str] = None


class SerperSearchParameters(BaseModel):
    query: str = Field(..., description="The search query to execute")
    num_results: int = Field(3, description="Number of search results to return")


class SerperSearchResponse(BaseModel):
    success: bool
    results: list = []
    error_message: str = ""


class SerperSearchAction(
    BaseAction[
        SerperSearchVocodeActionConfig,
        SerperSearchParameters,
        SerperSearchResponse,
    ]
):
    description: str = """Searches the web using Serper API.
    Input is the search query, and output is a list of search results.
    """
    parameters_type: Type[SerperSearchParameters] = SerperSearchParameters
    response_type: Type[SerperSearchResponse] = SerperSearchResponse

    def __init__(
        self,
        action_config: SerperSearchVocodeActionConfig,
    ):
        super().__init__(
            action_config,
            quiet=False,
            should_respond="always",
            is_interruptible=True,
        )
        self.api_key = action_config.params.get("api_key", os.environ.get("SERPER_API_KEY", ""))
        if not self.api_key:
            raise ValueError("Serper API key is required")

    async def run(
        self, action_input: ActionInput[SerperSearchParameters]
    ) -> ActionOutput[SerperSearchResponse]:
        query = action_input.params.query
        num_results = action_input.params.num_results

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "q": query,
                    "num": num_results
                }
                
                async with session.post(
                    "https://google.serper.dev/search",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract organic search results
                        results = []
                        if "organic" in data:
                            for item in data["organic"][:num_results]:
                                result = {
                                    "title": item.get("title", ""),
                                    "link": item.get("link", ""),
                                    "snippet": item.get("snippet", "")
                                }
                                results.append(result)
                        
                        return ActionOutput(
                            action_type=action_input.action_config.type,
                            response=SerperSearchResponse(
                                success=True,
                                results=results
                            ),
                        )
                    else:
                        error_text = await response.text()
                        return ActionOutput(
                            action_type=action_input.action_config.type,
                            response=SerperSearchResponse(
                                success=False,
                                error_message=f"API error: {response.status} - {error_text}"
                            ),
                        )
        except Exception as e:
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=SerperSearchResponse(
                    success=False,
                    error_message=f"Error performing search: {str(e)}"
                ),
            )
