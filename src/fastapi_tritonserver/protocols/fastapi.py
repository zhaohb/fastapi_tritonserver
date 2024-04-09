from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    prompts: Optional[List[str]] = None
    image: Optional[str] = None
    images: Optional[List[str]] = None
    timeout: Optional[float] = 6000
    only_return_output: Optional[bool] = False
    uuid: Optional[str] = None
    stream: Optional[bool] = False

