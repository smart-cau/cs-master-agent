from pydantic import BaseModel, Field
from typing import Optional


class IsResumeResult(BaseModel):
    is_resume: bool = Field(description="Whether the document is a resume.")
    reason: str = Field(description="Reason for the decision. Response in Korean. Reply less than 300 characters.", max_length=350)