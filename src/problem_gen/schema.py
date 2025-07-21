from pydantic import BaseModel, Field
from typing import Annotated, operator

class Problem_Content(BaseModel):
    question: str = Field(description="The question to be asked. Reply less than 500 characters.", max_length=500)
    explanation: str = Field(description="A brief explanation of why this question is relevant based on the resume. Reply less than 500 characters.", max_length=500)

class Problem_Contents(BaseModel):
    contents: Annotated[list[Problem_Content], operator.add] = Field(description="The problems to be solved.")
    
    
    

