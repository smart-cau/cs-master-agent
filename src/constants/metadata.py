from typing import Literal
import os


API_VERSION = os.getenv("API_VERSION", "0.0.1")

ApplyDocType = Literal["candidate_profile", "career_experience", "project_experience"]

Position = Literal["FE", "BE", "FS", "DEV_OPS", "DATA_SCIENTIST", "DATA_ENGINEER" ,"AI_ENGINEER", "OTHER"]

