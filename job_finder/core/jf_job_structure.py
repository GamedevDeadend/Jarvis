from pydantic import BaseModel, Field
from typing import Literal

class Job(BaseModel):
    id: str = Field(..., description="Unique job ID")
    source: str = Field(..., description="Source of the job listing")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    url: str = Field(..., description="URL to the job listing")
    salary_text: str = Field("not stated", description="Salary information")
    is_salary_estimated: int = Field(0, description="Is salary estimated")
    exact_salary: int | None = Field(None, description="Exact salary in INR per month")
    location: str = Field("", description="Job location related text")
    priority: Literal["primary", "secondary", "rejected"] = Field("secondary", description="Discord Job Channel priority")
    description: str = Field("", description="Job description")
    fit_score: int | None = Field(None, description="Fit score for the job")
    decision : Literal["accepted", "rejected"] = Field("accepted", description="Decision on the job")
    reason: str | None = Field(None, description="Why it was accepted or rejected")