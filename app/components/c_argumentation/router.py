from fastapi import APIRouter, File, HTTPException, UploadFile
from .schemas import CaseInput, CaseResponse
from .schemas_document import DocumentExtractionResponse
from .schemas_multi_agent import MultiAgentAssessment
from .pipeline.document_extractor import DocumentUploadError, extract_uploaded_document
from .service import assess_case, submit_case

router = APIRouter(tags=["C4 – Multi-Agent Argumentation Engine"])


@router.get("/ping")
async def ping():
    return {"component": "c_argumentation", "status": "ok"}

@router.post("/cases", response_model=CaseResponse)
def create_case(case: CaseInput):
    return submit_case(case)


@router.post("/assess", response_model=MultiAgentAssessment)
async def assess_case_route(case: CaseInput):
    """C4-only live endpoint for the local RAG and multi-agent assessment."""

    return await assess_case(case)


@router.post("/documents/extract", response_model=DocumentExtractionResponse)
async def extract_document_route(document: UploadFile = File(...)):
    """Extract temporary user document text without persistent storage or legal analysis."""

    try:
        payload = await document.read(10 * 1024 * 1024 + 1)
        result = extract_uploaded_document(document.filename, payload)
    except DocumentUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail="The document could not be extracted safely.") from exc
    response_data = {
        **result.__dict__,
        "sections": [section.__dict__ for section in result.sections],
    }
    return DocumentExtractionResponse.model_validate(response_data)
