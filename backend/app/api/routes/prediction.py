"""Prediction API endpoints."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image

from backend.app.core.config import settings
from backend.app.dependencies import get_prediction_service
from backend.app.schemas.prediction import NumericalFeaturesInput
from backend.app.schemas.response import (ErrorResponse, PredictionResponse, ImagePredictionResponse, NumericalPredictionResponse, FusionPredictionResponse, RiskIndexResponse)
from backend.app.services.prediction_service import PredictionService
from src.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    ModelError,
    ValidationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

async def _process_image(image: UploadFile) -> Path:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded image filename is missing.")
    suffix = Path(image.filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported image extension '{suffix}'.")
    try:
        content = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image size exceeds maximum.")
    try:
        pil_img = Image.open(io.BytesIO(content))
        pil_img.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Not a valid image: {exc}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    return tmp_path




@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Cyclone Risk with Verification & Explanation",
    description=(
        "Processes an uploaded satellite infrared image along with 12 atmospheric numerical variables. "
        "Executes the multimodal ML/DL models (CNN + MLP + Late Fusion), evaluates the 0–100 Development/Risk Index, "
        "triggers the secondary LLM Vision Verification Gate (if all signals >= 50%), and generates a structured "
        "meteorological explanation."
    ),
    responses={
        200: {"model": PredictionResponse, "description": "Successful prediction and explanation result"},
        400: {"model": ErrorResponse, "description": "Bad Request — Invalid, corrupted, or unsupported image"},
        422: {"model": ErrorResponse, "description": "Validation Error — Missing or out-of-range numerical input"},
        500: {"model": ErrorResponse, "description": "Internal Server Error — ML or explanation failure"},
        503: {"model": ErrorResponse, "description": "Service Unavailable — Missing model artifacts or configuration"},
    },
)
async def predict_cyclone_risk(
    image: UploadFile = File(
        ...,
        description="Satellite infrared image file (.jpg, .jpeg, or .png)",
    ),
    case_id: str = Form(
        "custom",
        description="Observation identifier or storm case ID (e.g. 'TC-01', 'TC-02')",
        examples=["TC-01"],
    ),
    numerical: NumericalFeaturesInput = Depends(NumericalFeaturesInput.as_form),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Execute end-to-end multimodal prediction, verification, and explanation."""
    # 1. Validate image upload presence & extension
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image filename is missing.",
        )

    suffix = Path(image.filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image extension '{suffix}'. "
                f"Allowed extensions are: {sorted(settings.allowed_extensions)}"
            ),
        )

    # 2. Read image content and validate size
    try:
        content = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded image: {exc}",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty (0 bytes).",
        )

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Uploaded image size ({len(content)} bytes) exceeds the maximum allowed "
                f"limit of {settings.max_upload_size_bytes} bytes."
            ),
        )

    # 3. Validate image integrity and color mode with PIL
    try:
        pil_img = Image.open(io.BytesIO(content))
        pil_img.verify()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is not a valid or readable image: {exc}",
        )

    try:
        pil_img = Image.open(io.BytesIO(content))
        pil_img.load()
        if pil_img.mode not in {"RGB", "L", "RGBA"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image color mode '{pil_img.mode}'. Supported modes are RGB, L, RGBA.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted image contents: {exc}",
        )

    # 4. Write image to safe temporary file on disk for existing image pipeline
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)

    # 5. Execute inference through existing pipeline with guaranteed cleanup
    try:
        numerical_dict = numerical.to_features_dict()
        result = service.predict(
            image_path=tmp_path,
            numerical_features=numerical_dict,
            case_id=case_id,
        )
        return result

    except DatasetValidationError as exc:
        logger.warning("Dataset validation failed during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Data validation error: {exc}",
        )
    except ValidationError as exc:
        logger.warning("Validation error during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {exc}",
        )
    except ConfigurationError as exc:
        logger.error("Configuration error during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service configuration error: {exc}",
        )
    except ModelError as exc:
        logger.error("Model failure during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference model execution error: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected error during prediction pipeline: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during multimodal prediction and verification.",
        )
    finally:
        # Secure cleanup: remove temporary file
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except OSError as err:
                logger.warning("Failed to remove temporary file %s: %s", tmp_path, err)


@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Isolated Image CNN Inference",
)
async def predict_image(
    image: UploadFile = File(...),
    case_id: str = Form("custom"),
    service: PredictionService = Depends(get_prediction_service),
):
    tmp_path = await _process_image(image)
    try:
        return service.predict_image(image_path=tmp_path, case_id=case_id)
    except Exception as exc:
        logger.error(f"Error in image prediction: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)


@router.post(
    "/predict/numerical",
    response_model=NumericalPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Isolated Numerical MLP Inference",
)
async def predict_numerical(
    case_id: str = Form("custom"),
    numerical: NumericalFeaturesInput = Depends(NumericalFeaturesInput.as_form),
    service: PredictionService = Depends(get_prediction_service),
):
    try:
        numerical_dict = numerical.to_features_dict()
        return service.predict_numerical(numerical_features=numerical_dict, case_id=case_id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Error in numerical prediction: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/predict/fusion",
    response_model=FusionPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Isolated Late-Fusion Inference",
)
async def predict_fusion(
    image: UploadFile = File(...),
    case_id: str = Form("custom"),
    numerical: NumericalFeaturesInput = Depends(NumericalFeaturesInput.as_form),
    service: PredictionService = Depends(get_prediction_service),
):
    tmp_path = await _process_image(image)
    try:
        numerical_dict = numerical.to_features_dict()
        return service.predict_fusion(image_path=tmp_path, numerical_features=numerical_dict, case_id=case_id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Error in fusion prediction: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)


@router.post(
    "/scoring/risk-index",
    response_model=RiskIndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Isolated Risk Index Calculation",
)
async def score_risk_index(
    image: UploadFile = File(...),
    case_id: str = Form("custom"),
    numerical: NumericalFeaturesInput = Depends(NumericalFeaturesInput.as_form),
    service: PredictionService = Depends(get_prediction_service),
):
    tmp_path = await _process_image(image)
    try:
        numerical_dict = numerical.to_features_dict()
        return service.calculate_risk_index(image_path=tmp_path, numerical_features=numerical_dict, case_id=case_id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Error in risk index calculation: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
