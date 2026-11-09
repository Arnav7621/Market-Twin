
from fastapi import APIRouter, HTTPException

from pydantic import BaseModel


router = APIRouter(
    prefix="/api/queue-mind",
    tags=["QueueMind"]
)


class FillRequest(BaseModel):

    ticker: str = "AAPL"

    order_side: str = "buy"

    order_quantity: int = 10

    seed: int = 42


@router.post("/predict")
def predict_fill(
    request: FillRequest
):

    # Step 1: Load market features.
    # Step 2: Prepare QueueMind input features.
    # Step 3: Call the actual QueueMind prediction
    #         function.
    # Step 4: Return the fill probability.

    try:

        # Replace this section with the actual
        # QueueMind function discovered in Task 1.

        raise NotImplementedError(
            "Connect the actual QueueMind function"
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
