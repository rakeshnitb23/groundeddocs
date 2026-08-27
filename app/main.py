import uuid

from fastapi import FastAPI
from sqlalchemy.orm import Session

from app.api.documents import TEMP_USER_ID, router as documents_router
from app.core.database import SessionLocal
from app.models.user import User

app = FastAPI(title="GroundedDocs")
app.include_router(documents_router)


@app.on_event("startup")
def on_startup():
    db: Session = SessionLocal()
    try:
        user = db.get(User, TEMP_USER_ID)
        if not user:
            db.add(User(id=TEMP_USER_ID))
            db.commit()
    finally:
        db.close()
