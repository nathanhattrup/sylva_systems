from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .database import Base

class Tree(Base):
    __tablename__ = "trees"

    id = Column(Integer, primary_key=True, index=True)  # auto-generated numeric ID
    tree_id = Column(String, unique=True, index=True)  # e.g., "A-014"
    zone = Column(String, index=True)
    severity = Column(String)
    risk = Column(String)
    confidence = Column(Float)
    last_seen = Column(DateTime, default=func.now())
    notes = Column(String)
    x = Column(Float)
    y = Column(Float)
    trend = Column(String)
