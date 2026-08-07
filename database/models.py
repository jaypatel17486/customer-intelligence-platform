from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, Float, String


class Base(DeclarativeBase):
    pass


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)

    prediction = Column(String)

    probability = Column(Float)