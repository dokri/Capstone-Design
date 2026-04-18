from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class Camera(Base):
    """카메라 정보 및 Homography 행렬 저장"""
    __tablename__ = "cameras"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    homography_matrix = Column(JSON, nullable=True)  # [h00,h01,...,h22] 9개 값
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    seats = relationship("Seat", back_populates="camera", lazy="selectin")


class Seat(Base):
    """좌석 정보 - 좌석 좌표계 기준 위치"""
    __tablename__ = "seats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)

    coord_x = Column(Float, nullable=False)
    coord_y = Column(Float, nullable=False)

    is_occupied = Column(Boolean, default=False, nullable=False)
    status = Column(String, default="vacant", nullable=False)  # using | temp | vacant
    last_detected_at = Column(DateTime(timezone=True), nullable=True)
    vacant_since = Column(DateTime(timezone=True), nullable=True)  # 비어있기 시작한 시각
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    camera = relationship("Camera", back_populates="seats", lazy="selectin")