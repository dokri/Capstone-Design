from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


# ── 좌석 상태 Enum ─────────────────────────────────────────────────────────

class SeatStatusEnum(str, Enum):
    using = "using"
    temp = "temp"
    vacant = "vacant"
    auto = "auto"


# ── YOLO 서버 → 메인 서버 ──────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """단일 탐지 결과의 bounding box"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None


class DetectionPayload(BaseModel):
    """YOLO 서버가 전송하는 탐지 결과"""
    camera_id: str
    timestamp: datetime
    detections: List[BoundingBox]
    clear_all: bool = False


# ── 메인 서버 → 웹 ─────────────────────────────────────────────────────────

class SeatStatus(BaseModel):
    """단일 좌석 상태 (목록 및 상세 공통)"""
    id: int
    name: str
    status: SeatStatusEnum

    # 추가: 관리자페이지 새로고침 후 X/Y 좌표 유지용
    coord_x: Optional[float] = None
    coord_y: Optional[float] = None

    last_updated: Optional[datetime]
    vacant_since: Optional[datetime]
    vacant_seconds: Optional[float]
    is_booked: bool

    class Config:
        from_attributes = True


class SeatStatusResponse(BaseModel):
    """전체 좌석 상태 응답"""
    camera_id: str
    seats: List[SeatStatus]
    fetched_at: datetime


# ── 웹 → 메인 서버 (예약) ─────────────────────────────────────────────────

class ReservationRequest(BaseModel):
    """웹에서 보내는 좌석 예약 요청"""
    seat_id: int


class ReservationResponse(BaseModel):
    """예약 결과 응답"""
    success: bool
    message: str
    seat_id: int
    status: SeatStatusEnum


# ── 카메라 / 좌석 등록 (관리용 API) ───────────────────────────────────────

class PointPair(BaseModel):
    src: List[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="카메라 이미지 좌표 [x, y]",
    )
    dst: List[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="실제 좌석 좌표계 [x, y]",
    )


class HomographyUpdate(BaseModel):
    """관리자가 전송하는 Homography 계산용 대응점 4쌍"""
    points: List[PointPair] = Field(..., min_length=4, max_length=4)


class CameraCreate(BaseModel):
    id: str
    name: str


class CameraResponse(BaseModel):
    id: str
    name: str
    homography_matrix: Optional[List[float]]

    class Config:
        from_attributes = True


class SeatCreate(BaseModel):
    name: str
    camera_id: str
    coord_x: float
    coord_y: float


class SeatResponse(BaseModel):
    id: int
    name: str
    camera_id: str
    coord_x: float
    coord_y: float
    is_occupied: bool

    class Config:
        from_attributes = True