from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ── YOLO 서버 → 메인 서버 ──────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """단일 탐지 결과의 bounding box"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None


class DetectionPayload(BaseModel):
    """YOLO 서버가 전송하는 탐지 결과
    
    detections가 빈 리스트이면 "이 카메라 화면에 아무도 없음"을 의미.
    → timeout 방식과 별개로, 즉시 전체 좌석 해제를 트리거할 수 있음.
    """
    camera_id: str
    timestamp: datetime
    detections: List[BoundingBox]
    # YOLO가 명시적으로 "아무도 없음"을 보낼 때 True로 설정
    # True면 detections 내용과 무관하게 해당 카메라 전체 좌석 즉시 해제
    clear_all: bool = False


# ── 메인 서버 → 웹 ─────────────────────────────────────────────────────────

class SeatStatus(BaseModel):
    """단일 좌석 상태"""
    id: int
    name: str
    is_occupied: bool
    last_updated: Optional[datetime]

    class Config:
        from_attributes = True


class SeatStatusResponse(BaseModel):
    """전체 좌석 상태 응답"""
    camera_id: str
    seats: List[SeatStatus]
    fetched_at: datetime


# ── 카메라 / 좌석 등록 (관리용 API) ───────────────────────────────────────

class CameraCreate(BaseModel):
    id: str
    name: str
    homography_matrix: Optional[List[float]] = Field(
        None, description="3x3 행렬을 행 우선으로 flatten한 9개 값"
    )


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
