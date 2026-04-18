import math
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Seat, Camera
from schemas import BoundingBox, SeatStatusEnum
from services.homography import compute_center, apply_homography


OCCUPANCY_TIMEOUT_SECONDS = 30


def find_nearest_seat(seats: List[Seat], mapped_x: float, mapped_y: float) -> Optional[Seat]:
    if not seats:
        return None
    return min(seats, key=lambda s: math.hypot(s.coord_x - mapped_x, s.coord_y - mapped_y))


async def process_detections(
    db: AsyncSession,
    camera_id: str,
    timestamp: datetime,
    detections: List[BoundingBox],
    clear_all: bool = False,
) -> List[Seat]:
    """
    YOLO 탐지 결과를 받아 좌석 상태를 갱신합니다.

    상태 전환:
      감지됨                → using
      감지 안 됨 (timeout 전) → temp  (일시비움)
      timeout 완료          → vacant
      clear_all=True        → vacant (즉시)
    """
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera: Optional[Camera] = result.scalar_one_or_none()
    if camera is None:
        raise ValueError(f"카메라를 찾을 수 없습니다: {camera_id}")

    result = await db.execute(select(Seat).where(Seat.camera_id == camera_id))
    seats: List[Seat] = result.scalars().all()

    now = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)

    # ── [명시적 해제] clear_all → 전체 vacant ─────────────────────────────
    if clear_all:
        for seat in seats:
            if seat.status != SeatStatusEnum.vacant:
                seat.is_occupied = False
                seat.status = SeatStatusEnum.vacant
                seat.vacant_since = now
        await db.commit()
        return seats

    if not camera.homography_matrix:
        raise ValueError(f"카메라 {camera_id}의 Homography 행렬이 설정되지 않았습니다.")

    # ── Step 1: BoundingBox → 중심점 계산 ────────────────────────────────
    occupied_seat_ids = set()
    for bbox in detections:
        try:
            cx, cy = compute_center(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            mapped_x, mapped_y = apply_homography(camera.homography_matrix, cx, cy)
            nearest = find_nearest_seat(seats, mapped_x, mapped_y)
            if nearest is not None:
                occupied_seat_ids.add(nearest.id)
        except ValueError:
            continue

    # ── Step 2: 좌석 상태 갱신 ───────────────────────────────────────────
    for seat in seats:
        if seat.id in occupied_seat_ids:
            seat.is_occupied = True
            seat.status = SeatStatusEnum.using
            seat.last_detected_at = now
            seat.vacant_since = None

        elif seat.status == SeatStatusEnum.using and seat.last_detected_at is not None:
            last = seat.last_detected_at
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            elapsed = (now - last).total_seconds()

            if elapsed >= OCCUPANCY_TIMEOUT_SECONDS:
                seat.is_occupied = False
                seat.status = SeatStatusEnum.vacant
                seat.vacant_since = now
            else:
                seat.status = SeatStatusEnum.temp

    await db.commit()
    return seats


async def release_timed_out_seats(db: AsyncSession) -> int:
    """백그라운드 태스크용: using/temp 상태에서 timeout된 좌석을 vacant으로 전환"""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(Seat).where(Seat.status.in_([SeatStatusEnum.using, SeatStatusEnum.temp]))
    )
    seats: List[Seat] = result.scalars().all()

    released = 0
    for seat in seats:
        if seat.last_detected_at is None:
            seat.is_occupied = False
            seat.status = SeatStatusEnum.vacant
            seat.vacant_since = now
            released += 1
            continue
        last = seat.last_detected_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if (now - last).total_seconds() >= OCCUPANCY_TIMEOUT_SECONDS:
            seat.is_occupied = False
            seat.status = SeatStatusEnum.vacant
            seat.vacant_since = now
            released += 1

    if released:
        await db.commit()
    return released