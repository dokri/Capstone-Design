"""
crud.py — 비동기 DB 접근 로직 전담
모든 함수는 async def이며, AsyncSession을 받아 await로 쿼리합니다.
"""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Camera, Seat
from schemas import CameraCreate, SeatCreate


# ══════════════════════════════════════════════════════════════════════════════
# Camera
# ══════════════════════════════════════════════════════════════════════════════

async def get_camera(db: AsyncSession, camera_id: str) -> Optional[Camera]:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    return result.scalar_one_or_none()


async def get_all_cameras(db: AsyncSession) -> List[Camera]:
    result = await db.execute(select(Camera))
    return result.scalars().all()


async def create_camera(db: AsyncSession, payload: CameraCreate) -> Camera:
    camera = Camera(
        id=payload.id,
        name=payload.name,
        homography_matrix=payload.homography_matrix,
    )
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    return camera


async def update_homography(db: AsyncSession, camera_id: str, matrix: List[float]) -> Optional[Camera]:
    camera = await get_camera(db, camera_id)
    if not camera:
        return None
    camera.homography_matrix = matrix
    await db.commit()
    await db.refresh(camera)
    return camera


# ══════════════════════════════════════════════════════════════════════════════
# Seat
# ══════════════════════════════════════════════════════════════════════════════

async def get_seats_by_camera(db: AsyncSession, camera_id: str) -> List[Seat]:
    result = await db.execute(select(Seat).where(Seat.camera_id == camera_id))
    return result.scalars().all()


async def get_seat(db: AsyncSession, seat_id: int) -> Optional[Seat]:
    result = await db.execute(select(Seat).where(Seat.id == seat_id))
    return result.scalar_one_or_none()


async def create_seat(db: AsyncSession, payload: SeatCreate) -> Seat:
    seat = Seat(
        name=payload.name,
        camera_id=payload.camera_id,
        coord_x=payload.coord_x,
        coord_y=payload.coord_y,
    )
    db.add(seat)
    await db.commit()
    await db.refresh(seat)
    return seat


async def delete_seat(db: AsyncSession, seat_id: int) -> bool:
    seat = await get_seat(db, seat_id)
    if not seat:
        return False
    await db.delete(seat)
    await db.commit()
    return True


async def get_vacant_duration(db: AsyncSession, seat_id: int) -> Optional[Seat]:
    """자리비움 시간 계산용 — 좌석 단건 조회 (vacant_since 포함)"""
    return await get_seat(db, seat_id)


async def reserve_seat(db: AsyncSession, seat_id: int) -> tuple[bool, str, Optional[Seat]]:
    """
    웹 예약 요청 처리.
    반환: (success, message, seat)
      - 좌석 없음   → (False, "not_found", None)
      - 이미 점유   → (False, "already_occupied", seat)
      - 예약 성공   → (True,  "using", seat)
    """
    seat = await get_seat(db, seat_id)
    if not seat:
        return False, "not_found", None
    if seat.is_occupied:
        return False, "already_occupied", seat

    from datetime import datetime, timezone
    from schemas import SeatStatusEnum
    seat.is_occupied = True
    seat.status = SeatStatusEnum.using
    seat.last_detected_at = datetime.now(timezone.utc)
    seat.vacant_since = None
    await db.commit()
    await db.refresh(seat)
    return True, "using", seat