"""
crud.py — 비동기 DB 접근 로직 전담
수정 사항: 
1. create_camera에서 payload에 없는 속성 참조 제거
2. DB 작업 시 try-except 및 rollback 로직 추가 (안정성 강화)
"""
from typing import List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from schemas import SeatStatusEnum


from models import Camera, Seat
import schemas # schemas.CameraCreate 등을 사용하기 위해 전체 임포트 권장


# ══════════════════════════════════════════════════════════════════════════════
# Camera
# ══════════════════════════════════════════════════════════════════════════════

async def get_camera(db: AsyncSession, camera_id: str) -> Optional[Camera]:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    return result.scalar_one_or_none()


async def get_all_cameras(db: AsyncSession) -> List[Camera]:
    result = await db.execute(select(Camera))
    return result.scalars().all()


async def create_camera(db: AsyncSession, payload: schemas.CameraCreate) -> Camera:
    # ⚠️ 핵심 수정: CameraCreate에는 homography_matrix가 없으므로 
    # id와 name만 사용하여 객체를 생성합니다.
    camera = Camera(
        id=payload.id,
        name=payload.name,
        homography_matrix=None  # 명시적으로 None 설정 (또는 생략 가능)
    )
    
    db.add(camera)
    try:
        await db.commit()
        await db.refresh(camera)
    except Exception as e:
        await db.rollback()
        raise e
    return camera


async def update_homography(db: AsyncSession, camera_id: str, matrix: List[float]) -> Optional[Camera]:
    camera = await get_camera(db, camera_id)
    if not camera:
        return None
    
    camera.homography_matrix = matrix
    try:
        await db.commit()
        await db.refresh(camera)
    except Exception as e:
        await db.rollback()
        raise e
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


async def create_seat(db: AsyncSession, payload: schemas.SeatCreate) -> Seat:
    seat = Seat(
        name=payload.name,
        camera_id=payload.camera_id,
        coord_x=payload.coord_x,
        coord_y=payload.coord_y,
    )
    db.add(seat)
    try:
        await db.commit()
        await db.refresh(seat)
    except Exception as e:
        await db.rollback()
        raise e
    return seat


async def delete_seat(db: AsyncSession, seat_id: int) -> bool:
    seat = await get_seat(db, seat_id)
    if not seat:
        return False
    try:
        await db.delete(seat)
        await db.commit()
        return True
    except Exception:
        await db.rollback()
        return False


async def reserve_seat(db: AsyncSession, seat_id: int):
    # 1. Seat 클래스를 직접 사용합니다 (models.Seat 대신 Seat)
    result = await db.execute(select(Seat).where(Seat.id == seat_id))
    seat = result.scalar_one_or_none()

    if not seat:
        return False, "not_found", None

    from datetime import datetime, timezone
    from schemas import SeatStatusEnum

    # 2. 값 주입
    # models.py에 is_booked가 추가되었으므로 hasattr 없이 직접 접근해도 됩니다.
    seat.is_booked = True 
    seat.status = SeatStatusEnum.vacant
    seat.vacant_since = None
    seat.is_occupied = False 

    try:
        # 3. 변경 사항 저장
        db.add(seat) 
        await db.commit() 
        await db.refresh(seat) 
        return True, "success", seat
    except Exception as e:
        print(f"❌ DB 저장 중 에러 발생: {e}")
        await db.rollback()
        return False, "db_error", None
    


async def get_my_reservations(db):
    result = await db.execute(
        select(Seat).where(
            Seat.status.in_([
                SeatStatusEnum.using,
                SeatStatusEnum.temp
            ])
        )
    )

    return result.scalars().all()


async def return_seat(db, seat_id: int):
    seat = await get_seat(db, seat_id)

    if not seat:
        return None, "not_found"

    seat.status = SeatStatusEnum.vacant
    seat.vacant_since = None
    seat.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(seat)

    return seat, "success"


async def temp_leave_seat(db, seat_id: int):
    seat = await get_seat(db, seat_id)

    if not seat:
        return None, "not_found"

    if seat.status != SeatStatusEnum.using:
        return None, "invalid_status"

    seat.status = SeatStatusEnum.temp
    seat.vacant_since = datetime.now(timezone.utc)
    seat.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(seat)

    return seat, "success"


async def return_from_temp(db, seat_id: int):
    seat = await get_seat(db, seat_id)

    if not seat:
        return None, "not_found"

    if seat.status != SeatStatusEnum.temp:
        return None, "invalid_status"

    seat.status = SeatStatusEnum.using
    seat.vacant_since = None
    seat.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(seat)

    return seat, "success"