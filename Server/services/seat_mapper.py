import math
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Seat, Camera
from schemas import BoundingBox, SeatStatusEnum
from services.homography import compute_center, apply_homography

# ── 설정 값 ────────────────────────────────────────────────────────────
REQUIRED_OCCUPANCY_SECONDS = 5  
OCCUPANCY_TIMEOUT_SECONDS = 5   
AUTO_RELEASE_SECONDS = 100          
DISTANCE_THRESHOLD = 500  
# ──────────────────────────────────────────────────────────────────────

# [추가] 중복 요청 제어용 변수
last_process_time = {}
DEBOUNCE_INTERVAL = 0.8  # 0.8초 이내의 연속 요청은 무시

def find_nearest_seat(seats: List[Seat], mapped_x: float, mapped_y: float) -> Optional[Seat]:
    if not seats:
        return None

    min_dist = float('inf')
    nearest_seat = None

    for s in seats:
        dist = math.hypot(s.coord_x - mapped_x, s.coord_y - mapped_y)
        if dist < min_dist:
            min_dist = dist
            nearest_seat = s
    
    # print 제거함 (속도 향상)
    if min_dist > DISTANCE_THRESHOLD:
        return None

    return nearest_seat

async def process_detections(
    db: AsyncSession,
    camera_id: str,
    timestamp: datetime,
    detections: List[BoundingBox],
    clear_all: bool = False,
) -> List[Seat]:
    global last_process_time
    now = (timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    
    # [수정] 중복 요청 컷트 (DB 세션 열기 전에 실행)
    last_time = last_process_time.get(camera_id)
    if not clear_all and last_time and (now - last_time).total_seconds() < DEBOUNCE_INTERVAL:
        return [] 

    last_process_time[camera_id] = now

    try:
        # 1. 정보 조회
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        camera: Optional[Camera] = result.scalar_one_or_none()
        if camera is None:
            return []

        result = await db.execute(select(Seat).where(Seat.camera_id == camera_id))
        seats: List[Seat] = result.scalars().all()

        if clear_all:
            for seat in seats:
                seat.status            = SeatStatusEnum.vacant
                seat.vacant_since      = None
                seat.last_detected_at  = None
                seat.first_detected_at = None
            await db.commit()
            return seats

        if not camera.homography_matrix:
            return []

        # 2. 감지 매칭 (Step 1)
        occupied_seat_ids: set = set()
        for bbox in detections:
            try:
                cx, cy = compute_center(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                mx, my = apply_homography(camera.homography_matrix, cx, cy)
                nearest = find_nearest_seat(seats, mx, my)
                if nearest:
                    occupied_seat_ids.add(nearest.id)
            except ValueError:
                continue

        # 3. 상태 업데이트 (Step 2)
        for seat in seats:
            is_booked = getattr(seat, "is_booked", False)

            if seat.id in occupied_seat_ids:
                # [사람이 감지된 경우]
                if not is_booked:
                    seat.last_detected_at = now
                    continue

                if seat.first_detected_at is None:
                    seat.first_detected_at = now

                first_dt = seat.first_detected_at
                if first_dt.tzinfo is None:
                    first_dt = first_dt.replace(tzinfo=timezone.utc)

                if (now - first_dt).total_seconds() >= REQUIRED_OCCUPANCY_SECONDS:
                    seat.status       = SeatStatusEnum.using
                    seat.vacant_since = None # 사용 중이면 비어있는 시간 초기화

                seat.last_detected_at = now

            else:
                # [사람이 감지되지 않은 경우]
                seat.first_detected_at = None
                
                if not is_booked:
                    seat.status       = SeatStatusEnum.vacant
                    seat.vacant_since = None
                    continue

                # 핵심: 상태 변화에 따른 vacant_since 관리
                if seat.status == SeatStatusEnum.using:
                    # 방금 막 자리를 비운 경우 -> 시간 기록 시작
                    seat.status = SeatStatusEnum.temp
                    seat.vacant_since = now 
                
                else:
                    # [사람이 감지되지 않은 경우]
                    seat.first_detected_at = None

                    if not is_booked:
                        seat.status       = SeatStatusEnum.vacant
                        seat.vacant_since = None
                        continue

                    # 예약된 좌석: using 또는 vacant → temp로 전환
                    if seat.status in (SeatStatusEnum.using, SeatStatusEnum.vacant):
                        if seat.vacant_since is None:   # 이미 기록된 시간은 건드리지 않음
                            seat.vacant_since = now
                            seat.status = SeatStatusEnum.temp


                
        # 변경된 모든 상태(temp 유지 포함)를 DB에 반영
        await db.commit()
        return seats
    
    except Exception:
        await db.rollback() # 에러 시 롤백해서 DB 연결 꼬임 방지
        return []


async def release_timed_out_seats(db: AsyncSession) -> int:
    try:
        now = datetime.now(timezone.utc)  # ← 올바른 now 정의
        result = await db.execute(
            select(Seat).where(
                Seat.is_booked == True,
                Seat.status == SeatStatusEnum.temp,
            )
        )
        seats: List[Seat] = result.scalars().all()

        released = 0
        for seat in seats:
            vacant_since = seat.vacant_since
            if vacant_since is None:
                seat.vacant_since = now
                continue

            if vacant_since.tzinfo is None:
                vacant_since = vacant_since.replace(tzinfo=timezone.utc)

            if (now - vacant_since).total_seconds() >= AUTO_RELEASE_SECONDS:
                seat.status = SeatStatusEnum.auto
                released += 1

        if released:
            await db.commit()
        return released
    except Exception:
        await db.rollback()
        return 0