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
DISTANCE_THRESHOLD = 500       # 시연을 위해 500으로 넉넉하게 유지
# ──────────────────────────────────────────────────────────────────────

def find_nearest_seat(seats: List[Seat], mapped_x: float, mapped_y: float) -> Optional[Seat]:
    print("\n" + "="*50)
    print(f"📍 [실시간 사람 좌표] X: {mapped_x:.1f}, Y: {mapped_y:.1f}")
    print("="*50)

    if not seats:
        return None
    
    seat_distances = []
    for s in seats:
        dist = math.hypot(s.coord_x - mapped_x, s.coord_y - mapped_y)
        seat_distances.append((s, dist))
        print(f"   ㄴ 좌석 [{s.name}]와의 거리: {dist:.1f} (좌석좌표: {s.coord_x}, {s.coord_y})")

    nearest_seat, min_dist = min(seat_distances, key=lambda x: x[1])

    if min_dist > DISTANCE_THRESHOLD: 
        print(f"❌ 매칭 실패: {min_dist:.1f}px 떨어져 있음.")
        return None
        
    print(f"✅ 매칭 성공! [{nearest_seat.name}] 점유 중")
    return nearest_seat


async def process_detections(
    db: AsyncSession,
    camera_id: str,
    timestamp: datetime,
    detections: List[BoundingBox],
    clear_all: bool = False,
) -> List[Seat]:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera: Optional[Camera] = result.scalar_one_or_none()
    if camera is None: raise ValueError(f"카메라 없음: {camera_id}")

    result = await db.execute(select(Seat).where(Seat.camera_id == camera_id))
    seats: List[Seat] = result.scalars().all()
    now = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)

    if clear_all:
        for seat in seats:
            seat.status = SeatStatusEnum.vacant
            seat.vacant_since = None # 리셋
            seat.last_detected_at = None
        await db.commit()
        return seats

    if not camera.homography_matrix: raise ValueError("행렬 설정 안됨")

    # Step 1: 탐지된 좌석 ID 파악
    occupied_seat_ids = set()
    for bbox in detections:
        try:
            cx, cy = compute_center(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            mx, my = apply_homography(camera.homography_matrix, cx, cy)
            nearest = find_nearest_seat(seats, mx, my)
            if nearest: occupied_seat_ids.add(nearest.id)
        except ValueError: continue

    # Step 2: 좌석 상태 갱신 로직
    for seat in seats:
        last_dt = seat.last_detected_at.replace(tzinfo=timezone.utc) if seat.last_detected_at and seat.last_detected_at.tzinfo is None else seat.last_detected_at

        # [상황 A] 사람이 감지됨
        if seat.id in occupied_seat_ids:
            seat.status = SeatStatusEnum.using
            seat.last_detected_at = now
            seat.vacant_since = None  # 사람이 있으니 공석 시간 측정 안 함
        
        # [상황 B] 사람이 감지되지 않음
        else:
            # 1. 원래 사용 중이었거나 잠깐 자리를 비운 경우
            if seat.status in [SeatStatusEnum.using, SeatStatusEnum.temp]:
                elapsed = (now - last_dt).total_seconds() if last_dt else 9999
                
                if elapsed >= OCCUPANCY_TIMEOUT_SECONDS:
                    seat.status = SeatStatusEnum.vacant
                    # ⭐ 핵심: 예약 중(is_booked)일 때만 비어있는 시간 기록 시작
                    if getattr(seat, 'is_booked', False):
                        seat.vacant_since = now
                    else:
                        seat.vacant_since = None
                else:
                    seat.status = SeatStatusEnum.temp
            
            # 2. 이미 vacant 상태인 경우 (예약 여부에 따른 vacant_since 유지 여부)
            elif seat.status == SeatStatusEnum.vacant:
                if getattr(seat, 'is_booked', False):
                    # 예약 중인데 계속 비어있다면 최초 비기 시작한 시간 유지
                    if seat.vacant_since is None:
                        seat.vacant_since = now
                else:
                    # 예약도 안 되어 있고 사람도 없으면 시간 측정 안 함
                    seat.vacant_since = None

    await db.commit()
    return seats

async def release_timed_out_seats(db: AsyncSession) -> int:
    """
    백그라운드 태스크용: 
    예약 중(is_booked=True)인데 자리를 비운 지 오래된 좌석을 자동으로 해제합니다.
    """
    now = datetime.now(timezone.utc)
    # 사용 중이거나 잠시 비움 상태인 좌석 조회
    result = await db.execute(
        select(Seat).where(Seat.status.in_([SeatStatusEnum.using, SeatStatusEnum.temp]))
    )
    seats: List[Seat] = result.scalars().all()

    released = 0
    for seat in seats:
        last = seat.last_detected_at
        if last is None:
            # 감지 기록이 없으면 안전하게 vacant로 변경
            seat.status = SeatStatusEnum.vacant
            released += 1
            continue
            
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
            
        # 설정된 타임아웃(예: 30초) 이상 자리를 비웠다면
        if (now - last).total_seconds() >= OCCUPANCY_TIMEOUT_SECONDS:
            seat.is_occupied = False
            seat.status = SeatStatusEnum.vacant
            seat.vacant_since = now
            # seat.is_booked = False  # 필요시 예약까지 자동으로 취소하려면 이 주석을 푸세요.
            released += 1

    if released:
        await db.commit()
    return released