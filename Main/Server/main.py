import asyncio
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

import models
import crud
import schemas
from database import engine, get_db, AsyncSessionLocal
from models import Seat, Camera  # <--- select에 필요한 모델 명시
from schemas import (
    DetectionPayload,
    SeatStatus,
    SeatStatusEnum,
    SeatStatusResponse,
    ReservationRequest,
    ReservationResponse,
    CameraCreate,
    CameraResponse,
    HomographyUpdate,
    SeatCreate,
    SeatResponse,
    SensorPayload,
)
from services.seat_mapper import process_detections, release_timed_out_seats

# ── 헬퍼 함수 ──────────────────────────────────────────────────────────────────

def _calc_vacant_seconds(seat, now: datetime):
    # 계산할 상태가 아니면 무조건 0
    if not seat.vacant_since or seat.status not in [SeatStatusEnum.temp, SeatStatusEnum.auto]:
        return 0.0

    vs_dt = seat.vacant_since
    # DB 시각 처리 (Timezone-aware 강제 변환)
    if vs_dt.tzinfo is None:
        vs_dt = vs_dt.replace(tzinfo=timezone.utc)
    
    # 현재 시각 처리
    now_dt = now
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    # 타임스탬프 숫자 비교 (가장 확실하게 증가함)
    diff = now_dt.timestamp() - vs_dt.timestamp()
    return max(0.0, round(diff, 1))

def _to_seat_status(seat, now: datetime) -> SeatStatus:
    return SeatStatus(
        id=seat.id,
        name=seat.name,
        status=seat.status,
        is_booked=getattr(seat, 'is_booked', False),
         # 추가: 프론트 관리자페이지에서 좌석 X/Y 좌표 유지용
        coord_x=seat.coord_x,
        coord_y=seat.coord_y,
        
        last_updated=seat.updated_at,
        vacant_since=seat.vacant_since,
        vacant_seconds=_calc_vacant_seconds(seat, now),
    )

# ── 백그라운드 태스크 ──────────────────────────────────────────────────────────

async def periodic_seat_release(interval: int = 10):
    while True:
        await asyncio.sleep(interval)
        async with AsyncSessionLocal() as db:
            released = await release_timed_out_seats(db)
            if released:
                print(f"[auto-release] {released}개 좌석 노쇼 해제됨")

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

        # 추가: 서버 시작 시 테스트용 DB 자동 초기화
        await conn.execute(text("TRUNCATE TABLE seats RESTART IDENTITY CASCADE"))
        await conn.execute(text("TRUNCATE TABLE cameras RESTART IDENTITY CASCADE"))

    task = asyncio.create_task(periodic_seat_release(interval=10))
    yield
    task.cancel()

# ── FastAPI 앱 선언 (엔드포인트보다 반드시 위에 있어야 함) ──────────────────────────

app = FastAPI(title="Seat Management Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# YOLO 서버 → 메인 서버
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/detections", summary="YOLO 탐지 결과 수신")
async def receive_detections(payload: DetectionPayload, db: AsyncSession = Depends(get_db)):
    try:
        seats = await process_detections(
            db=db,
            camera_id=payload.camera_id,
            timestamp=payload.timestamp,
            detections=payload.detections,
            clear_all=payload.clear_all,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"processed": len(payload.detections), "seats_updated": len(seats)}

@app.post("/sensor-data", summary="HW 센서 데이터 수신")
async def receive_sensor_data(
    payload: SensorPayload,
    db: AsyncSession = Depends(get_db)
):
    seat = await crud.update_sensor_status(
        db,
        payload.seat_id,
        payload.occupied
    )

    if not seat:
        raise HTTPException(status_code=404, detail="좌석을 찾을 수 없습니다.")

    return {
        "success": True,
        "seat_id": seat.id,
        "sensor_occupied": seat.sensor_occupied,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 메인 서버 → React 웹
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/seats", response_model=list[SeatStatusResponse], summary="전체 카메라 좌석 현황 조회")
async def get_all_seats(db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    # 직접 select문을 써서 가져오는 방식으로 안정성 강화
    cameras_result = await db.execute(select(models.Camera))
    cameras = cameras_result.scalars().all()
    
    response = []
    for camera in cameras:
        seats_result = await db.execute(select(models.Seat).where(models.Seat.camera_id == camera.id))
        seats = seats_result.scalars().all()
        response.append(
            SeatStatusResponse(
                camera_id=camera.id,
                seats=[_to_seat_status(s, now) for s in seats],
                fetched_at=now,
            )
        )
    return response

@app.get("/seats/{camera_id}", response_model=SeatStatusResponse, summary="카메라별 좌석 현황 조회")
async def get_seat_status(camera_id: str, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    result = await db.execute(select(models.Camera).where(models.Camera.id == camera_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail=f"카메라 없음: {camera_id}")
    
    seats_result = await db.execute(select(models.Seat).where(models.Seat.camera_id == camera_id))
    seats = seats_result.scalars().all()
    return SeatStatusResponse(
        camera_id=camera_id,
        seats=[_to_seat_status(s, now) for s in seats],
        fetched_at=now,
    )

@app.get("/seats/{camera_id}/{seat_id}", response_model=SeatStatus, summary="단일 좌석 상세 조회")
async def get_seat_detail(camera_id: str, seat_id: int, db: AsyncSession = Depends(get_db)):
    seat = await crud.get_seat(db, seat_id)
    if not seat or seat.camera_id != camera_id:
        raise HTTPException(status_code=404, detail="좌석을 찾을 수 없습니다.")
    return _to_seat_status(seat, datetime.now(timezone.utc))

@app.post("/reservations", response_model=ReservationResponse, summary="좌석 예약")
async def reserve_seat(payload: ReservationRequest, db: AsyncSession = Depends(get_db)):
    success, status, seat = await crud.reserve_seat(db, payload.seat_id)
    if status == "not_found":
        raise HTTPException(status_code=404, detail="존재하지 않는 좌석입니다.")
    if status == "already_occupied":
        return ReservationResponse(
            success=False,
            message="이미 사용 중인 좌석입니다.",
            seat_id=payload.seat_id,
            status=SeatStatusEnum.using,
        )
    return ReservationResponse(
        success=True, message="예약이 완료되었습니다.", seat_id=payload.seat_id, status=SeatStatusEnum.using,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 관리용 API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/admin/cameras", response_model=CameraResponse)
async def create_camera(payload: CameraCreate, db: AsyncSession = Depends(get_db)):
    if await crud.get_camera(db, payload.id):
        raise HTTPException(status_code=409, detail="이미 존재하는 카메라 ID입니다.")
    return await crud.create_camera(db, payload)

@app.get("/admin/cameras", response_model=List[schemas.CameraResponse])
async def read_all_cameras(db: AsyncSession = Depends(get_db)):
    cameras = await crud.get_all_cameras(db)
    return cameras

@app.get("/admin/cameras/{camera_id}", response_model=schemas.CameraResponse)
async def read_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    camera = await crud.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다.")
    return camera

@app.put("/admin/cameras/{camera_id}/homography", response_model=CameraResponse)
async def update_homography(camera_id: str, payload: HomographyUpdate, db: AsyncSession = Depends(get_db)):
    from services.homography import compute_homography
    try:
        matrix = compute_homography([{"src": p.src, "dst": p.dst} for p in payload.points])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    camera = await crud.update_homography(db, camera_id, matrix)
    if not camera:
        raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다.")
    return camera

@app.post("/admin/seats", response_model=SeatResponse)
async def create_seat(payload: SeatCreate, db: AsyncSession = Depends(get_db)):
    if not await crud.get_camera(db, payload.camera_id):
        raise HTTPException(status_code=404, detail="카메라를 먼저 등록하세요.")
    return await crud.create_seat(db, payload)

@app.delete("/admin/seats/{seat_id}")
async def delete_seat(seat_id: int, db: AsyncSession = Depends(get_db)):
    if not await crud.delete_seat(db, seat_id):
        raise HTTPException(status_code=404, detail="좌석을 찾을 수 없습니다.")
    return {"deleted": seat_id}

@app.get("/health")
async def health():
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# MySeatPage API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/reservations/me", response_model=list[SeatStatus], summary="현재 예약한 좌석 조회")
async def get_my_reservations(db: AsyncSession = Depends(get_db)):
    """
    수정:
    - 예약된 좌석 전체 조회
    - 착석대기(status=vacant + is_booked=True) 상태 지원
    """

    result = await db.execute(
        select(models.Seat).where(
            models.Seat.is_booked == True
        )
    )

    seats = result.scalars().all()
    now = datetime.now(timezone.utc)

    return [_to_seat_status(seat, now) for seat in seats]


@app.delete("/reservations/{seat_id}", summary="좌석 반납")
async def return_seat(seat_id: int, db: AsyncSession = Depends(get_db)):
    seat = await crud.get_seat(db, seat_id)

    if not seat:
        raise HTTPException(status_code=404, detail="좌석을 찾을 수 없습니다.")

    # 상태 초기화
    seat.status = SeatStatusEnum.vacant
    seat.vacant_since = None

    # 추가:
    # 반납 시 예약 상태까지 완전히 해제
    seat.is_booked = False
    seat.is_occupied = False
    seat.sensor_occupied = False

    seat.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(seat)

    return {
        "success": True,
        "message": "좌석이 반납되었습니다.",
        "seat_id": seat.id,
        "status": seat.status,
    }

@app.patch("/reservations/{seat_id}/temp", summary="일시비움")
async def temp_leave_seat(seat_id: int, db: AsyncSession = Depends(get_db)):
    seat = await crud.get_seat(db, seat_id)

    if not seat:
        raise HTTPException(status_code=404, detail="좌석을 찾을 수 없습니다.")

    if seat.status != SeatStatusEnum.using:
        raise HTTPException(
            status_code=400,
            detail="사용 중인 좌석만 일시비움 가능합니다."
        )

    seat.status = SeatStatusEnum.temp
    seat.vacant_since = datetime.now(timezone.utc)
    seat.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(seat)

    return {
        "success": True,
        "message": "일시비움 처리되었습니다.",
        "seat_id": seat.id,
        "status": seat.status,
    }


@app.patch("/reservations/{seat_id}/return", summary="복귀")
async def return_from_temp(seat_id: int, db: AsyncSession = Depends(get_db)):
    seat = await crud.get_seat(db, seat_id)

    if not seat:
        raise HTTPException(status_code=404, detail="좌석을 찾을 수 없습니다.")

    if seat.status != SeatStatusEnum.temp:
        raise HTTPException(
            status_code=400,
            detail="일시비움 상태인 좌석만 복귀 가능합니다."
        )

    seat.status = SeatStatusEnum.using
    seat.vacant_since = None
    seat.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(seat)

    return {
        "success": True,
        "message": "복귀 처리되었습니다.",
        "seat_id": seat.id,
        "status": seat.status,
    }