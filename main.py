from contextlib import asynccontextmanager
from datetime import datetime, timezone

import asyncio
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

import models
import crud
from database import engine, get_db, AsyncSessionLocal
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
)
from services.seat_mapper import process_detections, release_timed_out_seats


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def _calc_vacant_seconds(seat, now: datetime):
    """vacant 상태일 때만 경과 초를 계산, 나머지는 None"""
    if seat.status == SeatStatusEnum.vacant and seat.vacant_since is not None:
        vs = seat.vacant_since
        if vs.tzinfo is None:
            vs = vs.replace(tzinfo=timezone.utc)
        return (now - vs).total_seconds()
    return None


def _to_seat_status(seat, now: datetime) -> SeatStatus:
    return SeatStatus(
        id=seat.id,
        name=seat.name,
        status=seat.status,
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
                print(f"[auto-release] {released}개 좌석 해제됨")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    task = asyncio.create_task(periodic_seat_release(interval=10))
    yield
    task.cancel()


# ── FastAPI 앱 ─────────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# 메인 서버 → React 웹
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/seats", response_model=list[SeatStatusResponse], summary="전체 카메라 좌석 현황 조회")
async def get_all_seats(db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    cameras = await crud.get_all_cameras(db)
    return [
        SeatStatusResponse(
            camera_id=camera.id,
            seats=[_to_seat_status(s, now) for s in await crud.get_seats_by_camera(db, camera.id)],
            fetched_at=now,
        )
        for camera in cameras
    ]


@app.get("/seats/{camera_id}", response_model=SeatStatusResponse, summary="카메라별 좌석 현황 조회")
async def get_seat_status(camera_id: str, db: AsyncSession = Depends(get_db)):
    if not await crud.get_camera(db, camera_id):
        raise HTTPException(status_code=404, detail=f"카메라 없음: {camera_id}")
    now = datetime.now(timezone.utc)
    seats = await crud.get_seats_by_camera(db, camera_id)
    return SeatStatusResponse(
        camera_id=camera_id,
        seats=[_to_seat_status(s, now) for s in seats],
        fetched_at=now,
    )


@app.get("/seats/{camera_id}/{seat_id}", response_model=SeatStatus, summary="단일 좌석 상세 조회")
async def get_seat_detail(camera_id: str, seat_id: int, db: AsyncSession = Depends(get_db)):
    """웹에서 좌석을 클릭했을 때 호출합니다."""
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
        success=True,
        message="예약이 완료되었습니다.",
        seat_id=payload.seat_id,
        status=SeatStatusEnum.using,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 관리용 API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/admin/cameras", response_model=CameraResponse, summary="카메라 등록")
async def create_camera(payload: CameraCreate, db: AsyncSession = Depends(get_db)):
    if await crud.get_camera(db, payload.id):
        raise HTTPException(status_code=409, detail="이미 존재하는 카메라 ID입니다.")
    return await crud.create_camera(db, payload)


@app.put("/admin/cameras/{camera_id}/homography", response_model=CameraResponse, summary="Homography 행렬 업데이트")
async def update_homography(camera_id: str, payload: HomographyUpdate, db: AsyncSession = Depends(get_db)):
    """
    카메라 이미지 좌표 ↔ 실제 좌석 좌표 대응점 4쌍을 받아
    Homography 행렬을 계산 후 저장합니다.
    """
    from services.homography import compute_homography
    try:
        matrix = compute_homography([{"src": p.src, "dst": p.dst} for p in payload.points])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    camera = await crud.update_homography(db, camera_id, matrix)
    if not camera:
        raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다.")
    return camera


@app.post("/admin/seats", response_model=SeatResponse, summary="좌석 등록")
async def create_seat(payload: SeatCreate, db: AsyncSession = Depends(get_db)):
    if not await crud.get_camera(db, payload.camera_id):
        raise HTTPException(status_code=404, detail="카메라를 먼저 등록하세요.")
    return await crud.create_seat(db, payload)


@app.delete("/admin/seats/{seat_id}", summary="좌석 삭제")
async def delete_seat(seat_id: int, db: AsyncSession = Depends(get_db)):
    if not await crud.delete_seat(db, seat_id):
        raise HTTPException(status_code=404, detail="좌석을 찾을 수 없습니다.")
    return {"deleted": seat_id}


@app.get("/health")
async def health():
    return {"status": "ok"}