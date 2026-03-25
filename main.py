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
    SeatStatusResponse,
    CameraCreate,
    CameraResponse,
    SeatCreate,
    SeatResponse,
)
from services.seat_mapper import process_detections, release_timed_out_seats


# 백그라운드 태스크
async def periodic_seat_release(interval: int = 10):
    """interval초마다 timeout된 좌석 자동 해제 — 이벤트 루프 블로킹 없음"""
    while True:
        await asyncio.sleep(interval)
        async with AsyncSessionLocal() as db:          # async 세션 직접 생성
            released = await release_timed_out_seats(db)
            if released:
                print(f"[auto-release] {released}개 좌석 해제됨")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB 테이블 생성 (async)
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    task = asyncio.create_task(periodic_seat_release(interval=10))
    yield
    task.cancel()

# -----------------------------------------------------------------------------


# FastAPI 앱
app = FastAPI(title="Seat Management Server", lifespan=lifespan)

# 미들웨어 등록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 권한 허용, 배포시 필요한 도메인만 허용하도록 변경
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

@app.get("/seats/{camera_id}", response_model=SeatStatusResponse, summary="카메라별 좌석 현황 조회")
async def get_seat_status(camera_id: str, db: AsyncSession = Depends(get_db)):
    if not await crud.get_camera(db, camera_id):
        raise HTTPException(status_code=404, detail=f"카메라 없음: {camera_id}")

    seats = await crud.get_seats_by_camera(db, camera_id)
    return SeatStatusResponse(
        camera_id=camera_id,
        seats=[SeatStatus(id=s.id, name=s.name, is_occupied=s.is_occupied, last_updated=s.updated_at) for s in seats],
        fetched_at=datetime.now(timezone.utc),
    )


@app.get("/seats", response_model=list[SeatStatusResponse], summary="전체 카메라 좌석 현황 조회")
async def get_all_seats(db: AsyncSession = Depends(get_db)):
    cameras = await crud.get_all_cameras(db)
    result = []
    for camera in cameras:
        seats = await crud.get_seats_by_camera(db, camera.id)
        result.append(SeatStatusResponse(
            camera_id=camera.id,
            seats=[SeatStatus(id=s.id, name=s.name, is_occupied=s.is_occupied, last_updated=s.updated_at) for s in seats],
            fetched_at=datetime.now(timezone.utc),
        ))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 관리용 API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/admin/cameras", response_model=CameraResponse, summary="카메라 등록")
async def create_camera(payload: CameraCreate, db: AsyncSession = Depends(get_db)):
    if await crud.get_camera(db, payload.id):
        raise HTTPException(status_code=409, detail="이미 존재하는 카메라 ID입니다.")
    return await crud.create_camera(db, payload)


@app.put("/admin/cameras/{camera_id}/homography", response_model=CameraResponse, summary="Homography 행렬 업데이트")
async def update_homography(camera_id: str, matrix: list[float], db: AsyncSession = Depends(get_db)):
    if len(matrix) != 9:
        raise HTTPException(status_code=400, detail="Homography 행렬은 9개의 값이어야 합니다 (3x3 flatten).")
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
