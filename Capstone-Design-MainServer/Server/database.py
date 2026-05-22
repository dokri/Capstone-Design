import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# .env 파일 로드
load_dotenv(dotenv_path=".env", override=True)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "Server/.env 파일에서 DATABASE_URL 항목을 찾을 수 없습니다."
    )

# --- 커넥션 풀 설정 값 (환경 변수에서 읽고, 없으면 기본값 사용) ---
# pool_size: 동시에 유지할 기본 연결 수
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
# max_overflow: pool_size를 초과해서 추가로 허용할 연결 수
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "40"))

engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_size=DB_POOL_SIZE,       # 추가
    max_overflow=DB_MAX_OVERFLOW, # 추가
    pool_recycle=1800,            # 추가: 30분마다 연결 재사용 (DB 연결 끊김 방지)
    pool_timeout=30               # 추가: 연결 대기 시간 최대 30초
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=True,
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close() # 확실하게 닫아줌