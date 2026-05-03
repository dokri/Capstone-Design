import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# 현재 실행 경로에 있는 .env 파일을 강제로 읽어옵니다.
# override=True를 주면 시스템 변수보다 .env 파일 내용을 우선시합니다.
load_dotenv(dotenv_path=".env", override=True)

DATABASE_URL = os.getenv("DATABASE_URL")

# 디버깅용: 만약 여기서 에러가 난다면 진짜 파일 내용이 문제인 겁니다.
if not DATABASE_URL:
    raise RuntimeError(
        "Server/.env 파일은 존재하지만, 그 안에서 DATABASE_URL 항목을 찾을 수 없습니다. "
        "파일 내용이 'DATABASE_URL=...' 형식인지 확인하세요."
    )

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session