"""에이블: 사용자 정보 등록 화면 (단일 사용자용 MVP).

실행: streamlit run app.py
저장: 프로젝트 폴더의 data/user_profile.db (SQLite)

주의: 이 버전은 개인 실습/로컬 시연용입니다. 인증 기능이 없으므로
공개된 멀티유저 서비스로 배포하지 마세요.
Streamlit Community Cloud는 로컬 파일의 영구 보존을 보장하지 않습니다.
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import streamlit as st


# ------------------------------------------------------------
# 1. 영구 저장 경로와 데이터베이스 함수
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ABLE_DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = DATA_DIR / "user_profile.db"
PROFILE_ID = 1  # 이 버전은 한 명만 사용하는 개인 비서 프로토타입입니다.
EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)


def init_database() -> None:
    """DB 파일과 사용자 테이블을 없으면 생성합니다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def get_saved_profile() -> dict[str, str] | None:
    """이전에 입력한 이름과 이메일을 DB에서 불러옵니다."""
    init_database()
    with closing(sqlite3.connect(DB_PATH)) as connection:
        row = connection.execute(
            "SELECT name, email FROM user_profile WHERE id = ?", (PROFILE_ID,)
        ).fetchone()
    return {"name": row[0], "email": row[1]} if row else None


def save_profile(name: str, email: str) -> None:
    """사용자 정보를 저장하거나 기존 정보를 수정합니다."""
    init_database()
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute(
            """
            INSERT INTO user_profile (id, name, email)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                email = excluded.email,
                updated_at = CURRENT_TIMESTAMP
            """,
            (PROFILE_ID, name.strip(), email.strip().lower()),
        )
        connection.commit()


def get_saved_email() -> str | None:
    """후속 이메일 발송·캘린더 연결 코드에서 사용할 이메일 주소입니다."""
    profile = get_saved_profile()
    return profile["email"] if profile else None


def delete_profile() -> None:
    """사용자가 요청하면 저장된 정보를 삭제합니다."""
    init_database()
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute("DELETE FROM user_profile WHERE id = ?", (PROFILE_ID,))
        connection.commit()


# ------------------------------------------------------------
# 2. 화면 구성
# ------------------------------------------------------------
def registration_screen(existing_profile: dict[str, str] | None) -> None:
    """첫 방문에는 등록 화면, 수정할 때에는 기존 값이 채워진 화면을 표시합니다."""
