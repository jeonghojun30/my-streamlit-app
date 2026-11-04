# ==============================================================================
# 파일명: app.py
# 목적: AI 개인비서 '에이블' Streamlit 사용자 인터페이스(UI) 구현
# 주요 기능:
#   1. 회의록 파일(.txt) 업로드 및 내용 확인 (출처: 3 - 3. 텍스트 파일 읽기)
#   2. 사용자 명령 입력 (텍스트 입력 및 음성 녹음 STT 지원) (출처: 7 - 2. PY 파일 만들기)
#   3. LLM 기반 페르소나 응답 및 회의록 요약 (출처: 6 - 미션 2, 3)
#   4. 함수 호출(Tool Calling)을 통한 이메일 자동 발송 (출처: 6 - 미션 1, 7)
#   5. AI 응답 음성 합성(TTS) 및 오디오 재생 (출처: 6 - 미션 5)
#   6. 세션 상태 기반 대화 이력 관리 (출처: 2 - 5. 리스트 상태 - 히스토리 저장)
# ==============================================================================

import os
import json
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from openai import OpenAI

# ----------------------------------------------------
# 1. 페이지 설정 및 초기 환경 구성
# ----------------------------------------------------
# (출처: 4 - 4. 출력 함수 맛보기) 브라우저 탭 타이틀, 아이콘 및 와이드 레이아웃 설정
st.set_page_config(
    page_title="AI 개인비서 에이블",
    page_icon="🤖",
    layout="wide"
)

# (출처: 6 - 환경 준비 (5)) OpenAI API 클라이언트 초기화
# Streamlit Cloud의 Secrets나 시스템 환경 변수에 등록된 OPENAI_API_KEY를 자동으로 참조합니다.
client = OpenAI()

# (출처: 2 - 4. 상태 키 초기화 패턴) 세션 상태(st.session_state) 초기화
# Streamlit 특성상 위젯 조작 시 스크립트가 재실행되므로, 데이터를 유지하기 위해 session_state에 등록합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []          # 대화 이력 (role, content 딕셔너리 리스트)
if "meeting_content" not in st.session_state:
    st.session_state.meeting_content = ""   # 업로드된 회의록 텍스트 저장용
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None     # TTS로 생성된 답변 음성 바이너리 저장용


# ----------------------------------------------------
# 2. 외부 도구 및 유틸리티 함수 정의
# ----------------------------------------------------
# (출처: 6 - 미션 1. 이메일 발송) Gmail SMTP를 활용한 이메일 발송 함수
def send_email(to: str, subject: str, body: str) -> dict:
    """
    지정된 수신자에게 이메일을 발송합니다.
    환경변수(Secrets)의 GMAIL_ADDRESS와 GMAIL_APP_PASSWORD를 발신 정보로 사용합니다.
    """
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    # 인증 정보 유효성 검사 (출처: 3 - 7. 파일이 없을 때 UX 처리 패턴 응용)
    if not sender or not password:
        return {"status": "failed", "error": "GMAIL 설정 정보가 환경변수(Secrets)에 없습니다."}

    # MIMEText를 이용한 메일 메시지 본문 및 헤더 구성
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        # Gmail SMTP 서버(587 포트) 접속 및 TLS 암호화 연결
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# (출처: 6 - 미션 7. 도구 사용) OpenAI Function Calling 규격에 맞춘 도구(Tool) 명세 정의
# LLM이 사용자 요청에 '이메일 발송' 의도가 포함되어 있다고 판단할 때 이 명세를 참조하여 호출합니다.
tools = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "사용자가 요약이나 업무 내용을 이메일로 발송해 달라고 명시적으로 요청했을 때만 메일을 발송합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "메일을 수신할 상대방의 이메일 주소"
                    },
                    "subject": {
                        "type": "string",
                        "description": "이메일 제목"
                    },
                    "body": {
                        "type": "string",
                        "description": "이메일 본문 내용 (회의 요약 또는 업무 정리 내용 포함)"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]


# ----------------------------------------------------
# 3. 사이드바 UI 구성 (설정 및 비서 페르소나 관리)
# ----------------------------------------------------
# (출처: 1 - 4. 사이드바 / 1 - 5. 사이드바 + 본 화면 분리 패턴)
# 설정 및 입력 옵션은 사이드바로 모아 본 화면을 결과 중심으로 유지합니다.
with st.sidebar:
    st.header("⚙️ 비서 설정")
    
    # 기본 수신자 이메일 주소 입력 위젯 (미입력 시 본인 Gmail 계정 기본값 적용)
    recipient_email = st.text_input("수신자 이메일 주소", value=os.getenv("GMAIL_ADDRESS", ""))
    
    # (출처: 1 - 7. 확장 영역 Expander) 고급 옵션 및 시스템 프롬프트 관리
    with st.expander("에이블 시스템 페르소나 설정"):
        # (출처: 6 - 미션 3. 역할 부여) AI 비서 '에이블'의 기본 지침 정의
        default_role = """당신은 '김 대리'의 개인 비서 '에이블'입니다.
김 대리의 요청에 친절하고 정확하게 한국어로 정중하게 답변합니다.
담당 업무:
1. 미팅 내용을 읽고 핵심 내용을 간결하게 요약합니다.
2. 미팅에서 김 대리에게 배정된 업무가 있으면 업무 내용, 기한, 관련자를 자세히 정리합니다.
3. 배정된 업무가 없으면 "배정된 업무가 없습니다"라고 알립니다.
4. 사실에 없는 내용은 지어내지 않습니다.
5. 이메일 발송 요청이 있을 경우에만 send_email 도구를 호출합니다."""
        
        # 시스템 프롬프트 수정 가능 영역 (출처: 5 - 4. 기본 입력 위젯 - 텍스트)
        sys_role = st.text_area("시스템 프롬프트", value=default_role, height=220)

    # (출처: 2 - 3. 카운터 앱 - 상태 적용 및 초기화 버튼) 세션 데이터 초기화 버튼
    if st.button("대화 내용 초기화", use_container_width=True):
        st.session_state.messages.clear()
        st.session_state.audio_bytes = None
        st.session_state.meeting_content = ""
        st.rerun()


# ----------------------------------------------------
# 4. 메인 화면 레이아웃 (좌우 2열 분할 구조)
# ----------------------------------------------------
# (출처: 4 - 4. 출력 함수 / 1 - 8. 구분선과 섹션 제목)
st.title("🤖 AI 개인비서 '에이블'")
st.markdown("회의록 텍스트 분석, 텍스트/음성 명령, 미팅 내용 요약 및 이메일 전송을 지원합니다.")
st.markdown("---")

# (출처: 1 - 2. 컬럼 비율 조정 / 1 - 미니 실습 1) 좌우 1:1 동일 비율로 레이아웃 분할
col1, col2 = st.columns([1, 1])

# ====================================================
# [좌측 컬럼]: 사용자 입력 및 명령 제출 패널
# ====================================================
with col1:
    st.subheader("📥 입력 영역")

    # (출처: 3 - 3. 텍스트 파일 읽기 / 3 - 7. 파일 UX 처리) 회의록 파일 업로더
    uploaded_file = st.file_uploader("미팅 내용 파일 업로드 (.txt)", type=["txt"])
    if uploaded_file is not None:
        # 업로드된 파일 바이트를 utf-8 문자열로 변환하여 세션 상태에 저장
        st.session_state.meeting_content = uploaded_file.read().decode("utf-8")
        st.success(f"파일 로드 완료: {uploaded_file.name}")
        
        # 파일 내용을 접어서 볼 수 있는 Expander 배치 (출처: 1 - 7. 확장 영역)
        with st.expander("미팅 내용 본문 미리보기"):
            st.text_area("미팅 본문", st.session_state.meeting_content, height=150, disabled=True)
    else:
        st.info("회의록 텍스트(.txt) 파일을 업로드하면 회의 요약 및 업무 추출이 가능합니다.")

    # (출처: 2 - 7. st.form 이벤트 묶기 / 7 - 2. 지시 영역 구성)
    # 위젯 입력 시마다 페이지가 재실행되는 것을 방지하고 버튼 클릭 시 일괄 전송
    with st.form("command_form", clear_on_submit=False):
        # 텍스트 입력창 (출처: 5 - 4. 텍스트 입력)
        text_query = st.text_input("텍스트 질문/명령 입력 (음성 미사용 시)")
        
        # 음성 입력 위젯 (출처: 7 - 2. st.audio_input)
        audio_query = st.audio_input("또는 음성으로 명령하기")
        
        # 제출 버튼 (출처: 5 - 7. 버튼 처리 시점 제어)
        submit_btn = st.form_submit_button("요청 처리하기", use_container_width=True)


# ====================================================
# [요청 처리 로직]: 제출 버튼 클릭 시 실행
# ====================================================
if submit_btn:
    user_prompt = ""

    # 1) 음성 입력 우선 처리 (STT: Speech-to-Text)
    # (출처: 6 - 미션 4: STT / 7 - 2. STT 변환)
    if audio_query is not None:
        with st.spinner("음성을 텍스트로 변환 중입니다..."):
            try:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", audio_query, "audio/wav")
                )
                user_prompt = transcript.text
            except Exception as e:
                st.error(f"음성 변환(STT) 오류: {e}")
                st.stop()
    # 2) 음성이 없을 경우 텍스트 입력값 사용
    elif text_query.strip():
        user_prompt = text_query.strip()
    # 3) 입력이 둘 다 없을 경우 중단 (출처: 3 - 7. 조건 검증 및 경고 처리)
    else:
        st.warning("텍스트를 입력하거나 음성을 녹음해 주세요.")
        st.stop()

    # 4) 프롬프트 결합 (미팅 회의록 내용 + 사용자 지시사항)
    # (출처: 6 - 미션 3, 4: 프롬프트 구성)
    final_user_content = user_prompt
    if st.session_state.meeting_content:
        final_user_content += f"\n\n[미팅 내용]\n{st.session_state.meeting_content}"

    # 사용자 질문을 대화 이력에 추가 (출처: 2 - 미니 실습 2. 채팅 UI 구조)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 5) LLM 호출 및 도구 실행(Function Calling)
    # (출처: 6 - 미션 2, 7: 기본 응답 및 도구 사용)
    with st.spinner("에이블이 요청을 분석하고 답변을 준비하고 있습니다..."):
        try:
            # API에 전달할 메시지 배열 구성 (시스템 역할 + 기존 대화 내역 누적)
            api_messages = [{"role": "system", "content": sys_role}]
            for msg in st.session_state.messages[:-1]:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            # 현재 턴의 최종 프롬프트 추가
            api_messages.append({"role": "user", "content": final_user_content})

            # LLM 1차 호출 (도구 호출 여부 판단)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # LLM이 도구(이메일 발송) 호출을 요청한 경우 처리
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "send_email":
                        # 모델이 추출한 인자(수신자, 제목, 본문) 파싱
                        args = json.loads(tool_call.function.arguments)
                        target_to = args.get("to") or recipient_email
                        
                        # 실제 이메일 발송 실행 (출처: 6 - 미션 1)
                        send_res = send_email(
                            to=target_to,
                            subject=args.get("subject", "미팅 요약 및 업무 보고"),
                            body=args.get("body", "")
                        )
                        
                        # 이메일 발송 결과 알림
                        if send_res["status"] == "success":
                            st.success(f"이메일을 성공적으로 전송했습니다. (수신자: {target_to})")
                        else:
                            st.error(f"이메일 전송 실패: {send_res['error']}")

                        # 도구 실행 결과를 LLM 맥락에 추가하여 최종 응답 생성 유도
                        api_messages.append(response_message)
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(send_res)
                        })
                        
                        # LLM 2차 호출 (발송 결과 안내 문구 생성)
                        second_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=api_messages
                        )
                        final_text = second_response.choices[0].message.content
            else:
                # 도구 호출이 없는 일반 응답
                final_text = response_message.content

            # 최종 답변을 대화 이력에 추가
            st.session_state.messages.append({"role": "assistant", "content": final_text})

            # 6) 음성 합성(TTS: Text-to-Speech) 처리
            # (출처: 6 - 미션 5: 음성으로 듣기)
            tts_res = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=final_text[:1000]  # 브라우저 재생 및 응답 속도를 위해 입력 길이 조절
            )
            # 생성된 음성 바이너리 데이터를 세션 상태에 보관
            st.session_state.audio_bytes = tts_res.content

        except Exception as e:
            st.error(f"LLM 응답 처리 중 오류 발생: {e}")


# ====================================================
# [우측 컬럼]: 결과 출력 및 대화 히스토리 패널
# ====================================================
with col2:
    st.subheader("📤 결과 및 대화 기록")

    # (출처: 6 - 미션 5, 6: 음성 출력 위젯) 음성 답변이 생성되어 있으면 오디오 플레이어 노출
    if st.session_state.audio_bytes:
        st.write("🔊 **에이블의 음성 답변:**")
        st.audio(st.session_state.audio_bytes, format="audio/mp3")

    # (출처: 2 - 5. 리스트 상태 - 히스토리 순회 출력 / 2 - 미니 실습 2) 대화 내역 렌더링
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
    else:
        st.info("회의록을 등록하고 질문이나 명령을 입력하면 분석 결과가 여기에 표시됩니다.")
