# (출처: 4 - 1. 첫 앱 실행) %%writefile app.py 패턴 적용
# (출처: 7 - 2. PY 파일 만들기) AI 개인비서 '에이블' 인터페이스 및 단계 1 기능 연동

import os
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from openai import OpenAI

# ----------------------------------------------------
# 1. 페이지 설정 및 초기화
# ----------------------------------------------------
# (출처: 4 - 4. 출력 함수 맛보기) st.set_page_config 설정
st.set_page_config(
    page_title="AI 개인비서 에이블",
    page_icon="🤖",
    layout="wide"
)

# OpenAI 클라이언트 인스턴스 생성 (Streamlit Cloud Secrets / 환경변수 자동 연동)
# (출처: 6 - 환경 준비 (5))
client = OpenAI()

# (출처: 2 - 4. 상태 키 초기화 패턴) 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "meeting_content" not in st.session_state:
    st.session_state.meeting_content = ""
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# ----------------------------------------------------
# 2. 유틸리티 함수 정의
# ----------------------------------------------------
# (출처: 6 - 미션 1. 이메일 발송) 이메일 발송 함수
def send_email(to: str, subject: str, body: str) -> dict:
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender or not password:
        return {"status": "failed", "error": "GMAIL 설정 정보가 환경변수(Secrets)에 없습니다."}

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# (출처: 6 - 미션 7. 도구 사용) 함수 호출용 도구 선언
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
                        "description": "수신자 이메일 주소"
                    },
                    "subject": {
                        "type": "string",
                        "description": "이메일 제목"
                    },
                    "body": {
                        "type": "string",
                        "description": "이메일 본문 내용"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]


# ----------------------------------------------------
# 3. 사이드바 구성 (설정 및 이메일 수신자 관리)
# ----------------------------------------------------
# (출처: 1 - 4. 사이드바 / 5. 사이드바 + 본 화면 분리 패턴)
with st.sidebar:
    st.header("⚙️ 비서 설정")
    recipient_email = st.text_input("수신자 이메일 주소", value=os.getenv("GMAIL_ADDRESS", ""))
    
    # (출처: 1 - 7. 확장 영역 Expander)
    with st.expander("에이블 시스템 페르소나"):
        # (출처: 6 - 미션 3. 역할 부여)
        default_role = """당신은 '김 대리'의 개인 비서 '에이블'입니다.
김 대리의 요청에 친절하고 정확하게 한국어로 정중하게 답변합니다.
담당 업무:
1. 미팅 내용을 읽고 핵심 내용을 간결하게 요약합니다.
2. 미팅에서 김 대리에게 배정된 업무가 있으면 업무 내용, 기한, 관련자를 자세히 정리합니다.
3. 배정된 업무가 없으면 "배정된 업무가 없습니다"라고 알립니다.
4. 사실에 없는 내용은 지어내지 않습니다.
5. 이메일 발송 요청이 있을 경우에만 send_email 도구를 호출합니다."""
        sys_role = st.text_area("시스템 프롬프트", value=default_role, height=220)

    # (출처: 2 - 3. 카운터 앱 - 상태 적용 및 초기화 버튼)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages.clear()
        st.session_state.audio_bytes = None
        st.session_state.meeting_content = ""
        st.rerun()


# ----------------------------------------------------
# 4. 메인 화면 레이아웃
# ----------------------------------------------------
# (출처: 4 - 4. 출력 함수)
st.title("🤖 AI 개인비서 '에이블'")
st.markdown("회의록 파일 업로드, 텍스트/음성 명령, 요약 및 이메일 전송을 통합 처리합니다.")
st.markdown("---")  # (출처: 1 - 8. 구분선과 섹션 제목)

# (출처: 1 - 2. 컬럼 비율 조정: 1:1 비율 분할)
col1, col2 = st.columns([1, 1])

# ----------------- 좌측 컬럼: 입력 패널 -----------------
with col1:
    st.subheader("📥 입력 영역")

    # (출처: 3 - 3. 텍스트 파일 읽기 / 3 - 7. 파일 UX 처리)
    uploaded_file = st.file_uploader("미팅 내용 파일 업로드 (.txt)", type=["txt"])
    if uploaded_file is not None:
        st.session_state.meeting_content = uploaded_file.read().decode("utf-8")
        st.success(f"파일 로드 완료: {uploaded_file.name}")
        with st.expander("미팅 내용 미리보기"):
            st.text_area("미팅 본문", st.session_state.meeting_content, height=150, disabled=True)
    else:
        st.info("회의록 텍스트(.txt)를 업로드하면 더 정밀한 요약이 가능합니다.")

    # (출처: 2 - 7. st.form 이벤트 묶기 / 7 - 2. 지시 영역 구성)
    with st.form("command_form", clear_on_submit=False):
        text_query = st.text_input("텍스트 질문/명령 입력 (음성 미사용 시)")
        
        # (출처: 7 - 2. st.audio_input)
        audio_query = st.audio_input("또는 음성으로 명령하기")
        
        # (출처: 5 - 7. 버튼 처리 시점 제어)
        submit_btn = st.form_submit_button("요청 처리하기", use_container_width=True)


# ----------------- 요청 처리 로직 -----------------
if submit_btn:
    user_prompt = ""

    # (1) 음성 입력 우선 확인 (STT 처리)
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
    elif text_query.strip():
        user_prompt = text_query.strip()
    else:
        # (출처: 3 - 7. 조건 검증 및 경고 메시지)
        st.warning("텍스트를 입력하거나 음성을 녹음해 주세요.")
        st.stop()

    # (2) 프롬프트 결합 (미팅 내용 + 사용자 지시)
    # (출처: 6 - 미션 3, 4: 프롬프트 구성)
    final_user_content = user_prompt
    if st.session_state.meeting_content:
        final_user_content += f"\n\n[미팅 내용]\n{st.session_state.meeting_content}"

    # (출처: 2 - 미니 실습 2. 메시지 구조 누적)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # (3) LLM 호출 및 Function Calling 처리
    # (출처: 6 - 미션 2, 7)
    with st.spinner("에이블이 처리 중입니다..."):
        try:
            # 대화 맥락 구성
            api_messages = [{"role": "system", "content": sys_role}]
            # 이전 대화 기록 추가
            for msg in st.session_state.messages[:-1]:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            # 이번 입력 프롬프트(미팅 내용 포함) 전달
            api_messages.append({"role": "user", "content": final_user_content})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # 이메일 전송 도구 호출 확인
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "send_email":
                        import json
                        args = json.loads(tool_call.function.arguments)
                        target_to = args.get("to") or recipient_email
                        send_res = send_email(
                            to=target_to,
                            subject=args.get("subject", "미팅 요약 및 업무 보고"),
                            body=args.get("body", "")
                        )
                        if send_res["status"] == "success":
                            st.success(f"이메일을 성공적으로 전송했습니다. (수신자: {target_to})")
                        else:
                            st.error(f"이메일 전송 실패: {send_res['error']}")

                        # 도구 실행 결과를 바탕으로 최종 응답 생성
                        api_messages.append(response_message)
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(send_res)
                        })
                        second_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=api_messages
                        )
                        final_text = second_response.choices[0].message.content
            else:
                final_text = response_message.content

            # 최종 응답 세션에 저장
            st.session_state.messages.append({"role": "assistant", "content": final_text})

            # (출처: 6 - 미션 5: 음성으로 듣기 TTS)
            tts_res = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=final_text[:1000]  # 브라우저 재생용 핵심 요약 길이 제한
            )
            st.session_state.audio_bytes = tts_res.content

        except Exception as e:
            st.error(f"LLM 응답 생성 오류: {e}")


# ----------------- 우측 컬럼: 결과 출력 패널 -----------------
with col2:
    st.subheader("📤 결과 및 대화 기록")

    # (출처: 6 - 미션 5, 6: 음성 질문과 응답)
    if st.session_state.audio_bytes:
        st.write("🔊 **에이블의 음성 답변:**")
        st.audio(st.session_state.audio_bytes, format="audio/mp3")

    # (출처: 2 - 5. 리스트 상태 - 히스토리 순회 출력)
    if st.session_state.messages:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
    else:
        st.info("명령을 입력하면 분석 결과와 답변이 이곳에 표시됩니다.")
