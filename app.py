import os
import io
import re
import json
import textwrap
import tempfile
import datetime

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter

# =====================================================================
# 0. 기본 설정
# =====================================================================
st.set_page_config(page_title="AIVLE ROADMAP AI", page_icon="🤖", layout="wide")

# Streamlit Cloud: Secrets에 OPENAI_API_KEY 등록 시 환경변수로 연결
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

CHROMA_DIR = "chroma_db"          # Colab에서 만든 폴더와 같아야 함
COLLECTION = "aivle_docs"         # Colab에서 만든 컬렉션과 같아야 함
LLM_MODEL = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-small"

# 사용자 구분별 단계(수준) 이름
LEVELS = {
    "지원자": ["탐색 단계", "지원 준비", "합격 후"],
    "교육생": ["초급", "중급", "고급"],
}

# 사용자 구분별 역할(페르소나)
PERSONA = {
    "지원자": (
        "당신은 KT 에이블스쿨 지원을 고민하는 예비 지원자를 돕는 'AIVLE GUIDE'입니다. "
        "지원자의 불안과 궁금증을 공감하며 풀어 주고, 백서에 근거한 정확한 정보로 "
        "'나도 도전해 볼 수 있겠다'는 확신을 주는 따뜻한 선배처럼 답합니다."
    ),
    "교육생": (
        "당신은 KT 에이블스쿨 교육생의 학습·생활을 돕는 'AIVLE GUIDE' 학습 코치입니다. "
        "교육생의 수준에 맞춰 개념을 쉽게 풀고, 지금 바로 실행할 수 있는 학습 행동을 제안하며, "
        "끝까지 완주할 수 있도록 응원하는 든든한 선배처럼 답합니다."
    ),
}

# 답변 길이 조절
LENGTH = {
    "기본": "핵심 결론을 먼저 말한 뒤 필요한 만큼 친절하게 설명하세요(5~10문장 내외, 필요 시 목록 사용).",
    "짧게": "3문장 이내로 핵심만 간결하게 답하세요.",
    "자세히": "단계별로 자세히 설명하고, 예시와 확인 방법, 주의사항까지 충분히 포함하세요.",
}

# 시스템 메시지 (미션 4: 역할 지정 + 문서 기반 + 기수·규정 구분 + 도전 독려)
SYSTEM_TEMPLATE = """{persona}

[사용자 정보]
- 사용자 구분/수준: {level}

[답변 원칙]
1. 반드시 아래 [문서]에 근거해 답합니다. 문서에 없는 내용은 추측하지 말고
   "백서에서 확인되지 않는 내용입니다"라고 밝힌 뒤, 확인할 수 있는 공식 경로(AIVLE-EDU, 담당 매니저, 공식 공고 등)를 안내합니다.
2. 기수·규정 구분: 모집 일정, 지원 자격, 출결, 공가, 훈련장려금, 자부담금 등 운영 규정은
   어느 기수 기준인지(예: "10기 기준") 반드시 밝히고, 다음 기수나 개인 판정에는 다를 수 있으며
   최신 공지·담당자 안내가 우선한다고 덧붙입니다.
3. 정보의 성격을 구분합니다: 「공식 운영 기준」, 「수료생 경험」, 「실천 제안」.
4. 근거가 된 문서 위치를 문장 끝에 (p.N) 형태로 표시합니다.
5. 사용자의 수준에 맞는 용어와 깊이로 설명합니다.
6. 마지막 줄에는 사용자가 다음에 해볼 수 있는 한 가지 행동이나 짧은 응원 한마디를 덧붙입니다.
7. 답변 길이: {length_guide}

[문서]
{context}"""

# 기수·규정 관련 키워드 (답변 하단 주의 배지 표시용)
RULE_KEYWORDS = ["출결", "출석", "지각", "조퇴", "결석", "외출", "공가", "장려금", "모집", "지원 자격",
                 "자격", "일정", "수료", "제적", "자부담", "내일배움", "합격", "선발", "질병"]

# 추천 질문 (사용자 구분 × 수준)
RECOMMENDED = {
    ("지원자", "탐색 단계"): ["에이블스쿨은 어떤 교육과정인가요?",
                          "AI 개발자와 DX 컨설턴트 트랙은 어떻게 다른가요?",
                          "비전공자도 지원해서 잘 따라갈 수 있나요?"],
    ("지원자", "지원 준비"): ["지원서 에세이는 어떻게 준비하면 좋을까요?",
                          "국민내일배움카드는 꼭 필요한가요?",
                          "10기 지원 자격과 모집 일정은 어땠나요?"],
    ("지원자", "합격 후"): ["합격 후 입교 전에 무엇을 확인해야 하나요?",
                        "교육은 하루에 어떤 일정으로 진행되나요?",
                        "입교 전에 미리 공부해 두면 좋은 것은?"],
    ("교육생", "초급"): ["지각·조퇴·외출 기준은 어떻게 되나요?",
                      "비전공자는 복습을 어떻게 하면 좋을까요?",
                      "RAG에서 청킹과 임베딩이 뭔가요?"],
    ("교육생", "중급"): ["RAG 검색 설정을 바꾸면 결과가 어떻게 달라지나요?",
                      "미니프로젝트에서 팀워크는 어떻게 하면 좋을까요?",
                      "훈련장려금은 어떤 기준으로 지급되나요?"],
    ("교육생", "고급"): ["RAG 답변 품질은 어떻게 검증하나요?",
                      "빅프로젝트를 잘 수행한 선배들의 공통점은?",
                      "프로젝트를 포트폴리오로 어떻게 정리하나요?"],
}

SITUATIONS = ["출결·지각·조퇴", "공가·출석 인정", "훈련장려금", "실습·코드 오류",
              "프로젝트·팀워크", "AIVLE-EDU·접속 문제", "기타"]

INTERESTS = ["데이터 분석", "머신러닝", "딥러닝", "RAG·LLM", "클라우드",
             "프로젝트·협업", "취업·포트폴리오"]


# =====================================================================
# 1. RAG 파이프라인 (01번 노트북의 rag_pipeline 구조를 그대로 사용)
# =====================================================================
# 문서 포맷터
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# 페이지 번호 (PyMuPDF4LLMLoader의 page는 0부터 시작)
def page_of(doc):
    p = doc.metadata.get("page")
    return p + 1 if isinstance(p, int) else "?"


# 출처(페이지)를 붙인 문서 포맷터 — 답변에 (p.N) 표기를 유도
def format_docs_with_page(docs):
    return "\n\n".join(f"[p.{page_of(d)}]\n{d.page_content}" for d in docs)


# RAG 체인 구성 (input만 넣어도 동작, 앱에서는 context·history 등을 추가로 전달)
def build_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder("history", optional=True),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, streaming=True)

    def get_context(x):
        # 앱에서 미리 검색한 문서를 넘기면 그대로 사용 (출처 표시와 같은 문서 사용)
        if x.get("context"):
            return x["context"]
        return format_docs_with_page(retriever.invoke(x["input"]))

    rag_chain = (
        {
            "context": RunnableLambda(get_context),
            "input": itemgetter("input"),
            "history": RunnableLambda(lambda x: x.get("history", [])),
            "persona": RunnableLambda(lambda x: x.get("persona", PERSONA["교육생"])),
            "level": RunnableLambda(lambda x: x.get("level", "교육생 / 초급")),
            "length_guide": RunnableLambda(lambda x: x.get("length_guide", LENGTH["기본"])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# RAG 파이프라인 함수
def rag_pipeline(filepath, return_retriever=False):

    # PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""])
    chunks = splitter.split_documents(docs)

    # Embedding 모델
    embedding = OpenAIEmbeddings(model=EMB_MODEL)

    # 벡터저장소 만들기
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )

    # 기존 데이터 비우기
    vectorstore.reset_collection()

    # 문서 추가
    _ = vectorstore.add_documents(chunks)

    # 검색기 구성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM 선언 + 프롬프트 구성 + RAG 체인 구성
    rag_chain = build_chain(retriever)

    if return_retriever:
        return rag_chain, retriever, len(chunks)
    return rag_chain


# GitHub에 올린 chroma_db 불러오기 (PDF 업로드 없이 바로 사용)
@st.cache_resource(show_spinner="📚 백서 벡터 저장소를 불러오는 중...")
def load_existing():
    if not os.path.exists(CHROMA_DIR):
        return None
    embedding = OpenAIEmbeddings(model=EMB_MODEL)
    vectorstore = Chroma(collection_name=COLLECTION, embedding_function=embedding,
                         persist_directory=CHROMA_DIR)
    count = vectorstore._collection.count()
    if count == 0:
        return None
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return build_chain(retriever), retriever, count


# =====================================================================
# 2. 공통 LLM 도우미 함수
# =====================================================================
def retrieve(query):
    return st.session_state.retriever.invoke(query)


def retrieve_many(queries):
    """여러 질의로 검색해 중복 없이 합치기"""
    seen, docs = set(), []
    for q in queries:
        for d in retrieve(q):
            if d.page_content not in seen:
                seen.add(d.page_content)
                docs.append(d)
    return docs


def level_text():
    return f"{st.session_state.user_type} / {st.session_state.level}"


def task_system(task):
    """기능별 시스템 메시지 = 페르소나 + 기능 지시 + 공통 원칙"""
    return (f"{PERSONA[st.session_state.user_type]}\n\n[사용자 수준] {level_text()}\n\n[작업]\n{task}\n\n"
            "[공통 원칙] 제공된 [문서]에 근거하고, 문서에 없는 내용은 지어내지 않습니다. "
            "운영 규정은 기수 기준(예: 10기)을 밝히고 최신 공지·담당자 확인이 우선임을 안내합니다. "
            "근거 위치는 (p.N)으로 표기합니다.")


def llm_stream(system, user):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, streaming=True)
    return (llm | StrOutputParser()).stream([("system", system), ("human", user)])


def llm_json(system, user, temperature=0.4):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=temperature).bind(
        response_format={"type": "json_object"})
    content = llm.invoke([("system", system + "\n반드시 JSON 객체로만 출력합니다."),
                          ("human", user)]).content
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        return json.loads(m.group()) if m else {}


def is_rule_related(text):
    return any(k in text for k in RULE_KEYWORDS)


def show_sources(docs, key):
    if not docs:
        return
    pages = sorted({page_of(d) for d in docs}, key=lambda x: (isinstance(x, str), x))
    with st.expander(f"📄 답변 근거 확인 · 백서 p.{', p.'.join(map(str, pages))}"):
        for d in docs:
            st.markdown(f"**p.{page_of(d)}**")
            st.caption(d.page_content[:350].replace("\n", " ") + " …")


def rule_badge():
    st.caption("📌 운영 규정 관련 답변입니다. 백서는 **10기 기준** 자료이므로, 다음 기수·개인 판정은 "
               "AIVLE-EDU 공지와 담당 매니저 안내가 우선합니다.")


# =====================================================================
# 3. 로드맵 PNG 출력 (기능 13)
# =====================================================================
def find_korean_font():
    from matplotlib import font_manager
    for path in font_manager.findSystemFonts():
        name = os.path.basename(path).lower()
        if any(k in name for k in ["nanumgothic", "nanumbarungothic", "notosanscjk", "notosanskr",
                                    "malgun", "applegothic"]):
            return path
    return None


def roadmap_png(rm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch

    font_path = find_korean_font()
    fp = FontProperties(fname=font_path) if font_path else FontProperties()

    W = 10
    blocks = []
    for w in rm.get("weeks", []):
        lines = []
        for it in w.get("items", []):
            text = f"• {it.get('topic', '')} — {it.get('task', '')} ({it.get('hours', '?')}h)"
            lines += textwrap.wrap(text, 60) or [""]
        blocks.append((w, lines))

    heights = [0.9 + 0.32 * len(lines) for _, lines in blocks]
    H = 1.6 + sum(h + 0.25 for h in heights) + 0.4
    fig = plt.figure(figsize=(W, H), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(0.5, H - 0.6, rm.get("title", "나의 맞춤형 학습 로드맵"), fontproperties=fp,
            fontsize=18, weight="bold", color="#1F2937", va="center")
    ax.text(0.5, H - 1.05, f"AIVLE GUIDE · {level_text()} · AI가 생성한 개인 학습 제안 (공식 일정 아님)",
            fontproperties=fp, fontsize=9, color="#6B7280", va="center")

    y = H - 1.6
    colors = ["#E60012", "#F97316", "#2563EB", "#7C3AED", "#059669", "#DB2777", "#0891B2", "#4B5563"]
    for i, ((w, lines), h) in enumerate(zip(blocks, heights)):
        c = colors[i % len(colors)]
        ax.add_patch(FancyBboxPatch((0.4, y - h), W - 0.8, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    linewidth=1.5, edgecolor=c, facecolor="#F9FAFB"))
        ax.add_patch(FancyBboxPatch((0.4, y - 0.55), 1.3, 0.45, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    linewidth=0, facecolor=c))
        ax.text(1.05, y - 0.33, f"{w.get('week', i + 1)}주차", fontproperties=fp, fontsize=11,
                color="white", ha="center", va="center", weight="bold")
        ax.text(1.9, y - 0.33, w.get("goal", ""), fontproperties=fp, fontsize=11.5,
                color="#111827", va="center", weight="bold")
        for j, line in enumerate(lines):
            ax.text(0.7, y - 0.85 - 0.32 * j, line, fontproperties=fp, fontsize=9.5,
                    color="#374151", va="center")
        y -= h + 0.25

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue(), font_path is not None


def roadmap_markdown(rm):
    md = [f"# {rm.get('title', '나의 학습 로드맵')}", "", rm.get("summary", ""), ""]
    for w in rm.get("weeks", []):
        md.append(f"## {w.get('week')}주차 · {w.get('goal', '')}")
        for it in w.get("items", []):
            md.append(f"- **{it.get('topic')}**: {it.get('task')} ({it.get('hours')}h)")
        md.append("")
    if rm.get("tips"):
        md.append("## 선배 팁")
        md += [f"- {t}" for t in rm["tips"]]
    return "\n".join(md)


# =====================================================================
# 3-1. ★ 핵심 차별점: 내 학습 자료 기반 · 수준 진단 맞춤형 학습 도우미
#   교안·실습 파일 업로드 → 질문 수준 진단 → 학습 위치 추정 → 수준별 맞춤 설명
# =====================================================================
MATERIAL_TYPES = ["pdf", "ipynb", "py", "txt", "md", "csv"]
FILE_ICON = {"pdf": "📕", "ipynb": "📓", "py": "🐍", "txt": "📄", "md": "📝", "csv": "📊"}
LV_NUM = {"초급": 1, "중급": 2, "고급": 3}
NUM_LV = {1: "초급", 2: "중급", 3: "고급"}
LV_COLOR = {"초급": "#45C4AE", "중급": "#F2A93B", "고급": "#7C5CDB"}

# 학습 위치(단계) — 질문 내용으로 교육생이 어디쯤 있는지 판단하는 기준
STAGES = ["① 개념 이해", "② 코드 실행·따라하기", "③ 오류 해결", "④ 코드 수정·응용", "⑤ 평가·최적화·설계"]

# 수준별 설명 전략
LEVEL_STRATEGY = {
    "초급": ("- 전문 용어는 처음 나올 때 쉬운 말로 풀고, 일상 비유를 하나 듭니다.\n"
             "- 코드는 한 줄씩 무엇을 하는지 주석처럼 설명합니다.\n"
             "- 한 번에 개념 1~2개만 다루고, 선행 개념이 부족하면 먼저 짚어 줍니다."),
    "중급": ("- 동작 원리와 '왜 그렇게 하는지'를 설명합니다.\n"
             "- 자료의 코드를 어떻게 바꿔 볼 수 있는지 수정 예시와 비교 포인트를 제시합니다.\n"
             "- 자주 하는 실수와 확인 방법을 함께 알려 줍니다."),
    "고급": ("- 내부 원리, 설계 선택지의 장단점(트레이드오프), 성능·품질 관점을 다룹니다.\n"
             "- 실험·검증 방법(지표, 비교 설계)과 실무·프로젝트 적용 방안을 제시합니다.\n"
             "- 기초 설명은 생략하고 핵심만 밀도 있게 다룹니다."),
}

DIAG_SYSTEM = """당신은 KT 에이블스쿨 교육생의 질문을 분석해 학습 수준과 학습 위치를 진단하는 교육 전문가입니다.
판단 기준:
- 초급: 용어·개념 자체를 묻거나("~가 뭐예요?"), 코드를 그대로 실행하다 막힘, 기초 문법 오류
- 중급: 원리·이유를 묻거나, 코드 수정·설정 변경·비교를 시도, 오류 원인을 스스로 일부 파악
- 고급: 성능·품질 평가, 설계 대안 비교, 최적화·확장·배포, 실무 적용을 고민
학습 위치(stage)는 다음 중 하나: """ + ", ".join(STAGES) + """
질문에 쓰인 용어의 정확도, 시도한 내용, 질문의 구체성, 이전 대화 흐름을 근거로 판단합니다.
반드시 JSON 객체로만 출력합니다."""


def _doc(text, file, loc):
    from langchain_core.documents import Document
    return Document(page_content=text, metadata={"source_file": file, "loc": loc})


def load_material(uploaded):
    """업로드 파일 → Document 목록 (파일 형식별 처리)"""
    name = uploaded.name
    ext = name.rsplit(".", 1)[-1].lower()
    data = uploaded.getvalue()
    docs = []
    if ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(data)
            tmp = f.name
        try:
            for d in PyMuPDF4LLMLoader(tmp).load():
                docs.append(_doc(d.page_content, name, f"p.{page_of(d)}"))
        finally:
            os.remove(tmp)
    elif ext == "ipynb":
        nb = json.loads(data.decode("utf-8", errors="ignore"))
        for i, cell in enumerate(nb.get("cells", []), start=1):
            src = "".join(cell.get("source", []))
            if not src.strip():
                continue
            kind = "코드" if cell.get("cell_type") == "code" else "설명"
            text = f"[{kind} 셀]\n{src}"
            # 실행 결과·오류 메시지도 함께 저장 (오류 질문 대응)
            outs = []
            for o in cell.get("outputs", []):
                if o.get("output_type") == "error":
                    outs.append(f"{o.get('ename')}: {o.get('evalue')}")
                elif "text" in o:
                    outs.append("".join(o["text"])[:500])
            if outs:
                text += "\n[실행 결과]\n" + "\n".join(outs)
            docs.append(_doc(text, name, f"셀 {i}"))
    else:  # py, txt, md, csv
        text = data.decode("utf-8", errors="ignore")
        if ext == "csv":
            text = "\n".join(text.splitlines()[:60])   # CSV는 구조 파악용 앞부분만
        docs.append(_doc(text, name, "본문"))
    return docs


@st.cache_resource
def get_embedding():
    return OpenAIEmbeddings(model=EMB_MODEL)


def build_material_store(files):
    """학습 자료 전용 벡터 저장소 (세션별 메모리 저장 → 공용 백서·다른 사용자와 섞이지 않음)"""
    import uuid
    docs = []
    for f in files:
        docs += load_material(f)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120,
        separators=["\n\n", "\ndef ", "\nclass ", "\n", ".", " ", ""])
    chunks = splitter.split_documents(docs)
    store = Chroma(collection_name=f"my_materials_{uuid.uuid4().hex[:8]}",
                   embedding_function=get_embedding())
    if chunks:
        store.add_documents(chunks)
    return store, chunks


def analyze_materials(chunks):
    """등록한 자료의 학습 주제와 수준별 추천 질문 생성"""
    sample = "\n\n".join(c.page_content[:400] for c in chunks[:: max(1, len(chunks) // 12)][:12])
    return llm_json(
        "당신은 교안·실습 자료를 분석해 학습 주제를 정리하고 질문을 추천하는 AI 튜터입니다.",
        f"[자료 일부]\n{sample}\n\n"
        '다음 JSON으로 출력: {"summary": "자료 한 줄 요약", "topics": ["핵심 주제 5개 이내"], '
        '"questions": {"초급": ["질문"], "중급": ["질문"], "고급": ["질문"]}} '
        "수준별 질문은 각각 2개, 자료로 답할 수 있고 30자 이내.")


def diagnose_level(question, history_text):
    res = llm_json(DIAG_SYSTEM,
                   f"[이전 대화 요약]\n{history_text or '없음'}\n\n[이번 질문]\n{question}\n\n"
                   '{"level": "초급|중급|고급", "stage": "학습 위치", "intent": "질문 의도(15자 이내)", '
                   '"evidence": "판단 근거 1문장", "prerequisite": "먼저 알면 좋은 선행 개념(없으면 빈 문자열)"}',
                   temperature=0)
    if res.get("level") not in LV_NUM:
        res["level"] = "초급"
    if res.get("stage") not in STAGES:   # "개념 이해"처럼 번호 없이 답한 경우 보정
        res["stage"] = next((s for s in STAGES if str(res.get("stage", "")).strip("①②③④⑤ ") in s
                             and str(res.get("stage", "")).strip()), STAGES[0])
    return res


def learner_level():
    """최근 진단을 가중 평균해 교육생의 현재 수준 추정 (최근 질문일수록 가중치↑)"""
    hist = st.session_state.diag_history[-6:]
    if not hist:
        return None
    w = list(range(1, len(hist) + 1))
    score = sum(LV_NUM[h["level"]] * wi for h, wi in zip(hist, w)) / sum(w)
    return NUM_LV[min(3, max(1, round(score)))], score


def material_system(level, diag, use_whitepaper):
    return f"""{PERSONA["교육생"]}

당신은 교육생이 직접 올린 교안·실습 파일을 근거로 설명하는 맞춤형 튜터입니다.

[학습 수준 진단 결과]
- 이번 질문의 수준: {diag.get('level')} / 학습 위치: {diag.get('stage')} / 질문 의도: {diag.get('intent')}
- 먼저 알면 좋은 선행 개념: {diag.get('prerequisite') or '없음'}
- 이번 답변을 맞출 수준: {level}

[{level} 설명 전략]
{LEVEL_STRATEGY[level]}

[답변 형식]
1. 첫 줄: 질문의 핵심에 대한 한두 문장 결론
2. 본문: 위 설명 전략에 맞춘 설명 (자료의 코드·문장을 인용할 때는 (파일명 · 위치) 표기)
3. "✅ 이해 확인" — 스스로 답해 볼 짧은 확인 질문 1개
4. "➡️ 다음 단계" — 지금 학습 위치에서 다음으로 해볼 것 1가지

[원칙]
- [학습 자료]에 근거해 설명합니다. 자료에 없는 일반 지식으로 보충할 때는 "(일반 지식)"이라고 표시합니다.
- 선행 개념이 부족해 보이면 본론 전에 짧게 짚어 줍니다.
- {"[백서]는 학습 전략·운영 안내가 필요할 때만 참고합니다." if use_whitepaper else ""}
- 답변 길이: {LENGTH['기본']}"""


def format_materials(docs):
    return "\n\n".join(f"[{d.metadata.get('source_file')} · {d.metadata.get('loc')}]\n{d.page_content}"
                       for d in docs)


def diag_card(diag, applied_level):
    lv = diag.get("level", "초급")
    stage_idx = STAGES.index(diag["stage"]) if diag.get("stage") in STAGES else 0
    steps = "".join(
        f'<span class="stp {"on" if i <= stage_idx else ""}">{s}</span>' for i, s in enumerate(STAGES))
    pre = (f'<div class="dg-pre">📌 선행 개념: {diag["prerequisite"]}</div>'
           if diag.get("prerequisite") else "")
    return svg_html(f"""
    <div class="diag">
      <div class="dg-top"><span class="dg-title">🔍 질문 수준 진단</span>
        <span class="lv" style="background:{LV_COLOR[lv]}">{lv}</span>
        <span class="dg-intent">{diag.get('intent', '')}</span>
        <span class="dg-applied">→ <b>{applied_level}</b> 눈높이로 설명</span></div>
      <div class="dg-steps">{steps}</div>
      <div class="dg-ev">💡 {diag.get('evidence', '')}</div>{pre}
    </div>""")


def show_material_sources(docs):
    if not docs:
        return
    with st.expander("📎 참고한 내 학습 자료"):
        for d in docs:
            src = d.metadata.get("source_file")
            if src:
                st.markdown(f"**{FILE_ICON.get(src.rsplit('.', 1)[-1].lower(), '📄')} {src} · {d.metadata.get('loc')}**")
            else:
                st.markdown(f"**📘 백서 · p.{page_of(d)}**")
            st.caption(d.page_content[:300].replace("\n", " ") + " …")


# =====================================================================
# 4. 세션 상태 초기화
# =====================================================================
DEFAULTS = {
    "user_type": None, "level": None, "messages": [], "pending": None,
    "roadmap": None, "roadmap_docs": [], "done": {},
    "topic_output": None, "quiz": None, "quiz_result": None,
    "solution": None, "solution_docs": [], "inquiry": None,
    "chain": None, "retriever": None, "doc_name": None, "chunk_count": 0,
    # ★ 내 학습 자료 · 수준 진단
    "mat_store": None, "mat_files": [], "mat_info": {}, "mat_chunks": 0, "mat_messages": [],
    "mat_pending": None, "mat_uploader_id": 0, "diag_history": [], "auto_level": True,
    "use_whitepaper": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 벡터 저장소 준비
if st.session_state.chain is None:
    try:
        loaded = load_existing()
    except Exception as e:
        loaded = None
        st.session_state.load_error = str(e)
    if loaded:
        st.session_state.chain, st.session_state.retriever, st.session_state.chunk_count = loaded
        st.session_state.doc_name = "AIVLE School 백서 (기본)"


# =====================================================================
# 5. 스타일 · 일러스트 (ROADMAP AI 테마: 베이지 + 민트 + 네이비)
# =====================================================================
MINT, MINT_D, NAVY, BEIGE, CREAM, SAND = "#45C4AE", "#2FA893", "#1F2A44", "#F3EEE6", "#FBF8F3", "#E4D8C6"

st.markdown(f"""
<style>
.stApp {{background: {BEIGE};}}
header[data-testid="stHeader"] {{background: transparent;}}
.block-container {{padding-top: 1.6rem; max-width: 1150px; position: relative; z-index: 1;}}
section[data-testid="stSidebar"] {{background: {CREAM}; border-right: 1px solid {SAND};}}

/* 배경 장식 원 */
.deco {{position: fixed; border-radius: 50%; background: {SAND}; opacity: .75; z-index: 0; pointer-events: none;}}
.deco.c1 {{width: 320px; height: 320px; right: -110px; top: -120px;}}
.deco.c2 {{width: 70px; height: 70px; right: 230px; top: 60px; opacity: .6;}}
.deco.c3 {{width: 90px; height: 46px; right: -20px; top: 260px; border-radius: 30px; opacity: .6;}}
.deco.c4 {{width: 300px; height: 150px; left: -80px; bottom: -40px; border-radius: 0 150px 0 0; opacity: .6;}}
.deco.c5 {{width: 180px; height: 70px; left: 90px; bottom: 40px; border-radius: 40px; opacity: .45;
           transform: rotate(-35deg);}}

/* 로고 */
.logo {{display: flex; align-items: center; gap: 8px; font-weight: 800; color: {NAVY};}}
.logo .name {{font-size: 1.45rem; letter-spacing: .5px;}}
.logo .sub {{font-size: .85rem; font-weight: 700; margin-left: 4px; letter-spacing: .3px;}}
.logo .sub b {{color: {MINT_D};}}

/* 시작 화면 */
.welcome {{text-align: center; margin-top: 4px;}}
.welcome h1 {{color: {NAVY}; font-size: 2rem; font-weight: 800; margin: 0; padding: 0;}}
.welcome p {{color: #4B5563; font-size: 1.05rem; margin: 6px 0 0 0;}}
.robots {{display: flex; justify-content: center; align-items: flex-end; gap: 4px; margin: 14px 0 6px;}}
.robots svg {{animation: float 3.2s ease-in-out infinite;}}
.robots svg:nth-child(2) {{animation-delay: .4s;}}
.robots svg:nth-child(3) {{animation-delay: .8s;}}
@keyframes float {{0%,100% {{transform: translateY(0);}} 50% {{transform: translateY(-8px);}}}}

/* 역할 카드 (st.container(key=...) 에 스타일 적용) */
.st-key-card_applicant, .st-key-card_student {{
    background: {CREAM}; border: 1px solid {SAND}; border-radius: 20px;
    padding: 22px 22px 18px; box-shadow: 0 8px 22px rgba(80,60,30,.10);
    transition: transform .18s ease, box-shadow .18s ease; position: relative;}}
.st-key-card_applicant:hover, .st-key-card_student:hover {{
    transform: translateY(-5px); box-shadow: 0 14px 30px rgba(80,60,30,.16);}}
.role {{text-align: center; position: relative;}}
.role .badge {{position: absolute; right: 0; top: -6px; width: 34px; height: 34px; border-radius: 50%;
               background: #EFE6D8; display: flex; align-items: center; justify-content: center; font-size: 17px;}}
.role h3 {{color: {NAVY}; font-size: 1.55rem; font-weight: 800; margin: 10px 0 2px; padding: 0;}}
.role p {{color: #4B5563; margin: 0 0 4px;}}
.role .tags {{margin-top: 8px;}}
.chip {{display: inline-block; padding: 3px 10px; border-radius: 999px; background: #DDF3EE;
        color: {MINT_D}; font-size: .78rem; margin: 2px 3px; font-weight: 700;}}
.st-key-card_applicant button, .st-key-card_student button {{
    background: {MINT}; color: white; border: none; border-radius: 999px; font-weight: 700; padding: .55rem 0;}}
.st-key-card_applicant button:hover, .st-key-card_student button:hover {{background: {MINT_D}; color: white;}}

/* 메인 화면 헤더 */
.hero {{background: {CREAM}; border: 1px solid {SAND}; border-radius: 20px; padding: 18px 26px;
        margin-bottom: 16px; display: flex; align-items: center; gap: 18px;
        box-shadow: 0 6px 18px rgba(80,60,30,.08);}}
.hero h2 {{color: {NAVY}; margin: 0; font-size: 1.5rem; font-weight: 800; padding: 0;}}
.hero p {{color: #4B5563; margin: 4px 0 0; font-size: .95rem;}}
.notice {{background: #FFF7ED; border-left: 4px solid #F59E0B; padding: 10px 14px;
          border-radius: 8px; font-size: .9rem; color: #78350F;}}
div[data-testid="stTabs"] button p {{font-size: 1.05rem; font-weight: 700;}}

/* ★ 내 자료 학습: 흐름 안내 */
.flow {{display: flex; align-items: stretch; gap: 8px; margin: 4px 0 16px; flex-wrap: wrap;}}
.fl {{flex: 1; min-width: 170px; display: flex; gap: 10px; align-items: center; background: {CREAM};
      border: 1px solid {SAND}; border-radius: 16px; padding: 12px 14px; font-size: .92rem; color: {NAVY};}}
.fl span {{color: #6B7280; font-size: .8rem;}}
.fl-n {{min-width: 30px; height: 30px; border-radius: 50%; background: {MINT}; color: white;
        display: flex; align-items: center; justify-content: center; font-weight: 800;}}
.fl-arrow {{display: flex; align-items: center; color: {MINT_D}; font-weight: 800;}}
.st-key-upload_card, .st-key-profile_card {{background: {CREAM}; border: 1px solid {SAND};
      border-radius: 18px; padding: 16px 18px; box-shadow: 0 6px 16px rgba(80,60,30,.07);}}
.st-key-upload_card {{border: 2px dashed {MINT};}}
.fchips {{display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 8px;}}
.fchip {{background: white; border: 1px solid {SAND}; border-radius: 10px; padding: 4px 10px;
         font-size: .85rem; color: {NAVY};}}
.fchip i {{color: #9CA3AF; font-style: normal; font-size: .75rem; margin-left: 4px;}}

/* ★ 수준 진단 카드 */
.diag {{background: #F1FBF8; border: 1px solid #BFEADF; border-radius: 14px; padding: 10px 14px; margin-bottom: 10px;}}
.dg-top {{display: flex; align-items: center; gap: 8px; flex-wrap: wrap;}}
.dg-title {{font-weight: 800; color: {NAVY}; font-size: .92rem;}}
.lv {{display: inline-block; color: white; font-weight: 800; font-size: .8rem; padding: 2px 10px; border-radius: 999px;}}
.dg-intent {{color: #374151; font-size: .85rem; background: white; border-radius: 8px; padding: 1px 8px;}}
.dg-applied {{margin-left: auto; font-size: .85rem; color: {MINT_D};}}
.dg-steps {{display: flex; gap: 4px; margin: 8px 0 6px; flex-wrap: wrap;}}
.stp {{font-size: .74rem; padding: 2px 8px; border-radius: 999px; background: #E5E7EB; color: #6B7280;}}
.stp.on {{background: {MINT}; color: white; font-weight: 700;}}
.dg-ev {{font-size: .84rem; color: #374151;}}
.dg-pre {{font-size: .84rem; color: #92400E; margin-top: 3px;}}

/* ★ 학습 프로필 */
.prof-lv {{display: flex; justify-content: space-between; align-items: baseline; color: #6B7280; font-size: .85rem;}}
.prof-lv b {{font-size: 1.9rem;}}
.bar {{height: 10px; background: #E9E2D6; border-radius: 999px; overflow: hidden; margin-top: 4px;}}
.bar div {{height: 100%; background: linear-gradient(90deg, {MINT}, #F2A93B, #7C5CDB); border-radius: 999px;}}
.bar-lb {{display: flex; justify-content: space-between; font-size: .72rem; color: #9CA3AF; margin-bottom: 6px;}}
.stage-row {{font-size: .82rem; color: {NAVY}; background: white; border: 1px solid {SAND};
             border-radius: 8px; padding: 4px 10px; margin: 4px 0;}}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {{
    background: {MINT}; border-color: {MINT}; color: white;}}
.stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover {{
    background: {MINT_D}; border-color: {MINT_D};}}
</style>
<div class="deco c1"></div><div class="deco c2"></div><div class="deco c3"></div>
<div class="deco c4"></div><div class="deco c5"></div>
""", unsafe_allow_html=True)


def logo_svg(size=34):
    """AIVLE 로고 마크 (민트색 A 형태)"""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 40 40">
      <path d="M6 34 L20 6 L34 34" fill="none" stroke="{MINT}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M14 26 L26 26" stroke="{MINT}" stroke-width="4" stroke-linecap="round"/>
      <circle cx="20" cy="18" r="2.6" fill="{NAVY}"/></svg>"""


def logo_html():
    return (f'<div class="logo">{logo_svg()}<span class="name">AIVLE</span>'
            f'<span class="sub">ROADMAP <b>AI</b></span></div>')


def robot_svg(kind="wave", size=170):
    """마스코트 로봇: wave(인사), center(로고), grad(학사모·헤드셋)"""
    O = NAVY
    arm_l = '<rect x="10" y="98" width="17" height="42" rx="8.5" transform="rotate(38 18 132)" fill="white" stroke="{O}" stroke-width="3.5"/>'
    arm_l_down = '<rect x="30" y="116" width="16" height="38" rx="8" transform="rotate(18 38 120)" fill="white" stroke="{O}" stroke-width="3.5"/>'
    arm_r = '<rect x="133" y="98" width="17" height="42" rx="8.5" transform="rotate(-38 142 132)" fill="white" stroke="{O}" stroke-width="3.5"/>'
    arm_r_down = '<rect x="114" y="116" width="16" height="38" rx="8" transform="rotate(-18 122 120)" fill="white" stroke="{O}" stroke-width="3.5"/>'
    arms = {"wave": arm_l + arm_r_down, "center": arm_l_down + arm_r, "grad": arm_l_down + arm_r}[kind].format(O=O)

    if kind == "center":
        belly = (f'<path d="M70 150 L80 128 L90 150" fill="none" stroke="{MINT}" stroke-width="4.5" '
                 f'stroke-linecap="round" stroke-linejoin="round"/><path d="M74 143 L86 143" stroke="{MINT}" '
                 f'stroke-width="3.5" stroke-linecap="round"/>')
    else:
        belly = (f'<circle cx="80" cy="140" r="12" fill="{MINT}"/><path d="M74 140 L78.5 144.5 L86.5 135.5" '
                 f'fill="none" stroke="white" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')

    extra = ""
    if kind == "wave":   # 반짝 효과선
        extra = (f'<g stroke="{O}" stroke-width="3" stroke-linecap="round"><path d="M14 52 L4 46"/>'
                 f'<path d="M18 40 L12 28"/><path d="M28 34 L28 22"/></g>')
    if kind == "grad":   # 학사모 + 헤드셋 마이크
        extra = (f'<path d="M80 4 L132 22 L80 40 L28 22 Z" fill="{NAVY}"/>'
                 f'<rect x="56" y="26" width="48" height="14" rx="4" fill="{NAVY}"/>'
                 f'<path d="M132 22 L132 44" stroke="{NAVY}" stroke-width="2.5"/><circle cx="132" cy="47" r="4" fill="#F2B544"/>'
                 f'<path d="M134 86 Q136 108 112 108" fill="none" stroke="{O}" stroke-width="3.5" stroke-linecap="round"/>'
                 f'<circle cx="110" cy="108" r="4.5" fill="{O}"/>')
    antenna = "" if kind == "grad" else (f'<line x1="80" y1="34" x2="80" y2="18" stroke="{O}" stroke-width="3.5"/>'
                                          f'<circle cx="80" cy="14" r="6.5" fill="{MINT}" stroke="{O}" stroke-width="3"/>')

    return f"""<svg width="{size}" height="{size * 1.2:.0f}" viewBox="0 0 160 200">
      <ellipse cx="80" cy="191" rx="42" ry="6" fill="{SAND}"/>
      {antenna}
      {arms}
      <rect x="58" y="160" width="17" height="24" rx="7" fill="white" stroke="{O}" stroke-width="3.5"/>
      <rect x="85" y="160" width="17" height="24" rx="7" fill="white" stroke="{O}" stroke-width="3.5"/>
      <rect x="46" y="110" width="68" height="58" rx="26" fill="white" stroke="{O}" stroke-width="3.5"/>
      {belly}
      <rect x="20" y="62" width="16" height="30" rx="8" fill="{MINT}" stroke="{O}" stroke-width="3"/>
      <rect x="124" y="62" width="16" height="30" rx="8" fill="{MINT}" stroke="{O}" stroke-width="3"/>
      <rect x="30" y="34" width="100" height="80" rx="38" fill="white" stroke="{O}" stroke-width="3.5"/>
      <path d="M44 52 Q80 30 116 52" fill="none" stroke="{MINT}" stroke-width="7" stroke-linecap="round" opacity=".55"/>
      <rect x="44" y="52" width="72" height="48" rx="23" fill="{NAVY}"/>
      <path d="M58 76 Q65 66 72 76" fill="none" stroke="{MINT}" stroke-width="4.5" stroke-linecap="round"/>
      <path d="M88 76 Q95 66 102 76" fill="none" stroke="{MINT}" stroke-width="4.5" stroke-linecap="round"/>
      <path d="M72 84 Q80 93 88 84" fill="{MINT}" stroke="{MINT}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      {extra}
    </svg>"""


def applicant_svg(size=120):
    """지원자: 정장 + 서류 파일"""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 120 120">
      <defs><clipPath id="cpA"><circle cx="60" cy="60" r="56"/></clipPath></defs>
      <circle cx="60" cy="60" r="56" fill="#E9DFD0"/>
      <g clip-path="url(#cpA)">
        <path d="M18 120 Q20 82 60 80 Q100 82 102 120 Z" fill="#3B4A6B"/>
        <path d="M50 80 L60 100 L70 80 Z" fill="white"/>
        <path d="M57 84 L63 84 L62 104 L60 108 L58 104 Z" fill="{MINT}"/>
        <rect x="53" y="66" width="14" height="16" rx="5" fill="#F4C9A3"/>
        <ellipse cx="60" cy="52" rx="19" ry="22" fill="#F8D5B5"/>
        <path d="M40 50 Q38 26 60 27 Q84 26 80 50 Q76 38 62 38 Q48 38 40 50 Z" fill="#3A2A20"/>
        <circle cx="53" cy="54" r="2.2" fill="{NAVY}"/><circle cx="67" cy="54" r="2.2" fill="{NAVY}"/>
        <path d="M54 63 Q60 68 66 63" fill="none" stroke="#B5684A" stroke-width="2" stroke-linecap="round"/>
        <rect x="70" y="86" width="30" height="24" rx="3" fill="#F2B544" transform="rotate(-12 85 98)"/>
        <rect x="68" y="90" width="30" height="24" rx="3" fill="#F7CC6A" transform="rotate(-6 83 102)"/>
        <ellipse cx="74" cy="104" rx="7" ry="5" fill="#F8D5B5"/>
      </g></svg>"""


def student_svg(size=120):
    """교육생: 후드티 + 노트북"""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 120 120">
      <defs><clipPath id="cpS"><circle cx="60" cy="60" r="56"/></clipPath></defs>
      <circle cx="60" cy="60" r="56" fill="#E9DFD0"/>
      <g clip-path="url(#cpS)">
        <path d="M36 58 Q34 28 60 28 Q86 28 84 58 L86 80 L34 80 Z" fill="#5A3A28"/>
        <path d="M16 120 Q20 82 60 80 Q100 82 104 120 Z" fill="{MINT}"/>
        <path d="M50 80 Q60 90 70 80" fill="none" stroke="{MINT_D}" stroke-width="3"/>
        <rect x="53" y="66" width="14" height="16" rx="5" fill="#F4C9A3"/>
        <ellipse cx="60" cy="54" rx="18" ry="21" fill="#F8D5B5"/>
        <path d="M42 50 Q44 32 60 32 Q78 32 78 50 Q68 42 58 40 Q50 44 42 50 Z" fill="#5A3A28"/>
        <circle cx="53" cy="56" r="2.2" fill="{NAVY}"/><circle cx="67" cy="56" r="2.2" fill="{NAVY}"/>
        <path d="M54 64 Q60 69 66 64" fill="none" stroke="#B5684A" stroke-width="2" stroke-linecap="round"/>
        <path d="M24 96 L64 96 L70 118 L18 118 Z" fill="#B8C2CC"/>
        <rect x="20" y="88" width="46" height="30" rx="3" fill="#CBD5DF" transform="skewX(-10)"/>
        <circle cx="40" cy="103" r="3" fill="white" opacity=".8"/>
      </g></svg>"""


def svg_html(svg):
    """st.markdown 안에서 SVG가 들여쓰기 때문에 코드블록으로 오인되지 않도록 한 줄로 정리"""
    return re.sub(r">\s+<", "><", svg.strip()).replace("\n", " ")


# =====================================================================
# 6. 시작 화면: 지원자 / 교육생 구분 (기능 4)
# =====================================================================
if st.session_state.user_type is None:
    st.markdown(svg_html(logo_html()), unsafe_allow_html=True)
    st.markdown(svg_html(f"""
    <div class="welcome">
      <h1>에이블스쿨 ROADMAP AI에 오신 것을 환영합니다!</h1>
      <p>해당하는 역할을 선택해 주세요.</p>
      <div class="robots">{robot_svg("wave")}{robot_svg("center")}{robot_svg("grad")}</div>
    </div>"""), unsafe_allow_html=True)

    _, c1, c2, _ = st.columns([1, 2.2, 2.2, 1], gap="medium")
    with c1:
        with st.container(key="card_applicant"):
            st.markdown(svg_html(f"""
            <div class="role"><div class="badge">📋</div>{applicant_svg()}
              <h3>지원자이신가요?</h3>
              <p>지원 자격·일정·트랙을 함께 알아봐요</p>
              <div class="tags"><span class="chip">지원 자격</span><span class="chip">모집 일정</span>
              <span class="chip">트랙 선택</span></div></div>"""), unsafe_allow_html=True)
            if st.button("지원자로 시작하기 →", key="start_applicant", width="stretch"):
                st.session_state.user_type, st.session_state.level = "지원자", LEVELS["지원자"][0]
                st.rerun()
    with c2:
        with st.container(key="card_student"):
            st.markdown(svg_html(f"""
            <div class="role"><div class="badge">🎓</div>{student_svg()}
              <h3>교육생이신가요?</h3>
              <p>학습과 프로젝트를 함께 완주해요</p>
              <div class="tags"><span class="chip">학습 로드맵</span><span class="chip">퀴즈</span>
              <span class="chip">문제 해결</span></div></div>"""), unsafe_allow_html=True)
            if st.button("교육생으로 시작하기 →", key="start_student", width="stretch"):
                st.session_state.user_type, st.session_state.level = "교육생", LEVELS["교육생"][0]
                st.rerun()
    st.stop()


# =====================================================================
# 7. 사이드바: 공통 설정 (기능 1, 4, 6)
# =====================================================================
with st.sidebar:
    st.markdown("## ⚙️ 설정")

    # 사용자 구분 (실시간 변경)
    ut = st.radio("사용자 구분", ["지원자", "교육생"], horizontal=True,
                  index=["지원자", "교육생"].index(st.session_state.user_type))
    if ut != st.session_state.user_type:
        st.session_state.user_type, st.session_state.level = ut, LEVELS[ut][0]
        st.rerun()

    # 수준 (실시간 변경)
    st.session_state.level = st.select_slider(
        "나의 단계" if ut == "지원자" else "나의 학습 수준",
        options=LEVELS[ut], value=st.session_state.level)

    st.divider()

    # ★ 내 학습 자료 현황 (핵심 기능은 '📂 내 자료로 학습' 탭)
    st.markdown("#### 📂 내 학습 자료")
    if st.session_state.mat_files:
        for n, c in st.session_state.mat_files:
            st.caption(f"{FILE_ICON.get(n.rsplit('.', 1)[-1].lower(), '📄')} {n} · {c}조각")
    else:
        st.caption("'📂 내 자료로 학습' 탭에서 교안·실습 파일을 올려 보세요.")
    est = learner_level()
    if est:
        st.caption(f"🔍 AI 추정 수준: **{est[0]}** (질문 {len(st.session_state.diag_history)}개 기준)")

    st.divider()

    # 기준 문서(백서) 업로드·교체 (기능 1)
    st.markdown("#### 📘 기준 문서 (백서)")
    st.caption(f"현재: **{st.session_state.doc_name or '없음'}**"
               + (f" · {st.session_state.chunk_count}개 청크" if st.session_state.chunk_count else ""))
    with st.expander("백서 업데이트·교체"):
        uploaded = st.file_uploader("새 백서 PDF", type=["pdf"])
        if uploaded and st.button("📥 이 문서로 RAG 다시 구성", width="stretch"):
            with st.spinner("문서를 읽고 벡터 저장소를 다시 만드는 중..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded.getvalue())
                    tmp_path = f.name
                load_existing.clear()
                chain, retriever, n = rag_pipeline(tmp_path, return_retriever=True)
                os.remove(tmp_path)
            st.session_state.update(chain=chain, retriever=retriever, chunk_count=n,
                                    doc_name=uploaded.name, messages=[])
            st.success(f"반영 완료! {n}개 청크로 구성했어요.")
            st.rerun()

    st.divider()

    # 대화 관리 (기능 6)
    st.markdown("#### 💬 대화 관리")
    if st.session_state.messages:
        log = "\n\n".join(f"[{'나' if m['role'] == 'user' else 'AIVLE GUIDE'}]\n{m['content']}"
                          for m in st.session_state.messages)
        st.download_button("💾 대화 내용 저장(.txt)", log,
                           file_name=f"aivle_guide_{datetime.date.today()}.txt", width="stretch")
    if st.button("🗑️ 대화 초기화", width="stretch"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()
    if st.button("🔄 처음 화면으로", width="stretch"):
        for k, v in DEFAULTS.items():
            if k not in ("chain", "retriever", "doc_name", "chunk_count"):
                st.session_state[k] = v
        st.rerun()

    st.divider()
    st.markdown('<div class="notice">📌 백서의 운영 규정은 <b>10기 기준</b>입니다. '
                '최신 공지와 담당자 안내가 항상 우선합니다.</div>', unsafe_allow_html=True)


# =====================================================================
# 8. 헤더
# =====================================================================
emoji = "🌱" if st.session_state.user_type == "지원자" else "🚀"
robot = robot_svg("wave" if st.session_state.user_type == "지원자" else "grad", size=78)
st.markdown(svg_html(logo_html()), unsafe_allow_html=True)
st.markdown(svg_html(f"""
<div class="hero">{robot}
  <div><h2>{emoji} {level_text()} 모드</h2>
  <p>교안·실습 파일을 올리고 질문하면, AI가 <b>질문 수준으로 나의 학습 위치를 진단</b>해 눈높이에 맞게 설명해요.</p></div>
</div>"""), unsafe_allow_html=True)

if st.session_state.chain is None:
    st.warning("벡터 저장소(chroma_db)를 찾지 못했어요. 왼쪽 사이드바에서 **백서 PDF를 업로드**해 주세요.")
    if st.session_state.get("load_error"):
        st.caption(f"오류: {st.session_state.load_error}")
    st.stop()

tab_mat, tab_chat, tab_learn, tab_solve = st.tabs(
    ["📂 내 자료로 학습", "💬 AI 상담", "📚 나의 학습", "🛠️ 문제 해결"])

# =====================================================================
# 8-1. ★ 탭 0 — 내 자료로 학습 (핵심 차별점)
# =====================================================================
with tab_mat:
    st.markdown(svg_html("""
    <div class="flow">
      <div class="fl"><div class="fl-n">1</div><div><b>📂 학습 자료 올리기</b><br><span>교안 PDF · 실습 .ipynb · .py · 정리 노트</span></div></div>
      <div class="fl-arrow">➜</div>
      <div class="fl"><div class="fl-n">2</div><div><b>💬 궁금한 점 질문</b><br><span>개념 · 코드 · 오류 · 응용 무엇이든</span></div></div>
      <div class="fl-arrow">➜</div>
      <div class="fl"><div class="fl-n">3</div><div><b>🔍 AI 수준 진단</b><br><span>질문으로 학습 수준·위치 판단</span></div></div>
      <div class="fl-arrow">➜</div>
      <div class="fl"><div class="fl-n">4</div><div><b>🎯 눈높이 맞춤 설명</b><br><span>초급·중급·고급 전략으로 안내</span></div></div>
    </div>"""), unsafe_allow_html=True)

    main_col, prof_col = st.columns([2.3, 1], gap="large")

    # ---------------- 오른쪽: 나의 학습 프로필 ----------------
    with prof_col:
        with st.container(key="profile_card"):
            st.markdown("##### 🧑‍🎓 나의 학습 프로필")
            est = learner_level()
            if est:
                lv, score = est
                st.markdown(svg_html(f"""
                <div class="prof-lv"><span>AI가 추정한 현재 수준</span>
                  <b style="color:{LV_COLOR[lv]}">{lv}</b></div>
                <div class="bar"><div style="width:{(score - 1) / 2 * 100:.0f}%"></div></div>
                <div class="bar-lb"><span>초급</span><span>중급</span><span>고급</span></div>"""),
                            unsafe_allow_html=True)
                st.caption("최근 질문 흐름: " + " → ".join(
                    f"{h['level']}" for h in st.session_state.diag_history[-6:]))
                reached = max(STAGES.index(h["stage"]) for h in st.session_state.diag_history
                              if h.get("stage") in STAGES)
                st.caption(f"도달한 학습 위치: **{STAGES[reached]}**")
                pres = list(dict.fromkeys(h["prerequisite"] for h in st.session_state.diag_history
                                          if h.get("prerequisite")))[-4:]
                if pres:
                    st.markdown("**📌 보완하면 좋은 개념**")
                    for p in pres:
                        st.markdown(f"- {p}")
            else:
                st.caption("질문을 하면 AI가 질문의 용어·구체성·시도한 내용을 분석해 "
                           "학습 수준과 위치를 추정해요.")
                st.markdown(svg_html("".join(
                    f'<div class="stage-row">{s}</div>' for s in STAGES)), unsafe_allow_html=True)

            st.divider()
            st.session_state.auto_level = st.toggle(
                "진단 결과를 내 학습 수준에 자동 반영", value=st.session_state.auto_level,
                help="교육생 모드에서 사이드바의 '나의 학습 수준'이 진단 결과에 맞춰 자동으로 바뀝니다.")
            st.session_state.use_whitepaper = st.toggle(
                "백서도 함께 참고", value=st.session_state.use_whitepaper,
                help="학습 전략·운영 안내가 필요한 질문에 백서 내용을 함께 활용합니다.")
            if st.session_state.diag_history and st.button("진단 기록 초기화", width="stretch"):
                st.session_state.diag_history = []
                st.rerun()

    # ---------------- 왼쪽: 자료 업로드 + 대화 ----------------
    with main_col:
        with st.container(key="upload_card"):
            st.markdown("#### 📂 내 학습 자료")
            files = st.file_uploader(
                "교안·실습 파일을 올려 주세요 (여러 개 가능)", type=MATERIAL_TYPES,
                accept_multiple_files=True, key=f"mat_up_{st.session_state.mat_uploader_id}",
                help="PDF 교안, Jupyter 노트북(.ipynb, 실행 결과·오류 포함), 파이썬 코드, 텍스트·마크다운 정리 노트, CSV")
            b1, b2 = st.columns([2, 1])
            if files and b1.button(f"📥 {len(files)}개 자료 등록하고 분석하기", type="primary", width="stretch"):
                with st.status("자료를 학습 도우미에 등록하는 중...", expanded=True) as status:
                    st.write("📖 파일 읽기 · 셀/페이지 단위 분할")
                    store, chunks = build_material_store(files)
                    st.write(f"🧩 {len(chunks)}개 조각으로 임베딩 저장 완료")
                    st.write("🧠 학습 주제 분석 · 수준별 추천 질문 생성")
                    try:
                        info = analyze_materials(chunks) if chunks else {}
                    except Exception:
                        info = {}
                    status.update(label="등록 완료! 이제 자료에 대해 질문해 보세요.", state="complete")
                st.session_state.update(
                    mat_store=store, mat_chunks=len(chunks), mat_info=info, mat_messages=[],
                    mat_files=[(f.name, sum(1 for c in chunks if c.metadata.get("source_file") == f.name))
                               for f in files])
                st.rerun()
            if st.session_state.mat_files and b2.button("자료 비우기", width="stretch"):
                st.session_state.update(mat_store=None, mat_files=[], mat_info={}, mat_chunks=0,
                                        mat_messages=[], mat_uploader_id=st.session_state.mat_uploader_id + 1)
                st.rerun()

            if st.session_state.mat_files:
                chips = "".join(
                    f'<span class="fchip">{FILE_ICON.get(n.rsplit(".", 1)[-1].lower(), "📄")} {n} '
                    f'<i>{c}조각</i></span>' for n, c in st.session_state.mat_files)
                st.markdown(svg_html(f'<div class="fchips">{chips}</div>'), unsafe_allow_html=True)
                info = st.session_state.mat_info
                if info.get("summary"):
                    st.markdown(f"**📌 자료 요약** · {info['summary']}")
                if info.get("topics"):
                    st.markdown(svg_html("".join(f'<span class="chip">#{t}</span>' for t in info["topics"])),
                                unsafe_allow_html=True)
            else:
                st.caption("아직 등록된 자료가 없어요. 자료 없이 질문하면 백서를 바탕으로 수준을 진단해 답해요.")

        mat_area = st.container()
        with mat_area:
            mm = st.session_state.mat_messages
            # 처음: 자료 기반 수준별 추천 질문
            if not mm:
                qs = st.session_state.mat_info.get("questions") or {
                    "초급": ["청킹이 뭔가요?", "임베딩은 왜 필요한가요?"],
                    "중급": ["chunk_size를 바꾸면 결과가 어떻게 달라지나요?", "검색 k값은 어떻게 정하나요?"],
                    "고급": ["RAG 답변 품질은 어떻게 평가하나요?", "검색 성능을 높이는 방법은?"]}
                st.markdown("##### 💡 이런 질문으로 시작해 보세요 — 질문 수준에 따라 설명이 달라져요")
                qcols = st.columns(3)
                for ci, lv in enumerate(["초급", "중급", "고급"]):
                    with qcols[ci]:
                        st.markdown(svg_html(f'<span class="lv" style="background:{LV_COLOR[lv]}">{lv}</span>'),
                                    unsafe_allow_html=True)
                        for qi, q in enumerate(qs.get(lv, [])[:2]):
                            if st.button(q, key=f"mq_{lv}_{qi}", width="stretch"):
                                st.session_state.mat_pending = {"q": q}
                                st.rerun()

            for idx, m in enumerate(mm):
                with st.chat_message(m["role"], avatar="🙋" if m["role"] == "user" else "🤖"):
                    if m["role"] == "assistant":
                        st.markdown(diag_card(m["diag"], m["applied"]), unsafe_allow_html=True)
                    st.markdown(m["content"])
                    if m["role"] == "assistant":
                        show_material_sources(m.get("docs", []))
                        if idx == len(mm) - 1:
                            cur = LV_NUM[m["applied"]]
                            c1, c2, _ = st.columns([1.4, 1.4, 1.6])
                            if cur > 1 and c1.button(f"🐣 더 쉽게 ({NUM_LV[cur - 1]})", key="mat_easier"):
                                st.session_state.mat_pending = {"q": m["question"], "force": NUM_LV[cur - 1],
                                                                "replace": True}
                                st.rerun()
                            if cur < 3 and c2.button(f"🚀 더 깊게 ({NUM_LV[cur + 1]})", key="mat_deeper"):
                                st.session_state.mat_pending = {"q": m["question"], "force": NUM_LV[cur + 1],
                                                                "replace": True}
                                st.rerun()

        typed = st.chat_input("자료에 대해 질문해 보세요 (예: 3번 셀의 오류는 왜 나나요? / chunk_overlap은 왜 필요해요?)",
                              key="mat_input")
        req = {"q": typed} if typed else st.session_state.mat_pending
        st.session_state.mat_pending = None

        if req:
            q = req["q"]
            if req.get("replace") and len(st.session_state.mat_messages) >= 2:
                prev_diag = st.session_state.mat_messages[-1]["diag"]
                st.session_state.mat_messages = st.session_state.mat_messages[:-2]
            else:
                prev_diag = None
            recent = st.session_state.mat_messages[-4:]
            history_text = "\n".join(f"{'교육생' if m['role'] == 'user' else 'AI'}: {m['content'][:200]}"
                                     for m in recent)

            with mat_area:
                with st.chat_message("user", avatar="🙋"):
                    st.markdown(q)
                with st.chat_message("assistant", avatar="🤖"):
                    # 1) 질문 수준 진단 (다시 설명 요청이면 기존 진단 유지)
                    if prev_diag:
                        diag = prev_diag
                    else:
                        with st.spinner("🔍 질문 수준과 학습 위치를 진단하는 중..."):
                            diag = diagnose_level(q, history_text)
                    applied = req.get("force") or diag["level"]
                    st.markdown(diag_card(diag, applied), unsafe_allow_html=True)

                    # 2) 내 학습 자료(+백서) 검색
                    docs = []
                    if st.session_state.mat_store is not None:
                        docs += st.session_state.mat_store.similarity_search(q, k=4)
                    if st.session_state.use_whitepaper or not docs:
                        docs += retrieve(q)[: 2 if docs else 4]

                    # 3) 수준별 전략으로 맞춤 설명
                    context = "\n\n".join(
                        format_materials([d]) if d.metadata.get("source_file")
                        else f"[백서 · p.{page_of(d)}]\n{d.page_content}" for d in docs)
                    answer = st.write_stream(llm_stream(
                        material_system(applied, diag, st.session_state.use_whitepaper),
                        f"[학습 자료]\n{context}\n\n[이전 대화]\n{history_text or '없음'}\n\n[질문]\n{q}"))

            if not prev_diag:
                st.session_state.diag_history.append(diag)
                est = learner_level()
                if st.session_state.auto_level and st.session_state.user_type == "교육생" and est:
                    st.session_state.level = est[0]
            st.session_state.mat_messages += [
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer, "question": q, "diag": diag,
                 "applied": applied, "docs": docs}]
            st.rerun()





# =====================================================================
# 9. 탭 1 — AI 상담 (기능 2, 3, 5, 6, 10)
# =====================================================================
with tab_chat:
    chat_area = st.container()

    with chat_area:
        # 처음 사용자: 추천 질문 (기능 3)
        if not st.session_state.messages:
            st.markdown(f"#### 👋 반가워요! {st.session_state.level} {st.session_state.user_type}에게 추천하는 질문")
            cols = st.columns(3)
            for i, q in enumerate(RECOMMENDED[(st.session_state.user_type, st.session_state.level)]):
                if cols[i].button(f"💡 {q}", key=f"rec_{i}", width="stretch"):
                    st.session_state.pending = {"q": q, "length": "기본"}
                    st.rerun()

        # 대화 기록 출력
        msgs = st.session_state.messages
        for idx, m in enumerate(msgs):
            with st.chat_message(m["role"], avatar="🙋" if m["role"] == "user" else "🧭"):
                st.markdown(m["content"])
                if m["role"] == "assistant":
                    if m.get("rule"):
                        rule_badge()          # 기수·규정 구분 (기능 10)
                    show_sources(m.get("docs", []), key=f"src_{idx}")   # 근거 페이지 (기능 2)

                    # 마지막 답변에만: 답변 조절 + 연관 질문
                    if idx == len(msgs) - 1:
                        b1, b2, _ = st.columns([1, 1, 4])
                        if b1.button("✂️ 짧게", key="shorter"):
                            st.session_state.pending = {"q": m["question"], "length": "짧게", "replace": True}
                            st.rerun()
                        if b2.button("🔍 더 자세히", key="longer"):
                            st.session_state.pending = {"q": m["question"], "length": "자세히", "replace": True}
                            st.rerun()
                        if m.get("followups"):
                            st.caption("🔗 이어서 물어보면 좋은 질문")
                            fcols = st.columns(len(m["followups"]))
                            for j, fq in enumerate(m["followups"]):
                                if fcols[j].button(fq, key=f"fu_{j}", width="stretch"):
                                    st.session_state.pending = {"q": fq, "length": "기본"}
                                    st.rerun()

    typed = st.chat_input("에이블스쿨에 대해 무엇이든 물어보세요 (예: 비전공자도 따라갈 수 있을까요?)")

    request = None
    if typed:
        request = {"q": typed, "length": "기본"}
    elif st.session_state.pending:
        request = st.session_state.pending
        st.session_state.pending = None

    if request:
        q, length = request["q"], request["length"]
        if request.get("replace") and len(st.session_state.messages) >= 2:
            st.session_state.messages = st.session_state.messages[:-2]   # 이전 답변을 새 길이로 교체

        # 최근 대화 6개를 맥락으로 전달
        history = []
        for m in st.session_state.messages[-6:]:
            history.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))

        with chat_area:
            with st.chat_message("user", avatar="🙋"):
                st.markdown(q)
            with st.chat_message("assistant", avatar="🧭"):
                docs = retrieve(q)
                answer = st.write_stream(st.session_state.chain.stream({
                    "input": q,
                    "context": format_docs_with_page(docs),
                    "history": history,
                    "persona": PERSONA[st.session_state.user_type],
                    "level": level_text(),
                    "length_guide": LENGTH[length],
                }))
                with st.spinner("연관 질문을 찾는 중..."):
                    try:
                        fu = llm_json(
                            "당신은 에이블스쿨 백서 기반 챗봇의 후속 질문 추천기입니다.",
                            f"[사용자] {level_text()}\n[질문] {q}\n[답변] {answer}\n\n"
                            f"[백서 일부]\n{format_docs(docs)[:1500]}\n\n"
                            "사용자가 이어서 궁금해할 만하고 백서로 답할 수 있는 짧은 질문 3개를 "
                            '{"questions": ["...", "...", "..."]} 형식으로 만드세요. 각 질문은 30자 이내.')
                        followups = fu.get("questions", [])[:3]
                    except Exception:
                        followups = []

        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "question": q, "docs": docs,
            "followups": followups, "rule": is_rule_related(q + answer),
        })
        st.rerun()


# =====================================================================
# 10. 탭 2 — 나의 학습 (기능 7, 8, 12, 13)
# =====================================================================
with tab_learn:
    st.markdown("### 🗺️ 개인 맞춤형 학습 로드맵")
    st.caption("백서의 학습 전략·교육과정·선배 경험을 바탕으로 AI가 나만의 계획을 제안합니다. "
               "**AI가 만든 개인 학습 제안이며 공식 교육 일정이 아닙니다.**")

    with st.form("roadmap_form"):
        c1, c2 = st.columns(2)
        goal = c1.text_input("학습 목표", value="RAG 챗봇을 직접 구현하고 성능을 검증할 수 있게 되기")
        interests = c2.multiselect("관심 분야", INTERESTS, default=["RAG·LLM", "프로젝트·협업"])
        c3, c4 = st.columns(2)
        weeks = c3.slider("기간(주)", 1, 8, 4)
        hours = c4.slider("하루 학습 가능 시간(수업 외)", 1, 6, 2)
        background = st.text_input("나의 배경 (선택)", placeholder="예: 비전공자, 파이썬 기초만 앎")
        submitted = st.form_submit_button("✨ 로드맵 만들기", type="primary", width="stretch")

    if submitted:
        with st.spinner("백서를 참고해 나만의 로드맵을 설계하는 중..."):
            docs = retrieve_many([f"{st.session_state.level} 학습 전략 복습 방법",
                                  f"{goal} {' '.join(interests)}",
                                  "학습 계획 예시 주차별", "선배 후기 조언 프로젝트"])
            rm = llm_json(
                task_system("사용자의 수준·목표·기간·가용 시간에 맞는 주차별 학습 로드맵을 설계합니다. "
                            "수준별 차이: 초급은 개념 이해와 제공 코드 실행, 중급은 코드 수정·설정 비교, "
                            "고급은 품질 평가·최적화·배포 검증 중심으로 구성합니다."),
                f"[목표] {goal}\n[관심 분야] {', '.join(interests)}\n[기간] {weeks}주\n"
                f"[하루 가용 시간] {hours}시간\n[배경] {background or '미입력'}\n\n"
                f"[문서]\n{format_docs_with_page(docs)}\n\n"
                '다음 JSON 형식으로 작성: {"title": "로드맵 제목", "summary": "2~3문장 요약", '
                '"weeks": [{"week": 1, "goal": "주차 목표", "items": [{"topic": "학습 항목(15자 이내)", '
                '"task": "구체적 활동", "hours": 4}]}], "tips": ["백서 근거 선배 팁 (p.N)"]} '
                f"weeks는 정확히 {weeks}개, 주차별 items 3~4개.")
        st.session_state.update(roadmap=rm, roadmap_docs=docs, done={}, topic_output=None)

    rm = st.session_state.roadmap
    if rm and rm.get("weeks"):
        st.markdown(f"#### {rm.get('title', '나의 학습 로드맵')}")
        st.info(rm.get("summary", ""))

        all_items = [(w.get("week"), it) for w in rm["weeks"] for it in w.get("items", [])]
        done_cnt = sum(1 for k, v in st.session_state.done.items() if v)
        st.progress(done_cnt / max(len(all_items), 1), text=f"진행률 {done_cnt}/{len(all_items)}")

        wcols = st.columns(min(len(rm["weeks"]), 4))
        for i, w in enumerate(rm["weeks"]):
            with wcols[i % len(wcols)]:
                with st.container(border=True):
                    st.markdown(f"**{w.get('week')}주차**  \n{w.get('goal', '')}")
                    for j, it in enumerate(w.get("items", [])):
                        key = f"done_{i}_{j}"
                        st.session_state.done[key] = st.checkbox(
                            f"{it.get('topic')} ({it.get('hours')}h)", key=key,
                            value=st.session_state.done.get(key, False), help=it.get("task"))

        if rm.get("tips"):
            with st.expander("💬 백서 속 선배 팁"):
                for t in rm["tips"]:
                    st.markdown(f"- {t}")

        # 로드맵 내보내기 (기능 13)
        d1, d2 = st.columns(2)
        try:
            png, has_font = roadmap_png(rm)
            d1.download_button("🖼️ 로드맵 PNG 저장", png, file_name="my_aivle_roadmap.png",
                               mime="image/png", width="stretch")
            if not has_font:
                d1.caption("⚠️ 한글 폰트가 없어 글자가 깨질 수 있어요. 저장소에 packages.txt(fonts-nanum)를 추가하세요.")
        except Exception as e:
            d1.caption(f"PNG 생성 실패: {e}")
        d2.download_button("📝 로드맵 Markdown 저장", roadmap_markdown(rm),
                           file_name="my_aivle_roadmap.md", width="stretch")

        # 학습 항목 선택 → 설명 / 퀴즈 / 실무 활용 (기능 7 → 8, 12)
        st.markdown("---")
        st.markdown("### 🎯 학습 항목 선택")
        labels = [f"{wk}주차 · {it.get('topic')}" for wk, it in all_items]
        sel = st.selectbox("공부할 항목을 고르세요", range(len(labels)), format_func=lambda i: labels[i])
        topic = all_items[sel][1].get("topic", "")

        a1, a2, a3 = st.columns(3)
        action = None
        if a1.button("📖 관련 설명 보기", width="stretch"):
            action = "explain"
        if a2.button("📝 복습 퀴즈 풀기", width="stretch"):
            action = "quiz"
        if a3.button("💼 실무 활용 탐색", width="stretch"):
            action = "practice"

        if action in ("explain", "practice"):
            docs = retrieve(topic)
            if action == "explain":
                task = (f"'{topic}'을(를) 사용자 수준에 맞게 설명합니다. 구성: ① 한 줄 정의 ② 쉬운 비유 "
                        "③ 핵심 포인트 3가지 ④ 바로 해볼 실습 한 가지.")
            else:
                task = (f"'{topic}'을(를) KT·기업 DX 현장과 AI 개발자/DX 컨설턴트 직무, 미니·빅프로젝트에서 "
                        "어떻게 활용할 수 있는지 탐색합니다. 구성: ① 활용 장면 2~3개 ② 프로젝트 아이디어 1개 "
                        "③ 포트폴리오에 쓸 수 있는 한 문장. 백서 근거와 일반 지식을 구분해 표시하세요.")
            with st.container(border=True):
                out = st.write_stream(llm_stream(task_system(task),
                                                 f"[학습 항목] {topic}\n\n[문서]\n{format_docs_with_page(docs)}"))
            st.session_state.topic_output = {"title": topic, "text": out, "docs": docs}
        elif st.session_state.topic_output:
            with st.container(border=True):
                st.markdown(f"**{st.session_state.topic_output['title']}**")
                st.markdown(st.session_state.topic_output["text"])
        if st.session_state.topic_output:
            show_sources(st.session_state.topic_output["docs"], key="topic_src")
    else:
        topic, action = None, None

    # AI 퀴즈·복습 (기능 8)
    st.markdown("---")
    st.markdown("### 📝 AI 퀴즈 · 실력 점검")
    q1, q2, q3 = st.columns([3, 1, 1])
    quiz_topic = q1.text_input("퀴즈 주제", value=topic or (st.session_state.mat_info.get("topics") or ["RAG 개념과 구현"])[0],
                               key=f"quiz_topic_{topic}")
    diff = q2.selectbox("난이도", ["초급", "중급", "고급"],
                        index=["초급", "중급", "고급"].index(st.session_state.level)
                        if st.session_state.level in ["초급", "중급", "고급"] else 0)
    n_q = q3.selectbox("문항 수", [3, 5], index=0)
    quiz_src = st.radio("출제 범위", ["백서", "내 학습 자료"], horizontal=True,
                        index=1 if st.session_state.mat_store is not None else 0,
                        disabled=st.session_state.mat_store is None,
                        help="'📂 내 자료로 학습' 탭에서 자료를 등록하면 내 교안·실습 파일로 퀴즈를 낼 수 있어요.")

    if q1.button("🎲 퀴즈 만들기", type="primary") or action == "quiz":
        with st.spinner("백서를 바탕으로 문제를 출제하는 중..."):
            if quiz_src == "내 학습 자료" and st.session_state.mat_store is not None:
                docs = st.session_state.mat_store.similarity_search(quiz_topic, k=6)
                ctx = format_materials(docs)
            else:
                docs = retrieve_many([quiz_topic, f"{quiz_topic} 개념 정리"])
                ctx = format_docs_with_page(docs)
            quiz = llm_json(
                task_system(f"'{quiz_topic}'에 대해 {diff} 난이도의 4지선다 복습 퀴즈를 출제합니다. "
                            "초급은 개념 확인, 중급은 적용·비교, 고급은 원인 분석·평가 문제로 냅니다. "
                            "반드시 [문서] 내용으로 정답을 확정할 수 있는 문제만 냅니다."),
                f"[문서]\n{ctx}\n\n"
                f'{n_q}문항을 {{"questions": [{{"q": "문제", "options": ["보기1","보기2","보기3","보기4"], '
                '"answer": 0, "explanation": "해설 (p.N)"}]}} 형식으로. answer는 정답 보기의 인덱스(0~3).',
                temperature=0.6)
        st.session_state.quiz = {"topic": quiz_topic, "diff": diff, "items": quiz.get("questions", []),
                                 "id": datetime.datetime.now().strftime("%H%M%S%f")}
        st.session_state.quiz_result = None

    quiz = st.session_state.quiz
    if quiz and quiz["items"]:
        with st.form("quiz_form"):
            st.markdown(f"**주제: {quiz['topic']} · 난이도: {quiz['diff']}**")
            picks = []
            for i, item in enumerate(quiz["items"]):
                picks.append(st.radio(f"Q{i + 1}. {item['q']}", range(len(item["options"])),
                                      format_func=lambda k, it=item: it["options"][k],
                                      index=None, key=f"quiz_{i}_{quiz['id']}"))
            graded = st.form_submit_button("✅ 채점하기", type="primary")
        if graded:
            st.session_state.quiz_result = picks

        if st.session_state.quiz_result is not None:
            picks = st.session_state.quiz_result
            score = sum(1 for p, it in zip(picks, quiz["items"]) if p == it.get("answer"))
            total = len(quiz["items"])
            rate = score / total
            st.metric("점수", f"{score} / {total}", f"{rate:.0%}")
            if rate >= 0.8:
                st.success(f"🎉 {quiz['diff']} 수준을 잘 이해하고 있어요! 다음 난이도에 도전해 보세요.")
            elif rate >= 0.5:
                st.warning("👍 기본은 잡혔어요. 틀린 문제의 해설과 근거 페이지를 복습한 뒤 다시 풀어 보세요.")
            else:
                st.error("💪 한 단계 낮은 난이도로 개념부터 다시 다져 보세요. 꾸준함이 실력이 됩니다!")
            for i, (p, it) in enumerate(zip(picks, quiz["items"])):
                ok = p == it.get("answer")
                with st.expander(f"{'⭕' if ok else '❌'} Q{i + 1}. {it['q']}", expanded=not ok):
                    st.markdown(f"- 내 답: {it['options'][p] if p is not None else '미응답'}")
                    st.markdown(f"- 정답: **{it['options'][it.get('answer', 0)]}**")
                    st.markdown(f"- 해설: {it.get('explanation', '')}")


# =====================================================================
# 11. 탭 3 — 문제 해결 (기능 9, 10, 11)
# =====================================================================
with tab_solve:
    st.markdown("### 🛠️ 상황별 문제 해결")
    st.caption("출결·공가·실습 오류·프로젝트 등 지금 겪는 상황을 알려주세요. 백서 기준으로 할 일을 정리해 드려요.")

    category = st.pills("상황 유형", SITUATIONS, default=SITUATIONS[0]) or "기타"
    situation = st.text_area("상황 설명", height=110,
                             placeholder="예: 병원 진료 때문에 오전에 늦을 것 같아요. 어떻게 처리해야 하나요?")
    if category == "실습·코드 오류":
        err = st.text_area("오류 메시지 (선택)", height=90, placeholder="Traceback ... 을 붙여넣으세요")
    else:
        err = ""

    if st.button("🔎 해결 방법 안내받기", type="primary", disabled=not situation.strip()):
        docs = retrieve_many([f"{category} {situation}", category])
        task = ("사용자의 상황에 대해 다음 형식으로 안내합니다.\n"
                "### 1. 상황 요약\n### 2. 백서 기준 (기수 표기, 공식 운영 기준/실천 제안 구분)\n"
                "### 3. 지금 해야 할 일 (번호 목록, 기한·증빙 포함)\n"
                "### 4. 담당자 확인 필요 여부 — '필요' 또는 '불필요'와 그 이유\n"
                "코드 오류라면 원인 후보와 확인 순서를 제시합니다.")
        with st.container(border=True):
            sol = st.write_stream(llm_stream(
                task_system(task),
                f"[상황 유형] {category}\n[상황] {situation}\n[오류 메시지] {err or '없음'}\n\n"
                f"[문서]\n{format_docs_with_page(docs)}"))
        st.session_state.update(solution={"category": category, "situation": situation, "text": sol},
                                solution_docs=docs, inquiry=None)
    elif st.session_state.solution:
        with st.container(border=True):
            st.markdown(st.session_state.solution["text"])

    if st.session_state.solution:
        show_sources(st.session_state.solution_docs, key="solve_src")
        if is_rule_related(st.session_state.solution["text"]):
            rule_badge()

        # 공식 문의문 자동 작성 (기능 11)
        st.markdown("#### ✉️ 담당자 확인이 필요하다면")
        if st.button("공식 문의문 자동 작성"):
            s = st.session_state.solution
            with st.spinner("문의문을 작성하는 중..."):
                text = "".join(llm_stream(
                    task_system("담당 매니저에게 보낼 공손하고 간결한 공식 문의문을 작성합니다. "
                                "구성: 제목 / 인사 / 소속([기수]·[반]·[이름]) / 상황(날짜·시간 포함) / "
                                "문의 사항(번호 목록) / 첨부 예정 증빙 / 마무리 인사. "
                                "개인정보는 [ ] 빈칸으로 두고, 비밀번호·인증번호는 절대 넣지 않습니다."),
                    f"[상황 유형] {s['category']}\n[상황] {s['situation']}\n[안내 내용]\n{s['text']}"))
            st.session_state.inquiry = text
        if st.session_state.inquiry:
            edited = st.text_area("문의문 (수정 후 복사해 사용하세요)", st.session_state.inquiry, height=300)
            st.download_button("💾 문의문 저장(.txt)", edited, file_name="aivle_inquiry.txt")
