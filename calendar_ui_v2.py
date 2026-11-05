"""Compact calendar with event labels, manual registration, recommendations and PNG."""
import calendar
import io
import os
import html
from pathlib import Path
from datetime import date, time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from meeting_planner import add_schedule, add_todo


def _font(size):
    for name in (str(Path(__file__).parent / "assets/NanumGothic-Regular.ttf"), "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    raise RuntimeError("한글 글꼴이 없습니다. assets/NanumGothic-Regular.ttf를 함께 업로드하세요.")


def _wrap(draw, text, font, width):
    lines, line = [], ""
    for ch in str(text):
        if ch == "\n" or draw.textlength(line + ch, font=font) > width:
            lines.append(line)
            line = "" if ch == "\n" else ch
        else:
            line += ch
    return lines + [line]


def _png(lines, title):
    font = _font(20)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    rows = [row for line in lines for row in _wrap(probe, line, font, 880)]
    image = Image.new("RGB", (950, max(190, 120 + len(rows)*34)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25,25), title, font=_font(27), fill="#1f2937")
    for i, line in enumerate(rows):
        draw.text((25,90+i*34), line, font=font, fill="#334155")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def calendar_png(year, month, events):
    weeks = calendar.monthcalendar(year,month)
    img = Image.new('RGB',(1120,110+len(weeks)*125),'#191b22')
    d=ImageDraw.Draw(img)
    d.text((20,15),f'ABLE Calendar {year}-{month:02d}',font=_font(28),fill='#f1f5f9')
    for j,wd in enumerate(['월','화','수','목','금','토','일']):
        d.text((j*160+10,74),wd,font=_font(18),fill='#cbd5e1')
    for i,week in enumerate(weeks):
        for j,day in enumerate(week):
            x,y=j*160,110+i*125
            d.rectangle((x,y,x+159,y+124),outline='#383b49')
            if day:
                day_string=f'{year}-{month:02d}-{day:02d}'
                d.text((x+8,y+5),str(day),font=_font(18),fill='#f1f5f9')
                matches=[e for e in events if e['date']==day_string]
                for k,e in enumerate(matches[:3]):
                    d.text((x+8,y+32+k*27),(e.get('time','')+' '+e['schedule']).strip()[:10],font=_font(14),fill='#9d92ff')
    out=io.BytesIO()
    img.save(out,format='PNG')
    return out.getvalue()


def _recommended(title):
    if '회의' in title:
        return ['회의 안건 정리','회의 자료 준비','참석자 확인']
    if '발표' in title:
        return ['발표 자료 작성','발표 대본 준비','발표 리허설']
    if '보고서' in title:
        return ['자료 조사','보고서 초안 작성','최종 검토']
    return [f'{title} 준비 사항 확인']


def render_calendar():
    st.subheader('📅 일정 캘린더')
    year,month=st.session_state.calendar_year,st.session_state.calendar_month
    a,b,c,d=st.columns([1,3,1,2])
    if a.button('◀',key='prev_month'):
        month-=1
        if month<1:month,year=12,year-1
        st.session_state.calendar_year,st.session_state.calendar_month=year,month
        st.rerun()
    b.markdown(f'### {year}년 {month}월')
    if c.button('▶',key='next_month'):
        month+=1
        if month>12:month,year=1,year+1
        st.session_state.calendar_year,st.session_state.calendar_month=year,month
        st.rerun()
    d.download_button('📥 달력 PNG',calendar_png(year,month,st.session_state.schedules),file_name='calendar.png',mime='image/png',on_click='ignore')

    # HTML grid provides equal-height cells; date selection stays a native widget.
    colors = ['#7962d8', '#00a7af', '#4d68ed', '#dd3f8d', '#e99838']
    cells = ''.join(f'<div class="weekday">{wd}</div>' for wd in ['월','화','수','목','금','토','일'])
    for week in calendar.monthcalendar(year, month):
        for day in week:
            if not day:
                cells += '<div class="day muted"></div>'
                continue
            current = date(year, month, day)
            matches = [e for e in st.session_state.schedules if e['date'] == current.isoformat()]
            selected = ' selected' if current == st.session_state.selected_date else ''
            labels = ''.join(f'<div class="event" style="background:{colors[i%5]}" title="{html.escape(e["schedule"], quote=True)}">{html.escape((e.get("time", "") + " " + e["schedule"]).strip())}</div>' for i,e in enumerate(matches[:3]))
            if len(matches)>3: labels += f'<small>+{len(matches)-3}개</small>'
            cells += f'<div class="day{selected}"><span>{day}</span>{labels}</div>'
    st.markdown('<div class="able-calendar">'+cells+'</div>', unsafe_allow_html=True)
    selected = st.date_input('상세 일정을 볼 날짜', value=st.session_state.selected_date)
    if selected != st.session_state.selected_date:
        st.session_state.selected_date = selected
        st.session_state.calendar_year, st.session_state.calendar_month = selected.year, selected.month
        st.rerun()
    with st.expander('이번 달 일정 목록', expanded=True):
        events = sorted([e for e in st.session_state.schedules if e['date'].startswith(f'{year}-{month:02d}')], key=lambda e:(e['date'],e.get('time','')))
        for i,e in enumerate(events):
            st.markdown(f'<div class="agenda" style="border-left-color:{colors[i%5]}">{html.escape(e["date"] + " · " + e.get("time", "") + " " + e["schedule"])}</div>', unsafe_allow_html=True)
        if not events: st.caption('등록된 일정이 없습니다. 아래에서 일정을 추가하세요.')


def render_details():
    selected=st.session_state.selected_date
    left,right=st.columns([1.1,1],gap='medium')
    with left:
        st.subheader(f'📝 {selected.isoformat()} 상세 일정')
        matches=[x for x in st.session_state.schedules if x['date']==selected.isoformat()]
        if not matches:st.info('선택한 날짜의 일정이 없습니다.')
        for item in list(matches):
            a,b=st.columns([5,1])
            a.write('• '+(item.get('time','')+' '+item['schedule']).strip())
            if b.button('삭제',key=f'del_schedule_{item["id"]}'):
                st.session_state.schedules.remove(item)
                st.rerun()
        voice_entry('schedule')
        has_time=st.checkbox('시간 지정',key='has_time')
        with st.form('schedule_form',clear_on_submit=True):
            title=st.text_input('일정 내용',key='schedule_draft')
            when=st.time_input('일정 시간',value=time(9,0)) if has_time else None
            submitted=st.form_submit_button('선택한 날짜에 등록')
        if submitted:
            try:
                if add_schedule(title,selected.isoformat(),when.strftime('%H:%M') if when else ''):
                    st.session_state.calendar_recommendations=_recommended(title)
                    st.rerun()
                st.info('이미 등록된 일정입니다.')
            except ValueError as exc:
                st.warning(str(exc))
    with right:
        st.subheader('💡 추천 To-do List')
        analysis=st.session_state.get('meeting_analysis') or {}
        recs=[{'title':x['title'],'assignee':x.get('assignee',''),'parent':x.get('parent','')} for x in analysis.get('todos',[])]
        recs += [{'title':x,'assignee':'','parent':''} for x in st.session_state.calendar_recommendations]
        if not recs:st.caption('회의록 분석 또는 일정 등록 후 추천이 나타납니다.')
        for i,item in enumerate(recs):
            a,b=st.columns([5,1])
            a.write('• '+item['title'])
            if b.button('추가',key=f'add_rec_{i}'):
                add_todo(item['title'],item['assignee'],item['parent'])
                st.rerun()


def render_todos():
    st.subheader('✅ My To-do List')
    voice_entry('todo')
    with st.form('todo_form',clear_on_submit=True):
        title=st.text_input('새로운 할 일',key='todo_draft')
        submit=st.form_submit_button('업무 추가')
    if submit:
        try:
            if add_todo(title):st.rerun()
        except ValueError as exc:st.warning(str(exc))
    total=len(st.session_state.todos)
    done=sum(x['done'] for x in st.session_state.todos)
    st.caption(f'완료 {done}/{total}')
    for item in list(st.session_state.todos):
        a,b=st.columns([4,2])
        item['done']=a.checkbox(item['task'],value=item['done'],key=f'todo_{item["id"]}')
        if b.button('삭제',key=f'del_todo_{item["id"]}'):
            st.session_state.todos.remove(item)
            st.rerun()
    lines=[('[완료]' if x['done'] else '[미완료]')+' '+x['task'] for x in st.session_state.todos]
    st.download_button('📥 To-do PNG',_png(lines,'My To-do List'),file_name='todos.png',mime='image/png',on_click='ignore')


def voice_entry(kind):
    from openai import OpenAI
    from audio_utils import transcribe_audio
    with st.expander('음성으로 내용 입력'):
        audio = st.audio_input('녹음 후 텍스트로 변환하세요', key=f'{kind}_audio')
        if st.button('녹음 내용 변환', key=f'{kind}_transcribe', disabled=audio is None):
            try:
                with st.spinner('음성 인식 중...'):
                    audio.seek(0)
                    st.session_state[f'{kind}_draft'] = transcribe_audio(OpenAI(), audio)
                st.success('아래 입력란을 확인한 후 등록하세요.')
            except Exception as exc:
                st.error(f'음성 인식 실패: {exc}')
