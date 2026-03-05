"""
Flask Backend API for IFRS Digital Co-worker
Reuses logic from IFRS_chat_streamlit_final.py
"""

import os
import sys
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from io import BytesIO
import traceback
import pandas as pd

from session_store import SessionStore

# Add parent directory to path to import from RAG engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the RAG engine (no Streamlit dependencies)
from rag_engine.answer import answer_with_refine_chain
from rag_engine.exports import (
    REPORTLAB_AVAILABLE,
    FPDF_AVAILABLE,
    _build_pdf_reportlab,
    _build_pdf_fpdf,
    _build_html_export,
)
from rag_engine.formatting import (
    _format_duration,
    _make_excerpt,
    _unify_metadata,
    fix_citation_format,
    format_visible_answer,
    remove_citations,
    sanitize_text,
)
from rag_engine.tables import extract_markdown_tables_as_dfs
from rag_engine.translate import translate_to_arabic
from rag_engine.engine import filter_page_zero_references

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

session_store = SessionStore()
PROMPT_WORKERS = ThreadPoolExecutor(max_workers=4)
HISTORY_LOCK = threading.Lock()


def _get_user_context(data, args):
    username = (data.get("username") if data else None) or args.get("username")
    email = (data.get("email") if data else None) or args.get("email")
    user_id = (data.get("user_id") if data else None) or args.get("user_id")
    if not username or not email:
        return None, None, None, "username and email are required"
    if not user_id:
        user_id = f"{username.strip().lower()}::{email.strip().lower()}"
    return user_id, username.strip(), email.strip(), None


def _resolve_user_context(data, args):
    # Prompt-service compatible payload: accepts userId without requiring username/email.
    prompt_user_id = (data.get("userId") if data else None) or args.get("userId")
    if prompt_user_id is not None:
        prompt_user_id = str(prompt_user_id).strip()
        if not prompt_user_id:
            return None, None, None, "userId is required"
        username = (data.get("username") if data else None) or args.get("username") or prompt_user_id
        email = (data.get("email") if data else None) or args.get("email") or f"{prompt_user_id}@local"
        return prompt_user_id, username.strip(), email.strip(), None
    return _get_user_context(data, args)


def _generate_prompt_id():
    return f"IFRS-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"


def _derive_prompt_title(text, limit=80):
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return "Untitled prompt"
    return clean[:limit]


def _first_tabular_data(tables_payload):
    if not tables_payload:
        return {"headers": [], "rows": []}
    first = tables_payload[0] or {}
    return {
        "headers": first.get("columns") or [],
        "rows": first.get("rows") or [],
    }


def _prompt_payload(chat_entry):
    tabular = chat_entry.get("promptResponseTabularData", {"headers": [], "rows": []})
    headers = tabular.get("headers") or []
    rows = tabular.get("rows") or []
    tabular_content = {"header": headers, "rows": rows} if headers or rows else None

    return {
        "promptId": chat_entry.get("prompt_id"),
        "promptStatus": chat_entry.get("prompt_status", "Completed"),
        "promptTitle": chat_entry.get("prompt_title") or _derive_prompt_title(chat_entry.get("question", "")),
        "promptResponseText": chat_entry.get("answer", ""),
        "promptResponseTabularContent": tabular_content,
        # Keep full render-critical data once, without repeating top-level prompt fields.
        "renderData": {
            "mode": chat_entry.get("mode"),
            "kb": chat_entry.get("kb"),
            "userId": chat_entry.get("user_id"),
            "username": chat_entry.get("username"),
            "email": chat_entry.get("email"),
            "question": chat_entry.get("question"),
            "stageAnswers": chat_entry.get("stage_answers", {}),
            "sources": chat_entry.get("sources", []),
            "timeTakenSec": chat_entry.get("time_taken_sec"),
            "hasTables": chat_entry.get("has_tables", False),
            "tableData": chat_entry.get("table_data", []),
            "tables": chat_entry.get("tables", []),
            "isArabic": chat_entry.get("is_arabic", False),
            "originalPromptId": chat_entry.get("original_prompt_id"),
            "referencedPromptId": chat_entry.get("referenced_prompt_id"),
            "promptRequestText": chat_entry.get("promptRequestText"),
        },
    }


def _prompt_success_response(chat_entry):
    return {
        "success": True,
        "errors": None,
        "code": 200,
        "payload": _prompt_payload(chat_entry),
    }


def _prompt_error_response(message, code):
    return {
        "success": False,
        "errors": [message],
        "code": code,
        "payload": None,
    }


def get_session(user_id):
    history = session_store.load_history(user_id)
    return {"chat_history": history}


def save_session(user_id, session):
    session_store.save_history(user_id, session.get("chat_history", []))


def _insert_chat_entry(user_id, entry):
    with HISTORY_LOCK:
        session = get_session(user_id)
        session["chat_history"].insert(0, entry)
        save_session(user_id, session)


def _update_chat_entry(user_id, prompt_id, changes):
    with HISTORY_LOCK:
        session = get_session(user_id)
        history = session.get("chat_history", [])
        for idx, entry in enumerate(history):
            if entry.get("prompt_id") == prompt_id:
                updated = dict(entry)
                updated.update(changes)
                history[idx] = updated
                save_session(user_id, session)
                return updated
        return None


def _build_thinking_entry(user_id, username, email, question, prompt_id):
    return {
        "mode": "Answer from Database",
        "kb": "IFRS A/B/C",
        "user_id": user_id,
        "username": username,
        "email": email,
        "question": question,
        "answer": "Thinking",
        "stage_answers": {},
        "sources": [],
        "time_taken_sec": None,
        "has_tables": False,
        "table_data": [],
        "tables": [],
        "prompt_id": prompt_id,
        "original_prompt_id": prompt_id,
        "referenced_prompt_id": None,
        "prompt_status": "Thinking",
        "prompt_title": _derive_prompt_title(question),
        "promptRequestText": question,
        "promptResponseText": "Thinking",
        "promptResponseTabularData": {"headers": [], "rows": []},
        "is_arabic": False,
    }


def _run_answer_pipeline(question):
    import time
    start_t = time.perf_counter()
    res = answer_with_refine_chain(question)
    elapsed = time.perf_counter() - start_t

    answer = res.get("answer_text") or res.get("answer") or ""
    exception_section = res.get("exception_section", "")

    if exception_section:
        answer = answer.rstrip() + "\n\n" + exception_section

    if answer.strip().lower() != "sources not found.":
        answer = format_visible_answer(answer)

    stage_clean = {}
    for k, v in (res.get("stage_answers") or {}).items():
        vv = v if isinstance(v, str) else ""
        vv = format_visible_answer(vv)
        vv = remove_citations(vv)
        stage_clean[k] = vv

    display_sources = filter_page_zero_references(res.get("sources", []))
    if display_sources is None:
        display_sources = []

    sources_list = []
    for doc in display_sources:
        meta = _unify_metadata(getattr(doc, "metadata", {}) or {})
        chunk_text = getattr(doc, "page_content", "") or ""
        sources_list.append({
            "doc_name": meta.get("doc_name", "Document"),
            "chapter": meta.get("chapter", "—"),
            "chapter_name": meta.get("chapter_name", "—"),
            "para_number": meta.get("para_number", "—"),
            "header": meta.get("header", "—"),
            "page": meta.get("page", 0),
            "publisher": meta.get("publisher", "—"),
            "excerpt": _make_excerpt(chunk_text, max_chars=900),
        })

    tables_payload = res.get("tables") or []
    table_data = []
    for idx, t in enumerate(tables_payload, start=1):
        cols = t.get("columns") or []
        rows = t.get("rows") or []
        try:
            df = pd.DataFrame(rows, columns=cols)
        except Exception:
            continue
        df = df.fillna("").applymap(lambda v: "—" if str(v).strip() == "" else v)
        table_data.append({
            "index": idx,
            "csv": df.to_csv(index=False),
            "row_count": len(df),
            "col_count": len(df.columns),
        })

    return {
        "answer": answer,
        "stage_answers": stage_clean,
        "sources": sources_list,
        "time_taken_sec": elapsed,
        "has_tables": len(table_data) > 0,
        "table_data": table_data,
        "tables": tables_payload,
        "promptResponseTabularData": _first_tabular_data(tables_payload),
    }


def _complete_ask_prompt(user_id, prompt_id, question):
    try:
        result = _run_answer_pipeline(question)
        _update_chat_entry(user_id, prompt_id, {
            "answer": result["answer"],
            "stage_answers": result["stage_answers"],
            "sources": result["sources"],
            "time_taken_sec": result["time_taken_sec"],
            "has_tables": result["has_tables"],
            "table_data": result["table_data"],
            "tables": result["tables"],
            "prompt_status": "Completed",
            "promptResponseText": result["answer"],
            "promptResponseTabularData": result["promptResponseTabularData"],
        })
    except Exception as exc:
        traceback.print_exc()
        _update_chat_entry(user_id, prompt_id, {
            "prompt_status": "Failed",
            "answer": f"Request failed: {str(exc)}",
            "promptResponseText": f"Request failed: {str(exc)}",
            "promptResponseTabularData": {"headers": [], "rows": []},
            "has_tables": False,
            "table_data": [],
            "tables": [],
        })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'IFRS Backend API Running'})


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Main endpoint to ask a question"""
    try:
        data = request.json or {}
        question = (data.get('question') or data.get('promptRequestText') or '').strip()
        user_id, username, email, err = _resolve_user_context(data, request.args)
        if err:
            return jsonify(_prompt_error_response(err, 400)), 400

        print(f"\n{'='*80}")
        print(f"NEW QUESTION RECEIVED: {question}")
        print(f"User ID: {user_id}")
        print(f"{'='*80}\n")
        sys.stdout.flush()

        if not question:
            return jsonify(_prompt_error_response('Question is required', 400)), 400

        prompt_id = _generate_prompt_id()
        thinking_entry = _build_thinking_entry(user_id, username, email, question, prompt_id)
        _insert_chat_entry(user_id, thinking_entry)

        PROMPT_WORKERS.submit(_complete_ask_prompt, user_id, prompt_id, question)
        return jsonify(_prompt_success_response(thinking_entry)), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(_prompt_error_response(str(e), 500)), 500


@app.route('/api/followup', methods=['POST'])
def followup_question():
    """Follow-up question endpoint"""
    try:
        data = request.json or {}
        question = (data.get('question') or data.get('promptRequestText') or '').strip()
        referenced_prompt_id = (data.get("promptId") or "").strip()
        user_id, username, email, err = _resolve_user_context(data, request.args)
        if err:
            return jsonify(_prompt_error_response(err, 400)), 400

        if not question:
            return jsonify(_prompt_error_response('Question is required', 400)), 400

        session = get_session(user_id)

        # Prompt-service compatible follow-up:
        # if promptId is provided, use that exact prompt as context root.
        base_entry = None
        if referenced_prompt_id:
            for entry in session.get('chat_history', []):
                if entry.get("prompt_id") == referenced_prompt_id:
                    base_entry = entry
                    break
            if base_entry is None:
                return jsonify(_prompt_error_response('Unknown promptId for userId', 404)), 404

        # Fallback to latest non-Arabic entry if promptId not supplied
        if base_entry is not None:
            prev_q = (
                base_entry.get('question')
                or base_entry.get('promptRequestText')
                or ''
            )
            root_prompt_id = base_entry.get("original_prompt_id") or referenced_prompt_id
        else:
            prev_q = ''
            root_prompt_id = None
            for entry in session.get('chat_history', []):
                if not entry.get('is_arabic') and entry.get('prompt_status') == 'Completed':
                    prev_q = (
                        entry.get('question')
                        or entry.get('promptRequestText')
                        or ''
                    )
                    break
        combined = f"{prev_q} {question}".strip()

        # Call ask endpoint logic
        import time
        start_t = time.perf_counter()
        res = answer_with_refine_chain(combined)
        elapsed = time.perf_counter() - start_t

        # Get raw answer and exception section (same as ask endpoint)
        answer = res.get("answer_text") or res.get("answer") or ""
        exception_section = res.get("exception_section", "")

        # Combine raw sections BEFORE formatting
        if exception_section:
            answer = answer.rstrip() + "\n\n" + exception_section

        # Format the complete combined answer
        if answer.strip().lower() != "sources not found.":
            answer = format_visible_answer(answer)

        stage_clean = {}
        for k, v in (res.get("stage_answers") or {}).items():
            vv = v if isinstance(v, str) else ""
            vv = format_visible_answer(vv)
            vv = remove_citations(vv)
            stage_clean[k] = vv

        display_sources = filter_page_zero_references(res["sources"])
        if display_sources is None:
            display_sources = []

        sources_list = []
        for doc in display_sources:
            meta = _unify_metadata(getattr(doc, "metadata", {}) or {})
            chunk_text = getattr(doc, "page_content", "") or ""
            sources_list.append({
                'doc_name': meta.get('doc_name', 'Document'),
                'chapter': meta.get('chapter', '—'),
                'chapter_name': meta.get('chapter_name', '—'),
                'para_number': meta.get('para_number', '—'),
                'header': meta.get('header', '—'),
                'page': meta.get('page', 0),
                'publisher': meta.get('publisher', '—'),
                'excerpt': _make_excerpt(chunk_text, max_chars=900)
            })

        tables_payload = res.get("tables") or []
        table_data = []
        for idx, t in enumerate(tables_payload, start=1):
            cols = t.get("columns") or []
            rows = t.get("rows") or []
            try:
                df = pd.DataFrame(rows, columns=cols)
            except Exception:
                continue
            df = df.fillna("").applymap(lambda v: "—" if str(v).strip() == "" else v)
            table_data.append({
                'index': idx,
                'csv': df.to_csv(index=False),
                'row_count': len(df),
                'col_count': len(df.columns)
            })
        has_tables = len(table_data) > 0

        chat_entry = {
            "mode": "Answer from Database",
            "kb": "IFRS A/B/C",
            "user_id": user_id,
            "username": username,
            "email": email,
            "question": combined,
            "answer": answer,
            "stage_answers": stage_clean,
            "sources": sources_list,
            "time_taken_sec": elapsed,
            "has_tables": has_tables,
            "table_data": table_data,
            "tables": tables_payload,
            "prompt_id": _generate_prompt_id(),
            "original_prompt_id": root_prompt_id,
            "referenced_prompt_id": referenced_prompt_id or None,
            "prompt_status": "Completed",
            "prompt_title": _derive_prompt_title(question),
            "promptRequestText": answer,
            "promptResponseText": answer,
            "promptResponseTabularData": _first_tabular_data(tables_payload),
            "is_arabic": False
        }
        if not chat_entry["original_prompt_id"]:
            chat_entry["original_prompt_id"] = chat_entry["prompt_id"]

        session['chat_history'].insert(0, chat_entry)
        save_session(user_id, session)

        return jsonify(_prompt_success_response(chat_entry)), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(_prompt_error_response(str(e), 500)), 500


@app.route('/api/translate', methods=['POST'])
def translate():
    """Translate latest answer to Arabic"""
    try:
        data = request.json or {}
        user_id, username, email, err = _resolve_user_context(data, request.args)
        if err:
            return jsonify(_prompt_error_response(err, 400)), 400

        session = get_session(user_id)

        if not session['chat_history']:
            return jsonify(_prompt_error_response('No chat history to translate', 400)), 400

        latest = session['chat_history'][0]

        if latest.get('is_arabic'):
            return jsonify(_prompt_error_response('Already translated to Arabic', 400)), 400
        if latest.get('prompt_status') == 'Thinking':
            return jsonify(_prompt_error_response('Cannot translate while response is still thinking', 400)), 400

        # Translate
        q_ar = translate_to_arabic(latest["question"])
        a_ar = translate_to_arabic(latest["answer"])

        q_ar = remove_citations(fix_citation_format(q_ar))
        a_ar = remove_citations(fix_citation_format(a_ar))

        # Create Arabic entry
        arabic_entry = {
            "mode": latest.get("mode", "Answer from Database"),
            "kb": latest.get("kb", "IFRS A/B/C"),
            "user_id": user_id,
            "username": username,
            "email": email,
            "question": q_ar,
            "answer": a_ar,
            "sources": latest.get("sources", []),
            "stage_answers": latest.get("stage_answers", {}),
            "tables": latest.get("tables", []),
            "table_data": latest.get("table_data", []),
            "has_tables": latest.get("has_tables", False),
            "prompt_id": latest.get("prompt_id"),
            "original_prompt_id": latest.get("original_prompt_id"),
            "referenced_prompt_id": latest.get("referenced_prompt_id"),
            "prompt_status": latest.get("prompt_status", "Completed"),
            "prompt_title": latest.get("prompt_title", _derive_prompt_title(q_ar)),
            "promptRequestText": q_ar,
            "promptResponseText": a_ar,
            "promptResponseTabularData": latest.get("promptResponseTabularData", {"headers": [], "rows": []}),
            "is_arabic": True,
            "time_taken_sec": latest.get("time_taken_sec"),
        }

        session['chat_history'].insert(0, arabic_entry)
        save_session(user_id, session)

        return jsonify(_prompt_success_response(arabic_entry)), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(_prompt_error_response(str(e), 500)), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history"""
    user_id, username, email, err = _resolve_user_context({}, request.args)
    if err:
        return jsonify(_prompt_error_response(err, 400)), 400
    session = get_session(user_id)

    return jsonify({
        "success": True,
        "errors": None,
        "code": 200,
        "payload": {
            "history": [_prompt_payload(entry) for entry in session['chat_history']]
        },
    }), 200


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear chat history"""
    data = request.json or {}
    user_id, username, email, err = _resolve_user_context(data, request.args)
    if err:
        return jsonify(_prompt_error_response(err, 400)), 400

    session = get_session(user_id)
    session['chat_history'] = []
    save_session(user_id, session)

    return jsonify({
        "success": True,
        "errors": None,
        "code": 200,
        "payload": {
            "message": "Chat history cleared"
        },
    }), 200


@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Export table as CSV (supports multiple tables by index)"""
    try:
        data = request.json
        table_markdown = data.get('table_markdown', '')
        table_index = data.get('table_index', 0)  # 0-based index, default to first table

        if not table_markdown:
            return jsonify({'error': 'No table data provided'}), 400

        # Extract table
        tables = extract_markdown_tables_as_dfs(table_markdown)

        if not tables:
            return jsonify({'error': 'No valid table found'}), 400

        # Validate table index
        if table_index < 0 or table_index >= len(tables):
            return jsonify({'error': f'Invalid table index. Found {len(tables)} table(s).'}), 400

        # Convert specified table to CSV
        df = tables[table_index]
        csv_buffer = BytesIO()
        csv_buffer.write(df.to_csv(index=False).encode('utf-8'))
        csv_buffer.seek(0)

        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'table_{table_index + 1}_export.csv'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    """Export conversation as PDF"""
    try:
        data = request.json or {}
        user_id, username, email, err = _resolve_user_context(data, request.args)
        if err:
            return jsonify({'error': err}), 400

        session = get_session(user_id)

        if not session['chat_history']:
            return jsonify({'error': 'No chat history to export'}), 400

        # Convert sources back to Document-like objects
        history_for_pdf = []
        for chat in session['chat_history']:
            # Create simple objects that have the required attributes
            class SimpleDoc:
                def __init__(self, metadata, page_content):
                    self.metadata = metadata
                    self.page_content = page_content

            sources_as_docs = []
            for src in chat.get('sources', []):
                doc = SimpleDoc(
                    metadata={
                        'doc_name': src.get('doc_name'),
                        'chapter': src.get('chapter'),
                        'chapter_name': src.get('chapter_name'),
                        'para_number': src.get('para_number'),
                        'header': src.get('header'),
                        'page': src.get('page'),
                        'publisher': src.get('publisher'),
                    },
                    page_content=src.get('excerpt', '')
                )
                sources_as_docs.append(doc)

            history_for_pdf.append({
                'question': chat['question'],
                'answer': chat['answer'],
                'stage_answers': chat.get('stage_answers', {}),
                'sources': sources_as_docs,
                'time_taken_sec': chat.get('time_taken_sec', 0),
                'tables': chat.get('tables', []),
            })

        # Build PDF
        if REPORTLAB_AVAILABLE:
            pdf_bytes = _build_pdf_reportlab(history_for_pdf)
        elif FPDF_AVAILABLE:
            pdf_bytes = _build_pdf_fpdf(history_for_pdf)
        else:
            return jsonify({'error': 'No PDF library available'}), 500

        pdf_buffer = BytesIO(pdf_bytes)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='ifrs_chat_history.pdf'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export/html', methods=['POST'])
def export_html():
    """Export conversation as HTML"""
    try:
        data = request.json or {}
        user_id, username, email, err = _resolve_user_context(data, request.args)
        if err:
            return jsonify({'error': err}), 400

        session = get_session(user_id)

        if not session['chat_history']:
            return jsonify({'error': 'No chat history to export'}), 400

        history_for_export = []
        for chat in session['chat_history']:
            history_for_export.append({
                'question': chat['question'],
                'answer': chat['answer'],
                'time_taken_sec': chat.get('time_taken_sec', 0),
                'tables': chat.get('tables', []),
            })

        html_text = _build_html_export(history_for_export)
        html_buffer = BytesIO(html_text.encode("utf-8"))

        return send_file(
            html_buffer,
            mimetype='text/html; charset=utf-8',
            as_attachment=True,
            download_name='ifrs_chat_history.html'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/updatestatus', methods=['POST'])
def update_status():
    """Prompt-service compatible status lookup."""
    try:
        data = request.json or {}
        user_id, username, email, err = _resolve_user_context(data, request.args)
        if err:
            return jsonify(_prompt_error_response(err, 400)), 400

        prompt_id = (data.get("promptId") or "").strip()
        if not prompt_id:
            return jsonify(_prompt_error_response('promptId is required', 400)), 400

        session = get_session(user_id)
        record = None
        for entry in session.get("chat_history", []):
            if entry.get("prompt_id") == prompt_id:
                record = entry
                break

        if not record:
            return jsonify(_prompt_error_response('Unknown promptId for userId', 404)), 404

        return jsonify(_prompt_success_response(record)), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(_prompt_error_response(str(e), 500)), 500


@app.route('/health', methods=['GET'])
def health_check_prompt_compat():
    """Health endpoint alias for prompt-services clients."""
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    print("Starting IFRS Digital Co-worker Backend...")
    print("API will be available at http://localhost:3000")
    app.run(debug=True, host='0.0.0.0', port=3000)
