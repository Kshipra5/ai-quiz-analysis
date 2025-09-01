import streamlit as st
from dotenv import load_dotenv
import os
import PyPDF2
import json
import re
import random
import time
import textwrap

# --- Configure / Load key ---
load_dotenv("a.env")  # your a.env file
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    st.error("No GEMINI_API_KEY found in a.env. Put GEMINI_API_KEY=your_key in a.env and restart.")
    st.stop()

import google.generativeai as genai
genai.configure(api_key=GEMINI_KEY)


# ---------------- helpers ----------------
def extract_text_from_pdf(uploaded_file, max_pages=8):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        texts = []
        for i, p in enumerate(reader.pages[:max_pages]):
            t = p.extract_text() or ""
            texts.append(t)
        text = "\n".join(texts)
        text = " ".join(text.split())  # normalize whitespace
        return text
    except Exception as e:
        st.warning(f"PDF read failed: {e}")
        return ""


def extract_json_from_text(text):
    """Try multiple robust strategies to extract JSON from model output."""
    if not text or not isinstance(text, str):
        return None

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) find first balanced JSON object/array by scanning
    def find_balanced(start_chars):
        for start_char in start_chars:
            idx = text.find(start_char)
            if idx == -1:
                continue
            pairs = {'{': '}', '[': ']'}
            open_c = start_char
            close_c = pairs[open_c]
            depth = 0
            for i in range(idx, len(text)):
                ch = text[i]
                if ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break
        return None

    parsed = find_balanced(['[', '{'])
    if parsed is not None:
        return parsed

    # 3) look for triple-backtick blocks that contain JSON
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", text, flags=re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 4) fallback - try to extract first {...} or [...] using regex (less reliable)
    m2 = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass

    return None


def pretty_truncate(s, n=120):
    return (s.strip()[:n] + "...") if len(s.strip()) > n else s.strip()


def is_valid_quiz(obj, expected_n):
    """Validate the parsed JSON roughly matches expected schema."""
    if obj is None:
        return False
    # Accept either top-level list of question objects or dict with 'questions'
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and "questions" in obj and isinstance(obj["questions"], list):
        items = obj["questions"]
    else:
        return False

    if len(items) < 1:
        return False

    # Validate each question roughly
    for q in items:
        if not isinstance(q, dict):
            return False
        if "question" not in q:
            return False
        if "options" not in q or not isinstance(q["options"], list) or len(q["options"]) < 2:
            return False
        # allow either answer_index or answer (string)
        if not ("answer_index" in q or "answer" in q):
            return False

    # If expected_n provided, prefer that many (but be lenient)
    return True


def rule_based_fallback(content, num_questions=5, difficulty="medium"):
    """Simple fallback generator: pick sentences and create MCQs by sampling distractor sentences.
       Not perfect, but ensures the app returns something when Gemini fails."""
    # split into candidate sentences
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip()) > 30]
    if not sents:
        sents = [content.strip()][:5]

    # ensure some variety
    random.shuffle(sents)
    items = []
    for i in range(min(num_questions, max(1, len(sents)))):
        correct = sents[i]
        # pick 3 distractors from other sentences (or slightly mutated versions)
        others = [s for j, s in enumerate(sents) if j != i]
        random.shuffle(others)
        distractors = others[:3] if len(others) >= 3 else (others + ["Option not in text"] * 3)[:3]
        options = [pretty_truncate(correct, 130)] + [pretty_truncate(d, 130) for d in distractors]
        random.shuffle(options)
        answer_index = options.index(pretty_truncate(correct, 130))
        items.append({
            "question": pretty_truncate(correct, 200).replace('"', "'"),
            "options": options,
            "answer_index": answer_index,
            "explanation": ""  # fallback does not produce explanation
        })
    # wrap in top-level object for consistency
    return {"quiz_title": "Fallback Quiz (generated locally)", "difficulty": difficulty.capitalize(), "questions": items}


# ---------------- Gemini interaction & robust generation ----------------
def build_prompt(content, num_questions, difficulty):
    # Strict JSON schema in prompt
    schema = {
        "quiz_title": "string",
        "difficulty": "Easy|Medium|Hard",
        "questions": [
            {
                "question": "string",
                "options": ["string","string","string","string"],
                "answer_index": 0,
                "explanation": "string or empty"
            }
        ]
    }
    prompt = textwrap.dedent(f"""
    You are an expert teacher. Based ONLY on the content below, create exactly {num_questions} multiple-choice questions.
    Each question must have exactly 4 options. Provide the index (0-3) of the correct option.
    RETURN ONLY A SINGLE JSON OBJECT exactly following this schema (no extra text, no commentary, no code fences):

    {json.dumps(schema, indent=2)}

    Rules:
    - Questions should be concise, clear, and unambiguous.
    - Options must be plausible and similar in length.
    - answer_index must be an integer 0..3.
    - Keep explanations short (1-2 sentences) or empty if not needed.
    - If you cannot generate exactly {num_questions}, indicate that in the JSON by returning fewer questions (but still return valid JSON).

    Content:
    \"\"\"{content}\"\"\"

    Difficulty: {difficulty.capitalize()}
    """)
    return prompt


def generate_quiz_with_retries(content, num_questions=5, difficulty="medium", model_name="gemini-1.5-flash", max_retries=3, wait_s=1.0):
    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            prompt = build_prompt(content, num_questions, difficulty)
            model = genai.GenerativeModel(model_name)
            # Try a generation call. We keep it simple; if your client supports more options, you can add temperature/max_tokens etc.
            resp = model.generate_content(prompt)
            raw = resp.text if getattr(resp, "text", None) else str(resp)
            last_raw = raw

            parsed = extract_json_from_text(raw)
            if parsed and is_valid_quiz(parsed, num_questions):
                # normalize to object with 'questions'
                if isinstance(parsed, list):
                    parsed = {"quiz_title": f"Quiz on provided content", "difficulty": difficulty.capitalize(), "questions": parsed}
                # Ensure each question has 'explanation' and answer_index numeric
                for q in parsed.get("questions", []):
                    if "explanation" not in q:
                        q["explanation"] = ""
                    # if 'answer' string provided instead of index, convert
                    if "answer" in q and "answer_index" not in q:
                        # find index of the option that equals the answer text
                        try:
                            q["answer_index"] = q["options"].index(q["answer"])
                        except Exception:
                            # try fuzzy match
                            q["answer_index"] = 0
                    # ensure exactly 4 options (if fewer, pad)
                    opts = q.get("options", [])
                    if len(opts) < 4:
                        opts += ["(option)"] * (4 - len(opts))
                        q["options"] = opts[:4]
                return parsed, raw
            else:
                # parsing failed or schema mismatch: try again
                st.experimental_log(f"Attempt {attempt}: parsing failed or schema mismatch.")
                time.sleep(wait_s * attempt)
        except Exception as e:
            # network / 404 / auth / etc.
            st.warning(f"Attempt {attempt} failed: {type(e).__name__}: {e}")
            time.sleep(wait_s * attempt)
            last_raw = str(e)

    # all attempts failed — return None + last raw text for debugging
    return None, last_raw


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Teacher Assistant (Gemini)", layout="wide")
st.title("📝 AI Teacher Assistant — Gemini MCQ Generator")
st.markdown("Upload a PDF or paste text, pick difficulty and number of questions, then generate an interactive MCQ quiz.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-mini"], index=0)
    num_questions = st.slider("Number of questions", 1, 12, 5)
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    max_retries = st.slider("Max Gemini retries", 1, 5, 3)
    show_raw_on_error = st.checkbox("Show raw model text when parsing fails", value=True)

uploaded_pdf = st.file_uploader("Upload PDF (optional)", type=["pdf"])
text_input = st.text_area("Or paste topic / passage (optional)", height=200)
generate_btn = st.button("✨ Generate Quiz")

# session state initialization
if "quiz_obj" not in st.session_state:
    st.session_state.quiz_obj = None
if "raw_model" not in st.session_state:
    st.session_state.raw_model = ""
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "completed" not in st.session_state:
    st.session_state.completed = False

if generate_btn:
    final_text = ""
    if uploaded_pdf is not None:
        final_text = extract_text_from_pdf(uploaded_pdf, max_pages=10)
    elif text_input and text_input.strip():
        final_text = text_input.strip()
    else:
        st.error("Please upload a PDF or paste some text to generate a quiz.")
        st.stop()

    with st.spinner("Contacting Gemini and parsing response..."):
        quiz_obj, raw = generate_quiz_with_retries(final_text, num_questions=num_questions,
                                                  difficulty=difficulty, model_name=model_name,
                                                  max_retries=max_retries)
        st.session_state.raw_model = raw or ""
        if quiz_obj:
            st.success("Quiz generated by Gemini!")
            # normalize to expected structure: dict with 'questions' list
            st.session_state.quiz_obj = quiz_obj
            st.session_state.answers = {}
            st.session_state.current_q = 0
            st.session_state.completed = False
        else:
            st.error("Failed to generate a valid JSON quiz from Gemini after retries.")
            if show_raw_on_error and st.session_state.raw_model:
                st.subheader("Raw Gemini output (for debugging)")
                st.code(st.session_state.raw_model[:4000])
            st.info("Falling back to a simple local generator so you can still take a quiz.")
            fallback = rule_based_fallback(final_text, num_questions=num_questions, difficulty=difficulty)
            st.session_state.quiz_obj = fallback
            st.session_state.answers = {}
            st.session_state.current_q = 0
            st.session_state.completed = False

# If a quiz exists, show interactive UI
if st.session_state.quiz_obj:
    quiz = st.session_state.quiz_obj
    questions = quiz.get("questions", [])
    total = len(questions)
    st.markdown(f"### {quiz.get('quiz_title','Generated Quiz')}  — Difficulty: {quiz.get('difficulty','')}")
    progress = int((st.session_state.current_q / max(1, total)) * 100)
    st.progress(progress)

    # show current question
    idx = st.session_state.current_q
    q = questions[idx]
    st.write(f"**Q{idx+1}. {q.get('question','(no question)')}**")

    # options displayed as radio, but we must show labels A-D
    options = q.get("options", [])
    # ensure 4 options
    if len(options) < 4:
        options = options + ["(option)"] * (4 - len(options))
    labeled_options = [f"{label}) {opt}" for label, opt in zip(['A','B','C','D'], options)]

    # default selection if exists
    existing = st.session_state.answers.get(idx, None)
    choice = st.radio("Choose one", labeled_options, index=existing if existing is not None else 0, key=f"q_{idx}")

    # map choice to index 0..3
    selected_index = labeled_options.index(choice)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Previous") and idx > 0:
            st.session_state.answers[idx] = selected_index
            st.session_state.current_q = idx - 1
            st.experimental_rerun()
    with col2:
        if st.button("Next") and idx < total - 1:
            st.session_state.answers[idx] = selected_index
            st.session_state.current_q = idx + 1
            st.experimental_rerun()
    with col3:
        if st.button("Submit Quiz"):
            st.session_state.answers[idx] = selected_index
            # compute score
            score = 0
            for i, qq in enumerate(questions):
                correct_idx = qq.get("answer_index")
                # if answer_index missing but 'answer' text present, try to compute
                if correct_idx is None and qq.get("answer") and qq.get("options"):
                    try:
                        correct_idx = qq["options"].index(qq["answer"])
                    except Exception:
                        correct_idx = 0
                chosen = st.session_state.answers.get(i, None)
                if chosen is not None and correct_idx is not None and chosen == correct_idx:
                    score += 1
            st.session_state.completed = True
            st.session_state.score = score

    # show small hint / explanation toggle
    if st.checkbox("Show explanation for this question"):
        st.info(q.get("explanation", ""))

    st.caption(f"Question {idx+1} of {total}")

    # After completion: show results and breakdown
    if st.session_state.completed:
        st.success(f"Your score: {st.session_state.score}/{total}")
        st.markdown("### Detailed feedback")
        for i, qq in enumerate(questions):
            correct = qq.get("answer_index", 0)
            opts = qq.get("options", [])
            chosen = st.session_state.answers.get(i, None)
            chosen_text = opts[chosen] if chosen is not None and chosen < len(opts) else "(no answer)"
            correct_text = opts[correct] if correct is not None and correct < len(opts) else "(unknown)"
            st.write(f"**Q{i+1}.** {qq.get('question','')}")
            st.write(f"- Your answer: {chosen_text}")
            st.write(f"- Correct answer: {correct_text}")
            if qq.get("explanation"):
                st.write(f"- Explanation: {qq.get('explanation')}")
            st.write("---")

        if st.button("Try again / Generate new quiz"):
            st.session_state.quiz_obj = None
            st.session_state.answers = {}
            st.session_state.current_q = 0
            st.session_state.completed = False
            st.session_state.raw_model = ""
            st.experimental_rerun()

else:
    st.info("No quiz yet. Upload a PDF or paste text, choose settings in the sidebar, then click Generate Quiz.")
    st.write("Tip: If Gemini frequently returns invalid JSON, try reducing the number of questions or switching the model to 'gemini-1.5-flash' or 'gemini-1.5-mini'.")

