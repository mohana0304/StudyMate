# # llm.py
# import os
# import textwrap

# # Attempt to provide two options:
# # 1) IBM Watsonx Mixtral (if user sets env vars)
# # 2) Local lightweight model (flan-t5-small) via transformers if installed
# # 3) Finally, a simple deterministic synthesizer (concatenate retrieved contexts)

# WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "")
# WATSONX_URL = os.getenv("WATSONX_URL", "")
# WATSONX_MODEL = os.getenv("WATSONX_MODEL", "mistralai/mixtral-8x7b-instruct-v0-1")

# USE_WATSONX = bool(WATSONX_APIKEY and WATSONX_URL)


# def synthesize_answer_from_context(question: str, contexts: list):
#     """
#     Basic, deterministic answer synthesis used as fallback.
#     It concatenates contexts and returns them with a header.
#     """
#     header = f"**SYNTHESIZED ANSWER (no LLM configured)**\n\nQuestion: {question}\n\n"
#     body = "\n\n---\n\n".join([f"Source: {c['doc_id']} (page {c['page']})\n\n{c['text']}" for c in contexts])
#     suggestion = ("\n\nNote: Configure an LLM (IBM Watsonx or local transformers) for a natural-language synthesized answer. "
#                   "Current response returns the most relevant extracted snippets.")
#     return header + body + suggestion


# # Try to import optional IBM client
# if USE_WATSONX:
#     try:
#         # The real IBM watsonx client may have different import paths/usage; below is a placeholder.
#         # Replace with your own IBM client usage as needed.
#         from ibm_watsonx_ai.foundation_models import Model
#         def ask_watsonx(prompt: str, max_new_tokens: int = 300):
#             model = Model(
#                 model_id=WATSONX_MODEL,
#                 credentials={"apikey": WATSONX_APIKEY, "url": WATSONX_URL}
#             )
#             resp = model.generate_text(prompt=prompt, max_new_tokens=max_new_tokens)
#             # The structure depends on SDK version
#             text = resp.get("results", [{}])[0].get("generated_text", "")
#             return text
#         LLM_AVAILABLE = True
#         LLM_BACKEND = "watsonx"
#     except Exception as e:
#         print("Watsonx import failed:", e)
#         LLM_AVAILABLE = False
#         LLM_BACKEND = None
# else:
#     LLM_AVAILABLE = False
#     LLM_BACKEND = None

# # Try to use transformers fallback
# if not LLM_AVAILABLE:
#     try:
#         from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#         import torch

#         MODEL_NAME = os.getenv("FALLBACK_MODEL", "google/flan-t5-small")
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

#         def ask_local_flan(prompt: str, max_new_tokens: int = 200):
#             inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest").to(device)
#             out = model.generate(**inputs, max_new_tokens=max_new_tokens)
#             text = tokenizer.decode(out[0], skip_special_tokens=True)
#             return text

#         LLM_AVAILABLE = True
#         LLM_BACKEND = "flan"
#     except Exception as e:
#         # transformers or torch not installed / loaded
#         LLM_AVAILABLE = False
#         LLM_BACKEND = None


# def generate_answer(question: str, contexts: list, max_tokens: int = 300, style: str = "default"):
#     """
#     Build prompt and call LLM if available, otherwise return synthesized snippets.
#     contexts: list of metadata dicts with 'doc_id','page','text'
#     style: "default" | "bullet_points"
#     """
#     # Compact context string
#     ctx_texts = []
#     for idx, c in enumerate(contexts, start=1):
#         content = c.get("text", "")
#         header = f"[{idx}] Source: {c.get('doc_id')} (page {c.get('page')})"
#         ctx_texts.append(header + "\n" + content)

#     context_block = "\n\n".join(ctx_texts)

#     # Add style instructions
#     style_instruction = ""
#     if style == "bullet_points":
#         style_instruction = """
#         - Present the answer in short bullet points.
#         - Keep each point concise and clear.
#         - Use simple language.
#         """

#     prompt = textwrap.dedent(f"""
#         You are an expert assistant. Use ONLY the provided context to answer the question.
#         If the answer is not in the context, say "Answer not found in provided documents".

#         Context:
#         {context_block}

#         Question:
#         {question}

#         Instructions:
#         {style_instruction}

#         Answer (cite sources inline like [1], [2] referencing the context list):
#     """)

#     if LLM_AVAILABLE:
#         if LLM_BACKEND == "watsonx":
#             try:
#                 return ask_watsonx(prompt, max_new_tokens=max_tokens)
#             except Exception as e:
#                 print("Watsonx error:", e)
#                 return synthesize_answer_from_context(question, contexts)
#         elif LLM_BACKEND == "flan":
#             try:
#                 return ask_local_flan(prompt, max_new_tokens=max_tokens)
#             except Exception as e:
#                 print("Local flan error:", e)
#                 return synthesize_answer_from_context(question, contexts)

#     # fallback:
#     return synthesize_answer_from_context(question, contexts)

# llm.py
import os
import textwrap

# Environment variables for IBM Watsonx
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "")
WATSONX_URL = os.getenv("WATSONX_URL", "")
WATSONX_MODEL = os.getenv("WATSONX_MODEL", "mistralai/mixtral-8x7b-instruct-v0-1")

USE_WATSONX = bool(WATSONX_APIKEY and WATSONX_URL)


def synthesize_answer_from_context(question: str, contexts: list):
    """
    Simple deterministic fallback.
    Returns plain text so frontend can display in white.
    """
    if not contexts:
        return f"Answer not found in provided documents for: {question}"

    parts = []
    for c in contexts:
        doc_id = c.get("doc_id", "unknown")
        page = c.get("page", "?")
        text = c.get("text", "")
        parts.append(f"[{doc_id}, page {page}]: {text}")

    return "\n\n".join(parts)


# Watsonx setup
if USE_WATSONX:
    try:
        from ibm_watsonx_ai.foundation_models import Model

        def ask_watsonx(prompt: str, max_new_tokens: int = 300):
            model = Model(
                model_id=WATSONX_MODEL,
                credentials={"apikey": WATSONX_APIKEY, "url": WATSONX_URL}
            )
            resp = model.generate_text(prompt=prompt, max_new_tokens=max_new_tokens)
            return resp.get("results", [{}])[0].get("generated_text", "")

        LLM_AVAILABLE = True
        LLM_BACKEND = "watsonx"
    except Exception as e:
        print("Watsonx import failed:", e)
        LLM_AVAILABLE = False
        LLM_BACKEND = None
else:
    LLM_AVAILABLE = False
    LLM_BACKEND = None


# Transformers (Flan-T5) fallback
if not LLM_AVAILABLE:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        MODEL_NAME = os.getenv("FALLBACK_MODEL", "google/flan-t5-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

        def ask_local_flan(prompt: str, max_new_tokens: int = 200):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest").to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            return tokenizer.decode(out[0], skip_special_tokens=True)

        LLM_AVAILABLE = True
        LLM_BACKEND = "flan"
    except Exception as e:
        print("Flan load failed:", e)
        LLM_AVAILABLE = False
        LLM_BACKEND = None


def generate_answer(question: str, contexts: list, max_tokens: int = 300, style: str = "default"):
    """
    Main function: Build prompt and call Watsonx/Flan if available.
    Falls back to simple deterministic synthesis.
    """
    if not contexts:
        return "Answer not found in provided documents."

    # Build context
    ctx_texts = []
    for idx, c in enumerate(contexts, start=1):
        doc_id = c.get("doc_id", "unknown")
        page = c.get("page", "?")
        text = c.get("text", "")
        ctx_texts.append(f"[{idx}] {doc_id}, page {page}: {text}")

    context_block = "\n\n".join(ctx_texts)

    # Style
    style_instruction = ""
    if style == "bullet_points":
        style_instruction = (
            "- Present the answer in short bullet points.\n"
            "- Keep each point concise and clear.\n"
            "- Use simple language.\n"
        )

    prompt = textwrap.dedent(f"""
        You are an assistant. Use ONLY the provided context to answer the question.
        If the answer is not in the context, reply: "Answer not found in provided documents."

        Context:
        {context_block}

        Question:
        {question}

        {style_instruction}

        Answer:
    """)

    if LLM_AVAILABLE:
        try:
            if LLM_BACKEND == "watsonx":
                return ask_watsonx(prompt, max_new_tokens=max_tokens)
            elif LLM_BACKEND == "flan":
                return ask_local_flan(prompt, max_new_tokens=max_tokens)
        except Exception as e:
            print("LLM error:", e)

    # Fallback
    return synthesize_answer_from_context(question, contexts)
