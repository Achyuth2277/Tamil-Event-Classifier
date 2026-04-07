"""Tamil to English translation helpers."""

from __future__ import annotations

from typing import Optional

from preprocess import clean_tamil_text


def translate_tamil_to_english(
    tamil_text: str,
    translator: Optional[object] = None,
) -> str:
    """
    Translate Tamil text into English.

    An empty string is returned when no live translator is available so callers
    can distinguish "no translation" from "Tamil text echoed back".
    """
    cleaned_text = clean_tamil_text(tamil_text)
    if not cleaned_text:
        return ""

    try:
        if translator is not None:
            translated = translator.translate(cleaned_text, src="ta", dest="en")
            return translated.text.strip()

        try:
            from deep_translator import GoogleTranslator  # type: ignore

            translated_text = GoogleTranslator(source="ta", target="en").translate(cleaned_text)
            return translated_text.strip() if translated_text else ""
        except Exception:
            pass

        from googletrans import Translator  # type: ignore

        active_translator = Translator()
        translated = active_translator.translate(cleaned_text, src="ta", dest="en")
        return translated.text.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    sample = "தமிழ்நாட்டில் கடும் மழை"
    print("Tamil   :", sample)
    print("English :", translate_tamil_to_english(sample))
