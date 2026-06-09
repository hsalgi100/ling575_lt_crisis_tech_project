




def detect_language(text: str) -> translate.DetectedLanguage:
    client = translate.TranslationServiceClient()

    response = client.detect_language(parent=PARENT, content=text)

    return response.languages[0]