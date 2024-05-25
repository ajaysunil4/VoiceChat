import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("SPEECH_APIKEY"), region=os.getenv("SPEECH_REGION"))
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    while True:
        a = input("Press enter to speak")
        print("Listening... (say 'good bye' to end)")

        # Perform recognition
        speech_recognition_result = speech_recognizer.recognize_once_async().get()

        # Check the result
        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = speech_recognition_result.text
            print("Recognized: {}".format(recognized_text))

            # Break the loop if the user says "goodbye"
            if "goodbye" in recognized_text.lower():
                print("Stopping the Voice Transcript.")
                break

        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")
                break

if __name__ == "__main__":
    recognize_from_microphone()
