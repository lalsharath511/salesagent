import asyncio
import base64
import json
import os
import subprocess
import tempfile
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
from pynput import keyboard
from faster_whisper import WhisperModel
import pygame
from termcolor import colored
import websockets
import anthropic
import openai
import logging
from openai import AsyncOpenAI


# openai_key="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"
# ELEVENLABS_API_KEY="202df1045c34b743dc873202612f31ee"
# Define API keys and voice ID
ELEVENLABS_API_KEY = '202df1045c34b743dc873202612f31ee'
VOICE_ID = 'jsCqWAovK2LkecY7zXl4'
OPENAI_API_KEY = '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'

class FasterWhisperTranscriber:
    def __init__(self, model_size="large-v3", sample_rate=44100):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.is_recording = False

    def on_press(self, key):
        if key == keyboard.Key.space:
            if not self.is_recording:
                self.is_recording = True
                print("Recording started.")
    
    def on_release(self, key):
        if key == keyboard.Key.space:
            if self.is_recording:
                self.is_recording = False
                print("Recording stopped.")
                return False

    def record_audio(self):
        recording = np.array([], dtype='float64').reshape(0, 2)
        frames_per_buffer = int(self.sample_rate * 0.1)
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            while True:
                if self.is_recording:
                    chunk = sd.rec(frames_per_buffer, samplerate=self.sample_rate, channels=2, dtype='float64')
                    sd.wait()
                    recording = np.vstack([recording, chunk])
                if not self.is_recording and len(recording) > 0:
                    break
            listener.join()
        return recording

    def save_temp_audio(self, recording):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        write(temp_file.name, self.sample_rate, recording)
        return temp_file.name
    
    def transcribe_audio(self, file_path):
        segments, info = self.model.transcribe(file_path, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        full_transcription = ""
        for segment in segments:
            print(segment.text)
            full_transcription += segment.text + " "
        os.remove(file_path)
        return full_transcription, info  

    # def transcribe_audio(self, file_path):
    #     segments, info = self.model.transcribe(file_path, beam_size=5)
    #     print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    #     full_transcription = ""
    #     for segment in segments:
    #         print(segment.text)
    #         full_transcription += segment.text + " "
    #     os.remove(file_path)
    #     return full_transcription


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set OpenAI API key
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# def is_installed(lib_name):
#     return shutil.which(lib_name) is not None


async def text_chunker(chunks):
    """Split text into chunks, ensuring to not break sentences."""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""

    async for text in chunks:
        if text is None:  # Check if text is None and continue to the next iteration if so
            continue

        if buffer.endswith(tuple(splitters)):
            yield buffer + " "
            buffer = text
        elif text.startswith(tuple(splitters)):
            yield buffer + text[0] + " "
            buffer = text[1:]
        else:
            buffer += text

    if buffer:
        yield buffer + " "



async def stream(audio_stream):
    """Stream audio data using mpv player."""
    # if not is_installed("mpv"):
    #     raise ValueError(
    #         "mpv not found, necessary to stream audio. "
    #         "Install instructions: https://mpv.io/installation/"
    #     )

    mpv_process = subprocess.Popen(
        ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    print("Started streaming audio")
    async for chunk in audio_stream:
        if chunk:
            mpv_process.stdin.write(chunk)
            mpv_process.stdin.flush()

    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()


async def text_to_speech_input_streaming(voice_id, text_iterator):
    """Send text to ElevenLabs API and stream the returned audio."""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1"

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
            "xi_api_key": ELEVENLABS_API_KEY,
        }))

        async def listen():
            """Listen to the websocket for audio data and stream it."""
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if data.get("audio"):
                        yield base64.b64decode(data["audio"])
                    elif data.get('isFinal'):
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break

        listen_task = asyncio.create_task(stream(listen()))

        async for text in text_chunker(text_iterator):
            await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

        await websocket.send(json.dumps({"text": ""}))

        await listen_task


# async def chat_completion(query):
#     """Retrieve text from OpenAI and pass it to the text-to-speech function."""
#     response = await aclient.chat.completions.create(model='gpt-3.5-turbo', messages=[{'role': 'user', 'content': query}],
#     temperature=1, stream=True)

#     async def text_iterator():
#         async for chunk in response:
#             delta = chunk.choices[0].delta
#             yield delta.content

#     await text_to_speech_input_streaming(VOICE_ID, text_iterator())

async def handle_recording():
    whisper = FasterWhisperTranscriber()
    english_system_message  = {
            "role": "system",
            "content": """Never forget your name is Sally. You work as a Sales Representative.
You work at company named Kno2gether Spelled as Know Together. Kno2gether's business is the following: Kno2gether is an AI Automation Agency and help customers automate their business processes using AI.
You are contacting a potential prospect in order to find out whether customer is looking for an AI Automation for their business by buying our subscription services
Customer's means of contacting the you is through phone call

Kno2gether AI Development Subscription Package -
Starts at 3999 Dollar per month where customers can request unlimited number of AI Design/Development task through Trello Dashboard. Average Time to complete one task is 4 days.

Keep your responses in short length to retain the user's attention. Never produce lists, just answers.
Start the conversation by just a greeting and explain that you are an AI Assistant and contacting the prospect as customer filled up the Contact Form in the Website.
Your Main Target outcome is to get qualified prospect customer agreed into an one-to-one meeting through Calendly meeting invite using email address user have already provided in the website.
Always confirm your understanding of prospect's requirement before you provide response as part of qualifying the customer and their requirements.
When the conversation is over, say "Thank You For Calling. Hope you have a good day"
Always and you MUST think about at WHICH CONVERSATION STAGE you are at, before answering SPECIFIC and RELEVANT to that CONVERSATION STAGE:

1: Introduction: Start the conversation by introducing yourself and your company and ask user about their company and AI Usecase which Kno2gether can solve. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming.
2: Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.
3: Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.
4: Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.
5: Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.
6: Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.
7: Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.
8: End conversation: The prospect has to leave to call, the prospect is not interested, or next steps where already determined by the sales agent.
            """
        }

    german_system_message  = {
            "role": "system",
            "content": """Vergessen Sie niemals, dass Ihr Name Sally ist. Sie arbeiten als Vertriebsmitarbeiter. Sie sind bei einem Unternehmen namens Kno2gether beschäftigt, das als Know Together geschrieben wird. Das Geschäft von Kno2gether ist wie folgt: Kno2gether ist eine Agentur für KI-Automatisierung und hilft Kunden dabei, ihre Geschäftsprozesse mit KI zu automatisieren. Sie kontaktieren einen potenziellen Interessenten, um herauszufinden, ob der Kunde nach einer KI-Automatisierung für sein Geschäft sucht, indem er unsere Abonnementdienste kauft. Kunden können Sie über Telefonanrufe kontaktieren.

Kno2gether KI-Entwicklungs-Abonnementpaket -
Beginnt bei 3999 Dollar pro Monat, wo Kunden eine unbegrenzte Anzahl von KI-Design-/Entwicklungsaufgaben über das Trello-Dashboard anfordern können. Die durchschnittliche Zeit zur Fertigstellung einer Aufgabe beträgt 4 Tage.

Halten Sie Ihre Antworten kurz, um die Aufmerksamkeit des Nutzers zu behalten. Erstellen Sie niemals Listen, sondern nur Antworten. Beginnen Sie das Gespräch mit einer Begrüßung und erklären Sie, dass Sie ein KI-Assistent sind und den Interessenten kontaktieren, da dieser das Kontaktformular auf der Website ausgefüllt hat. Ihr Hauptziel ist es, einen qualifizierten Interessenten dazu zu bringen, einem persönlichen Treffen durch eine Calendly-Einladung zuzustimmen, wobei die E-Mail-Adresse verwendet wird, die bereits auf der Website angegeben wurde. Bestätigen Sie immer Ihr Verständnis für die Anforderungen des Interessenten, bevor Sie antworten, als Teil der Qualifizierung des Kunden und seiner Anforderungen. Wenn das Gespräch beendet ist, sagen Sie "Danke für Ihren Anruf. Ich hoffe, Sie haben einen guten Tag." 
Immer und Sie MÜSSEN darüber nachdenken, in WELCHER GESPRÄCHSPHASE Sie sich befinden, bevor Sie SPEZIFISCH und RELEVANT auf diese GESPRÄCHSPHASE antworten:

Einführung: Beginnen Sie das Gespräch, indem Sie sich und Ihr Unternehmen vorstellen und den Nutzer nach seinem Unternehmen und dem KI-Anwendungsfall fragen, den Kno2gether lösen kann. Seien Sie höflich und respektvoll, während Sie den Ton des Gesprächs professionell halten. Ihre Begrüßung sollte einladend sein.
Qualifikation: Qualifizieren Sie den Interessenten, indem Sie bestätigen, ob er die richtige Person ist, mit der Sie über Ihr Produkt/Ihre Dienstleistung sprechen können. Stellen Sie sicher, dass sie die Autorität haben, Kaufentscheidungen zu treffen.
Wertvorschlag: Erklären Sie kurz, wie Ihr Produkt/Ihre Dienstleistung dem Interessenten nutzen kann. Konzentrieren Sie sich auf die Alleinstellungsmerkmale und den Wertvorschlag Ihres Produkts/Ihrer Dienstleistung, der es von den Wettbewerbern abhebt.
Bedarfsanalyse: Stellen Sie offene Fragen, um die Bedürfnisse und Schmerzpunkte des Interessenten zu ermitteln. Hören Sie sorgfältig auf deren Antworten und machen Sie Notizen.
Lösungspräsentation: Präsentieren Sie Ihr Produkt/Ihre Dienstleistung als Lösung, die ihre Schmerzpunkte ansprechen kann, basierend auf den Bedürfnissen des Interessenten.
Einwandbehandlung: Gehen Sie auf alle Einwände ein, die der Interessent bezüglich Ihres Produkts/Ihrer Dienstleistung haben könnte. Seien Sie bereit, Beweise oder Testimonials zur Unterstützung Ihrer Behauptungen vorzulegen.
Abschluss: Fragen Sie nach dem Verkauf, indem Sie den nächsten Schritt vorschlagen. Dies könnte eine Demo, ein Test oder ein Treffen mit Entscheidungsträgern sein. Stellen Sie sicher, dass Sie zusammenfassen, was besprochen wurde, und die Vorteile erneut hervorheben.
Gesprächsende: Der Interessent muss das Gespräch beenden, der Interessent ist nicht interessiert, oder die nächsten Schritte wurden bereits vom Vertriebsmitarbeiter bestimmt.
            """
        }



    while True:
        try:
            print("\nPress and hold the spacebar to start recording...")
            recording = whisper.record_audio()
            file_path = whisper.save_temp_audio(recording)
            full_transcript, info = whisper.transcribe_audio(file_path)
                        # Select the system message based on detected language
            if info.language == "de":
                message_history = [german_system_message]
            else:
                message_history = [english_system_message]

            message_history.append({"role": "user", "content": full_transcript})
            # Take user input from the terminal instead of recording audio
            # user_input = input("\nEnter your message (or type 'exit' to quit): ")
            # if user_input.lower() == 'exit':
            #     print("\nExiting...")
            #     break
            
            # message_history.append({"role": "user", "content": user_input})
            
            response = await aclient.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=message_history,
                temperature=0.5,
                stream=True
            )

            assistant_response = ""

            async def text_iterator():
                async for chunk in response:
                    delta_content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                    print(colored(delta_content, "green"), end="", flush=True)
                    nonlocal assistant_response
                    assistant_response += delta_content
                    # logging.info(delta_content)
                    yield delta_content
            
            try:
                await text_to_speech_input_streaming(VOICE_ID, text_iterator())
            except asyncio.CancelledError:
                pass

            message_history.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\nExiting due to KeyboardInterrupt...")
            break


# Main execution
if __name__ == "__main__":
    asyncio.run(handle_recording())


