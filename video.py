import moviepy.editor as mp
import speech_recognition as sr

video = mp.VideoFileClip("/content/drive/MyDrive/ML PROJECT/Microsoft global outage： “Impact in India much lesser compared to Western countries…” Dr Sanjay Bahl [777hpBZDCkQ].webm")
audio_file = video.audio
audio_file.write_audiofile("demo.wav")


r = sr.Recognizer()

with sr.AudioFile("demo.wav") as source:
	data = r.record(source)

text = r.recognize_google(data)
print(text)