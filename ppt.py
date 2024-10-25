from tika import parser
parsed = parser.from_file("/content/drive/MyDrive/ML PROJECT/Golden Feathers-pitch_MR.pdf")
text = parsed["content"]
print(text)