import os
import zipfile

count = 1
while os.path.exists("../results/result" + format(count, "02")):
	count += 1
folder = "../results/result" + format(count, "02")
os.mkdir(folder)
zipf = zipfile.ZipFile(folder + "/project.zip", "w", zipfile.ZIP_DEFLATED)
for dirname, subdirs, files in os.walk(".."):
	if not dirname[3:].startswith("results"):
		zipf.write(dirname)
		for filename in files:
			zipf.write(dirname + "/" + filename)
zipf.close()
open(folder + "/details.txt", "w").close()
os.startfile(os.path.abspath(folder + "/details.txt"))
