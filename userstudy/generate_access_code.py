import uuid
import os

if os.path.exists("access_code"):
    exit(0)

f = open("access_code", "w")

stringLength = 16
n_group = 8
n_people = 15

for i in range(n_group):
    s = [uuid.uuid4().hex[0:stringLength] for i in range(n_people)]
    f.write(",".join(s) + "\n")