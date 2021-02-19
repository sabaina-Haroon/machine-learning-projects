import os

input_name = []

path = 'Input/Images'

names = os.listdir(path)

for n in names:
    input_name.append(n)

with open('input_names.txt', 'w') as f:
    for item in input_name:
        f.write("%s\n" % item)