import os
styles = ["starrynight", "lamuse", "feathers", "composition"]
for s in styles:
    print(s)
    os.system(f"python parsecsv.py \"{s}\"")