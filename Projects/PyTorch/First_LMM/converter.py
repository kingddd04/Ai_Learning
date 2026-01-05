"""
import quopri

with open("C:\\Users\\david_bbnm\\Downloads\\r.txt", "r", encoding="utf-8") as f:
    raw = f.read()
path = "C:\\Users\\david_bbnm\\Downloads\\r.txt"
decoded = quopri.decodestring(raw).decode("utf-8")

with open("C:\\Users\\david_bbnm\\Downloads\\c.txt", "w", encoding="utf-8") as f:
    f.write(decoded)
"""

with open("C:\\Users\\david_bbnm\\Downloads\\cpdv.txt", "r", encoding="utf8")  as txt:
    rows = txt.readlines()

clean_rows = []
for row in rows:
    ref, verse_text = row.split("\t", 1)
    clean_rows.append(verse_text)
    

with open("C:\\Users\\david_bbnm\\Downloads\\bible clean.txt", "w", encoding="utf8")  as uhh:
    uhh.writelines(clean_rows)

print("done")
