import os

f = open('LEMON_exceptions.txt', 'w', encoding='utf-8')  # Ensure the output file handles utf-8
path = "LOGS/LEMON_result"
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith("FAILED.txt"):
            # Specify the encoding here
            with open(os.path.join(root, file), 'r', encoding='utf-8') as ft:
                content = ft.read()
                f.write(content)
f.close()  # It's good practice to close the file
