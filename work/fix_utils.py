import sys
import re

path = sys.argv[1]
with open(path, 'rb') as f:
    data = f.read()

escaped = b'\\"'
plain = b'"'
count = data.count(escaped)
if count > 0:
    data = data.replace(escaped, plain)
    with open(path, 'wb') as f:
        f.write(data)
    print(f'Fixed {count} occurrences in {path}')
else:
    print(f'No escaped quotes found in {path}')
