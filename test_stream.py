import re

_BOUNDARY = re.compile(r'[.!?\n\u3002]+\s+')

tokens = ["Hello ", "world", ".", ".", ".", " However, ", "this ", "is ", "a test.\n\n", "And", " more."]
sentence_buffer = ""
pending = []

for token in tokens:
    sentence_buffer += token
    match = _BOUNDARY.search(sentence_buffer)
    if match:
        end_idx = match.end()
        sentence = sentence_buffer[:end_idx]
        sentence_buffer = sentence_buffer[end_idx:]
        if sentence.strip():
            pending.append(sentence)

if sentence_buffer.strip():
    pending.append(sentence_buffer)

print("Pending sentences:")
for s in pending:
    print(repr(s))
print("Joined:", "".join(pending))
