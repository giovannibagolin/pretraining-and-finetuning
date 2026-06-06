import sys
import json
import re
import random
from pathlib import Path

random.seed(42)


def process_text(text, filename):
    ref_matches = list(re.finditer(r"(?i)\breferences\b", text))
    if not ref_matches:
        print(f"{filename.name}: Removed 0.00%")
        return text

    last_ref = ref_matches[-1]
    ref_start = last_ref.start()
    total_len = len(text)

    if ref_start > 0.7 * total_len:
        thrown_out_len = total_len - ref_start
        if thrown_out_len <= 0.3 * total_len:
            print(f"{filename.name}: Removed {(thrown_out_len / total_len) * 100:.2f}%")
            return text[:ref_start]
    else:
        app_match = re.search(r"(?i)\bappendix\b", text[ref_start:])
        if app_match:
            app_start = ref_start + app_match.start()
            thrown_out_len = app_start - ref_start
            if thrown_out_len <= 0.3 * total_len:
                print(
                    f"{filename.name}: Removed {(thrown_out_len / total_len) * 100:.2f}%"
                )
                return text[:ref_start] + text[app_start:]

    print(f"{filename.name}: Removed 0.00%")
    return text


directory = Path(sys.argv[1])
all_files = list(directory.rglob("*.txt"))
num_files = len(all_files)
random.shuffle(all_files)

if num_files == 0:
    raise SystemExit(f"No .txt files found in {directory}")

# Use up to 50 validation files, but do not make the training split negative
# when running small test downloads.
if num_files <= 1:
    num_test_files = 0
elif num_files < 50:
    num_test_files = max(1, int(num_files * 0.2))
else:
    num_test_files = 50

num_train_files = num_files - num_test_files
print(f"Found files: {num_files}")
print(f"Train files: {num_train_files}, Test files: {num_test_files}")

val_path = f"cpt_val_dataset_{num_test_files}.jsonl"
train_path = f"cpt_train_dataset_{num_train_files}.jsonl"

with open(val_path, "w", encoding="utf-8") as f:
    for a in all_files[:num_test_files]:
        text = a.read_text(encoding="utf-8", errors="replace")
        content = {"text": process_text(text, a)}
        f.write(json.dumps(content, ensure_ascii=False) + "\n")

with open(train_path, "w", encoding="utf-8") as f:
    for a in all_files[num_test_files:]:
        text = a.read_text(encoding="utf-8", errors="replace")
        content = {"text": process_text(text, a)}
        f.write(json.dumps(content, ensure_ascii=False) + "\n")

print(f"Saved train dataset: {train_path}")
print(f"Saved validation dataset: {val_path}")
