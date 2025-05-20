import os
import json
from collections import defaultdict
import argparse
from transformers import AutoTokenizer


def load_docid_mapping(mapping_file):
    docid2name = {}
    with open(mapping_file, encoding="utf-8") as f:
        for line in f:
            docid, docname = line.strip().split()
            docid2name[docid] = docname
    return docid2name


def load_sentences(file_path):
    doc_sent_map = defaultdict(dict)
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                docname, sid, text = parts[0], int(parts[1]), parts[2]
                doc_sent_map[docname][sid] = text
    return doc_sent_map


def process_alignment_file(
    alignment_file,
    alignment_dir,
    mapping_dir,
    gold_dir,
    test_dir,
    output_dir,
    tokenizer,
):
    lang_pair = alignment_file.replace("_align_validation.tsv", "")
    mapping_path = os.path.join(mapping_dir, f"{lang_pair}_mapping.txt")
    docid2name = load_docid_mapping(mapping_path)

    src_lang = lang_pair.split("2")[0]
    tgt_lang = lang_pair.split("2")[1]

    src_file = f"medline_{lang_pair}_{src_lang}.txt"
    tgt_file = f"medline_{lang_pair}_{tgt_lang}.txt"

    src_path = os.path.join(test_dir, src_file)
    tgt_path = os.path.join(gold_dir, tgt_file)

    src_sents = load_sentences(src_path)
    tgt_sents = load_sentences(tgt_path)

    output_path = os.path.join(output_dir, f"{src_lang}-{tgt_lang}.jsonl")

    merged_docs = defaultdict(
        lambda: {
            f"{src_lang}_tokens": 0,
            f"{tgt_lang}_tokens": 0,
            f"{src_lang}_text": [],
            f"{tgt_lang}_text": [],
        }
    )

    with open(os.path.join(alignment_dir, alignment_file), encoding="utf-8") as f_align:

        for line in f_align:
            status, docid, src_seg, tgt_seg = line.strip().split("\t")
            if status != "OK":
                continue
            docname = docid2name.get(docid)
            if not docname:
                continue

            src_ids = list(map(int, src_seg.split(",")))
            tgt_ids = list(map(int, tgt_seg.split(",")))

            for s, t in zip(src_ids, tgt_ids):
                src_text = src_sents.get(docname, {}).get(s, "")
                tgt_text = tgt_sents.get(docname, {}).get(t, "")

                if src_text and tgt_text:
                    src_tokens = tokenizer(src_text, return_length=True)["length"][0]
                    tgt_tokens = tokenizer(tgt_text, return_length=True)["length"][0]

                    merged_docs[docname][f"{src_lang}_text"].append(src_text)
                    merged_docs[docname][f"{tgt_lang}_text"].append(tgt_text)
                    merged_docs[docname][f"{src_lang}_tokens"] += src_tokens
                    merged_docs[docname][f"{tgt_lang}_tokens"] += tgt_tokens

    with open(output_path, "w", encoding="utf-8") as fout:
        for docname, data in merged_docs.items():
            data[f"{src_lang}_text"] = " ".join(data[f"{src_lang}_text"])
            data[f"{tgt_lang}_text"] = " ".join(data[f"{tgt_lang}_text"])
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[✓] {lang_pair}: written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", type=str, required=True, help="Path to the base directory"
    )

    args = parser.parse_args()
    base_path = args.base_path

    alignment_dir = f"{base_path}/goldsets/alignment_files"
    mapping_dir = f"{base_path}/goldsets/docid_mapping_files"
    gold_dir = f"{base_path}/goldsets/gold_test_files"
    test_dir = f"{base_path}/testsets"
    output_dir = f"{base_path}/doc_level"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

    for fname in sorted(os.listdir(alignment_dir)):
        if fname.endswith("_align_validation.tsv"):
            process_alignment_file(
                fname,
                alignment_dir,
                mapping_dir,
                gold_dir,
                test_dir,
                output_dir,
                tokenizer,
            )


if __name__ == "__main__":
    main()
