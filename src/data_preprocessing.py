import os
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

class ViQuADCleaner:
    """Lớp đảm nhận khâu sửa lỗi lệch vị trí answer_start (Offset Relocation)"""
    def __init__(self, search_window: int = 20):
        self.search_window = search_window

    def clean(self, raw_data: dict) -> dict:
        fixed, dropped = 0, 0
        dropped_examples = []

        for art in raw_data["data"]:
            for para in art["paragraphs"]:
                ctx = para["context"]
                kept_qas = []
                for qa in para["qas"]:
                    answers = qa.get("answers", [])
                    if not answers:
                        kept_qas.append(qa)
                        continue

                    new_answers = []
                    bad = False
                    for a in answers:
                        start = a["answer_start"]
                        text = a["text"]
                        extracted = ctx[start:start + len(text)]

                        if extracted == text:
                            new_answers.append(a)
                            continue

                        # Relocate answer position within search_window
                        lo = max(0, start - self.search_window)
                        hi = min(len(ctx), start + len(text) + self.search_window)
                        window = ctx[lo:hi]
                        idx = window.find(text)
                        
                        if idx != -1:
                            new_start = lo + idx
                            new_answers.append({"answer_start": new_start, "text": text})
                            fixed += 1
                        else:
                            bad = True

                    if bad:
                        dropped += 1
                        dropped_examples.append({"id": qa["id"], "question": qa["question"]})
                        continue

                    qa["answers"] = new_answers
                    kept_qas.append(qa)

                para["qas"] = kept_qas

        print(f" [Clean Offset] Đã tự động sửa vị trí answer_start cho {fixed} câu hỏi.")
        print(f" [Clean Offset] Đã loại bỏ {dropped} câu hỏi do không tìm thấy đáp án khớp.")
        return raw_data


class ViQuADPreprocessor:
    """Lớp quản lý Pipeline Tiền xử lý Dữ liệu: Load -> Clean -> Flatten -> Stats -> Split -> Save"""
    def __init__(self, raw_path: str, output_dir: str, search_window: int = 20):
        self.raw_path = raw_path
        self.output_dir = output_dir
        self.cleaner = ViQuADCleaner(search_window=search_window)
        os.makedirs(self.output_dir, exist_ok=True)

    def _flatten_json(self, data: dict) -> pd.DataFrame:
        samples = []
        for article in data['data']:
            title = article.get('title', '')
            for p_idx, paragraph in enumerate(article['paragraphs']):
                context = paragraph['context']
                context_id = f"{title}_{p_idx}"

                for qa in paragraph['qas']:
                    answers = qa.get('answers', [])
                    if len(answers) > 0:
                        ans_text = answers[0]['text']
                        ans_start = answers[0]['answer_start']
                    else:
                        ans_text = ""
                        ans_start = -1

                    samples.append({
                        "qa_id": qa['id'],
                        "context_id": context_id,
                        "title": title,
                        "context": context,
                        "question": qa['question'],
                        "answer_text": ans_text,
                        "answer_start": ans_start,
                        "is_impossible": qa.get('is_impossible', False)
                    })
        return pd.DataFrame(samples)

    def log_statistics(self, df: pd.DataFrame) -> None:
        print("\n--- THỐNG KÊ CHI TIẾT DỮ LIỆU ĐÃ LÀM SẠCH ---")
        print(f"- Tổng số mẫu câu hỏi     : {len(df)}")
        print(f"- Số đoạn văn (Contexts)  : {df['context_id'].nunique()}")
        print(f"- Số câu hỏi Có Đáp Án    : {(~df['is_impossible']).sum()} ({(1-df['is_impossible'].mean())*100:.1f}%)")
        print(f"- Số câu hỏi Không Đáp Án : {df['is_impossible'].sum()} ({df['is_impossible'].mean()*100:.1f}%)")

        ctx_lens = df['context'].apply(lambda x: len(str(x).split()))
        q_lens = df['question'].apply(lambda x: len(str(x).split()))

        print(f"- Độ dài Context (từ)    : Min={ctx_lens.min()}, Mean={ctx_lens.mean():.1f}, Max={ctx_lens.max()}")
        print(f"- Độ dài Question (từ)   : Min={q_lens.min()}, Mean={q_lens.mean():.1f}, Max={q_lens.max()}")

    def split_dataset(self, df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, seed=42):
        test_ratio = 1.0 - train_ratio - val_ratio
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df['context_id']))

        df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        val_relative_ratio = val_ratio / (train_ratio + val_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_ratio, random_state=seed)
        train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['context_id']))

        df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
        df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

        return df_train, df_val, df_test

    def run_pipeline(self):
        print(f"1. Đang đọc dữ liệu gốc từ: {self.raw_path}")
        with open(self.raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        print("\n2. Đang quét và sửa lỗi lệch vị trí (relocate offsets)...")
        cleaned_data = self.cleaner.clean(raw_data)

        print("\n3. Đang chuyển đổi cấu trúc JSON thành dạng bảng (Flatten)...")
        df = self._flatten_json(cleaned_data)

        self.log_statistics(df)

        print("\n4. Tiến hành chia tập Train (80%) - Val (10%) - Test (10%) theo Group Context...")
        df_train, df_val, df_test = self.split_dataset(df)

        print(f"   - Train set : {len(df_train)} mẫu ({df_train['context_id'].nunique()} contexts)")
        print(f"   - Val set   : {len(df_val)} mẫu ({df_val['context_id'].nunique()} contexts)")
        print(f"   - Test set  : {len(df_test)} mẫu ({df_test['context_id'].nunique()} contexts)")

        df_train.to_parquet(os.path.join(self.output_dir, "train.parquet"), index=False)
        df_val.to_parquet(os.path.join(self.output_dir, "val.parquet"), index=False)
        df_test.to_parquet(os.path.join(self.output_dir, "test.parquet"), index=False)
        print(f"\n5. Hoàn tất! Tất cả dữ liệu sạch đã lưu tại thư mục `{self.output_dir}/`.")


if __name__ == "__main__":
    preprocessor = ViQuADPreprocessor(
        raw_path="data/raw/train_formatted.json",
        output_dir="data/processed",
        search_window=20
    )
    preprocessor.run_pipeline()