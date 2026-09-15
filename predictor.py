from pathlib import Path
import html
import json
import re
import unicodedata
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "model"

VOCABULARIES = [
    {
        "urgent", "urgently", "immediately", "immediate",
        "now", "today", "asap", "quickly", "deadline",
        "expire", "expired", "final", "action",
    },
    {
        "password", "passwd", "username", "login",
        "credential", "credentials", "verify", "verification",
        "authenticate", "authentication", "account",
    },
    {
        "suspend", "suspended", "terminate", "terminated",
        "blocked", "block", "close", "closed", "penalty",
        "fraud", "unauthorized", "warning", "security",
    },
    {
        "payment", "pay", "invoice", "money", "bank",
        "transfer", "transaction", "refund", "credit",
        "debit", "fee", "account", "billing",
    },
    {
        "click", "clicking", "visit", "open", "download",
        "confirm", "verify", "submit", "update",
        "activate", "login",
    },
]


def preprocess_email(text):
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        char for char in text
        if char in "\n\t"
        or not unicodedata.category(char).startswith("C")
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_nlp_features(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    lowercase_words = [word.lower() for word in words]

    counts = [
        sum(word in vocabulary for word in lowercase_words)
        for vocabulary in VOCABULARIES
    ]

    uppercase_ratio = sum(
        len(word) > 1 and word.isupper() for word in words
    ) / max(len(words), 1)

    counts.extend([
        len(re.findall(
            r"https?://\S+|www\.\S+", text, re.IGNORECASE
        )),
        len(re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text,
        )),
        len(re.findall(r"(?:\+?\d[\d\s().-]{7,}\d)", text)),
        text.count("!"),
        text.count("?"),
        uppercase_ratio,
        np.log1p(len(text)),
    ])

    return np.array(counts, dtype=np.float32)


class GatedHybridPhishingModel(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()

        self.transformer = AutoModel.from_config(backbone_config)

        self.context_projection = nn.Sequential(
            nn.Linear(backbone_config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask, nlp_features):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        context = self.context_projection(
            outputs.last_hidden_state[:, 0, :]
        )
        features = self.feature_projection(nlp_features.float())
        gate = self.gate(torch.cat([context, features], dim=1))
        fused = gate * context + (1 - gate) * features
        return self.classifier(self.fusion(fused))


class PhishingPredictor:
    def __init__(self):
        self.settings = json.loads(
            (MODEL_DIR / "model_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.mean = np.load(
            MODEL_DIR / "feature_mean.npy", allow_pickle=False
        )
        self.std = np.load(
            MODEL_DIR / "feature_std.npy", allow_pickle=False
        )

        if self.mean.shape != (12,) or self.std.shape != (12,):
            raise ValueError("Expected 12 feature normalisation values.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Invalid feature normalisation values.")
        if (self.std <= 0).any():
            raise ValueError("Feature standard deviations must be positive.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        config = AutoConfig.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )

        print("Loading saved model...")
        self.model = GatedHybridPhishingModel(config)

        weights = torch.load(
            MODEL_DIR / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        if "model_state_dict" in weights:
            weights = weights["model_state_dict"]
        elif "state_dict" in weights:
            weights = weights["state_dict"]

        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        print("Saved model loaded successfully.")

    def predict(self, text):
        cleaned = preprocess_email(text)
        if not cleaned:
            raise ValueError("Please provide non-empty email text.")

        features = extract_nlp_features(cleaned)
        normalised = (features - self.mean) / self.std

        tokens = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings["max_length"],
        )

        with torch.inference_mode():
            logits = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                nlp_features=torch.tensor(
                    normalised, dtype=torch.float32
                ).unsqueeze(0),
            )
            probabilities = torch.softmax(logits, dim=1)[0]

        label = int(probabilities.argmax().item())

        return {
            "label_id": label,
            "verdict": "phishing" if label == 1 else "legitimate",
            "score": round(float(probabilities[label].item()), 4),
            "class_scores": [
                float(probabilities[0].item()),
                float(probabilities[1].item()),
            ],
        }    

    def predict_probabilities(self, texts):
        results = [self.predict(text) for text in texts]
        return np.asarray(
            [result["class_scores"] for result in results],
            dtype=np.float64,
        )

    def predict_with_explanation(self, text):
        result = self.predict(text)

        explainer = LimeTextExplainer(
            class_names=["legitimate", "phishing"],
            random_state=42,
        )

        lime_result = explainer.explain_instance(
            text,
            self.predict_probabilities,
            labels=(result["label_id"],),
            num_features=8,
            num_samples=50,
        )

        terms = lime_result.as_list(label=result["label_id"])

        return {
            "label_id": result["label_id"],
            "verdict": result["verdict"],
            "score": result["score"],
            "explanation": {
                "method": "LIME",
                "class_explained": result["verdict"],
                "terms": [
                    {
                        "word": word,
                        "weight": round(float(weight), 6),
                    }
                    for word, weight in terms
                ],
                "num_samples": 50,
                "note": (
                    "Approximate local explanation. Positive weights "
                    "support the displayed verdict; negative weights "
                    "oppose it."
                ),
            },
        }
    # def predict_with_explanation(self, text, max_words=30, top_k=5):
    #     cleaned = preprocess_email(text)
    #     baseline = self.predict(cleaned)

    #     label = baseline["label_id"]
    #     baseline_score = baseline["class_scores"][label]

    #     # Check individual word occurrences in the beginning of the email.
    #     matches = list(re.finditer(r"\b[\w]+\b", cleaned))
    #     checked = matches[:max_words]

    #     contributions = []

    #     for match in checked:
    #         # Remove this occurrence, then recalculate all model inputs.
    #         modified = (
    #             cleaned[:match.start()]
    #             + " "
    #             + cleaned[match.end():]
    #         )

    #         if not modified.strip():
    #             continue

    #         changed = self.predict(modified)
    #         changed_score = changed["class_scores"][label]
    #         drop = baseline_score - changed_score

    #         if drop > 0:
    #             contributions.append({
    #                 "word": match.group(),
    #                 "start": match.start(),
    #                 "end": match.end(),
    #                 "score_drop_percentage_points": round(drop * 100, 6),
    #             })

    #     contributions.sort(
    #         key=lambda item: item["score_drop_percentage_points"],
    #         reverse=True,
    #     )

    #     strongest = contributions[:top_k]

    #     if strongest:
    #         summary = (
    #             "Removing these word occurrences reduced the model's "
    #             f"score for the {baseline['verdict']} prediction."
    #         )
    #     else:
    #         summary = (
    #             "No supporting word occurrences were identified "
    #             "among the words checked."
    #         )

    #     return {
    #         "label_id": label,
    #         "verdict": baseline["verdict"],
    #         "score": baseline["score"],
    #         "explanation": {
    #             "method": "word-removal sensitivity",
    #             "summary": summary,
    #             "influential_words": strongest,
    #             "words_checked": len(checked),
    #             "total_words": len(matches),
    #             "scope": (
    #                 "Approximate local explanation. Checks only the first "
    #                 f"{max_words} words and recalculates the NLP features "
    #                 "after each removal. Positions refer to cleaned text."
    #             ),
    #         },
    #     }    

    #     return {
    #         "label_id": label,
    #         "verdict": "phishing" if label == 1 else "legitimate",
    #         "score": round(float(probabilities[label].item()), 4),
    #         "class_scores": [
    #             float(probabilities[0].item()),
    #             float(probabilities[1].item()),
    #         ],
    #     }
      
        


if __name__ == "__main__":
    predictor = PhishingPredictor()
    email = input("\nPaste a short email and press Enter:\n")
    print(json.dumps(predictor.predict(email), indent=2))
