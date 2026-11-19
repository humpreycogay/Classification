#!/usr/bin/env python3
"""Generate the static model guide pages in this directory from the runner registry.

    python3 models/generate_pages.py          # writes models/index.html and one page per registry entry

Sample outputs come from models/samples.json, which models/capture_samples.py records by running each model.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
from apps import APPS as ALL_APPS, CONFIDENCE_KINDS, PAID_MAX_N, confidence_kind  # noqa: E402
import theme_toggle  # noqa: E402

# Registered in the runner but not benchmarked in this project. They get no model page; the index describes them in
# its "Worth knowing" section instead.
NOT_BENCHMARKED = {"dexy", "djev"}
APPS = [a for a in ALL_APPS if a.name not in NOT_BENCHMARKED]

OUT = ROOT / "models"
SAMPLES_FILE = OUT / "samples.json"
SAMPLE_TEXT = "Stocks rallied after the central bank decision."
AG_LABELS = {
    "world": "International news, politics, government, conflicts, diplomacy, elections",
    "sports": "Sports events, athletes, teams, games, tournaments, leagues",
    "business": "Companies, markets, economy, finance, earnings, trade, oil prices, mergers",
    "sci_tech": "Science, technology, computing, internet, software, space, health research",
}
AG_INSTRUCTION = "What is the topic of this news article?"


def h(value: object) -> str:
    return html.escape(str(value), quote=True)


def code(text: str, lang: str = "") -> str:
    label = f'<span class="lang">{h(lang)}</span>' if lang else ""
    return f'<div class="codeblock">{label}<pre><code>{h(text)}</code></pre></div>'


# ---------------------------------------------------------------------------------------------------------------
# Facts per app folder. Checked against each classifier's classifier.py, setup.sh and requirements.txt.
# ---------------------------------------------------------------------------------------------------------------
# model, publisher, weights or service, official page, summary
FACTS = {
    "jev": ("Jev", "TypeSafe", "Hosted API. The weights are not published.", "https://www.typesafe.ai/",
            "Jev is a paid decision API from TypeSafe. The adapter sends the text together with the list of labels as "
            "one choice question, and the service returns a probability for each label. The model cannot be run "
            "on this machine."),
    "laya": ("Laya", "Convai Innovations", "convaiinnovations/laya (English, multilingual and typed-decisions "
             "checkpoints in one repository)", "https://huggingface.co/convaiinnovations/laya",
             "Laya is an encoder decision model. It reads the input text, the instruction and the candidate labels "
             "together and gives each label a score. The repository holds three checkpoints: English, multilingual "
             "and typed-decisions."),
    "nli": ("DeBERTa-v3-large zero-shot v2.0", "Moritz Laurer", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            "https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            "This model treats classification as natural language inference. Each label becomes a sentence, for "
            "example \"The topic of this news article is Business (companies, markets, ...).\" The model estimates "
            "how strongly the input text supports each sentence, and the scores are normalized across the labels."),
    "modernbert_jev": ("ModernBERT-JEV", "ali-rehman-ML", "answerdotai/ModernBERT-base plus the LoRA adapter "
                       "ali-rehman-ML/modern-bert-jev", "https://github.com/ali-rehman-ML/modern-bert-jev",
                       "ModernBERT-JEV adds a LoRA adapter and a choice head to ModernBERT-base. It scores each "
                       "pair of text and label, then applies the temperature that ships with the release. Its "
                       "training data includes Banking77."),
    "gliclass": ("GLiClass base v3.0", "Knowledgator", "knowledgator/gliclass-base-v3.0",
                 "https://huggingface.co/knowledgator/gliclass-base-v3.0",
                 "GLiClass is a zero-shot classifier. It reads the text and all candidate labels in one forward pass "
                 "and returns a score for each label."),
    "verdict": ("Verdict", "Heman10x", "heman10x/rlcd-modernbert-151m",
                "https://huggingface.co/heman10x/rlcd-modernbert-151m",
                "Verdict is a decision model with 151 million parameters: ModernBERT-base with a GLiClass head. It is "
                "loaded with the gliclass package. It always adds an \"insufficient evidence\" option, divides the "
                "logits by a temperature chosen for the number of options, and accepts 2 to 24 labels."),
    "embedsim": ("BGE-M3 embedding similarity", "BAAI, served by Ollama", "bge-m3:567m (Ollama)",
                 "https://ollama.com/library/bge-m3",
                 "BGE-M3 turns the input text and each label description into vectors. The label whose vector has "
                 "the highest cosine similarity to the text is chosen. A temperature fitted on training data turns "
                 "the similarities into probabilities."),
    "SemIf": ("SemIf on Qwen3.5-4B", "SemIf (library), Qwen (model)", "Qwen/Qwen3.5-4B at the revision pinned in "
              "classifier.py", "https://github.com/TheoLeeCJ/SemIf-OpenJev",
              "SemIf reads the scores of the answer tokens directly from Qwen3.5-4B, running on MLX. It does not "
              "generate text. It accepts 2 to 16 labels, so Banking77 needs the shortlist variant."),
    "nimble": ("Bespoke-Nimble-9B", "Bespoke Labs", "bespokelabs/Bespoke-Nimble-9B",
               "https://huggingface.co/bespokelabs/Bespoke-Nimble-9B",
               "Bespoke-Nimble-9B is a decision model built on Qwen3.5-9B. Each label is given a one-token answer "
               "code, and the model's logits for those codes become the label probabilities. It accepts up to 255 "
               "choices."),
    "llm": ("Qwen 3.8 27B (Q4_K_M)", "Qwen, served by Ollama", "qwen3.8:27b-q4_K_M by default; LLMQWEN_MODEL selects "
            "another tag", "https://ollama.com/library/qwen3.8",
            "This entry asks a general chat model to pick an answer. The labels are listed with the letters A to P, "
            "the reply is constrained to one letter, and the adapter reads the probability the model gave to each "
            "letter. It does not use the written answer unless those probabilities are missing."),
    "decider": ("Decider-4B v3", "Mapika", "Mapika/decider-4b (v3 release commit, about 8.4 GB in BF16)",
                "https://huggingface.co/Mapika/decider-4b",
                "Decider-4B v3 is a decision model built on Qwen3.5-4B, run through the decider-ai package. Its "
                "probabilities use the temperature from the release. Its training mixture includes the train splits "
                "of AG News and Banking77, so its results on those tasks are in-domain, not zero-shot."),
    "strands_decider": ("Strands Decider 2B hobson-v19", "Strands Labs", "StrandsAgents/strands-decider-2B-hobson-v19 "
                        "(about 92 MB) on Qwen/Qwen3.5-2B-Base (about 4.5 GB)",
                        "https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19",
                        "Strands Decider is a small set of trained weights on top of Qwen3.5-2B-Base, run with MLX. It "
                        "is a different model from Decider-4B, with its own code and weights. It was trained on the AG "
                        "News and Banking77 train splits, so its results on those tasks are in-domain."),
    "d1": ("LiquidAI d1-3B", "Liquid AI", "LiquidAI/d1-3B", "https://huggingface.co/LiquidAI/d1-3B",
           "d1-3B is a decision model built on LFM2.5-VL-3B. It returns a probability for each candidate choice "
           "through its native System One interface. No tokens are generated."),
    "d1_omni": ("LiquidAI d1-omni-600M", "Liquid AI", "LiquidAI/d1-omni-600M",
                "https://huggingface.co/LiquidAI/d1-omni-600M",
                "d1-omni-600M is a smaller Liquid AI decision model with 600 million parameters. The adapter uses "
                "the choice probabilities the model produces directly."),
    "naluri": ("Naluri encoder", "Jericho Siahaya", "jerichosiahaya/Naluri-279M-encoder or "
               "jerichosiahaya/Naluri-308M-encoder", "https://huggingface.co/jerichosiahaya",
               "Naluri is a family of encoders trained to score a piece of text against a list of candidate "
               "decisions. The release reserves 192 tokens for the question and the options. Above about 20 options "
               "each option is cut short, so the adapter refuses tasks with more than 20 labels."),
    "d3": ("Decision 3.0", "vLLM Semantic Router", "vllm-sr/d3-edge, vllm-sr/d3-lite or vllm-sr/d3-nano",
           "https://huggingface.co/vllm-sr",
           "Decision 3.0 is a family of small decision models in three sizes: edge, lite and nano. Each returns a "
           "probability for every option. The runtime is written for CUDA and CPU. Apple MPS support is "
           "experimental, so this project runs it on the CPU by default."),
    "kev": ("Kev-0.8B", "jaredpalmer", "jaredpalmer/kev-0.8b on Qwen/Qwen3.5-0.8B-Base",
            "https://huggingface.co/jaredpalmer/kev-0.8b",
            "Kev-0.8B adds a pointer head to Qwen3.5-0.8B-Base and runs with MLX. The pointer head picks one of the "
            "listed options. Its training includes AG News and Banking77, so its results on those tasks are "
            "in-domain."),
    "dexy": ("Dexy-Decision-9B", "Patrick-18", "Patrick-18/Dexy-Decision-9B",
             "https://huggingface.co/Patrick-18/Dexy-Decision-9B",
             "Dexy-Decision-9B is an experimental 9B decision checkpoint. It is supported on CUDA. It has not been "
             "verified on Apple MPS in this project."),
    "plumb": ("Plumb-4B v5", "crh225", "crh225/plumb-4b", "https://huggingface.co/crh225/plumb-4b",
              "Plumb-4B is a decision model fine-tuned from JevK5 v0.2, which is Qwen3.5-4B with a merged LoRA. The "
              "prompt, answer letters and temperature come from the jevk5 runtime, and the forward pass runs on MLX. "
              "It accepts 2 to 16 options."),
    "winnow": ("Winnow-12B (Q8_0)", "Eldan Ring", "EldanRing/Winnow-12B, file gguf/Winnow-12B-Q8_0.gguf",
               "https://huggingface.co/EldanRing/Winnow-12B",
               "Winnow-12B is a decision fine-tune of Gemma 4 12B, quantized to Q8_0 and served by llama.cpp on "
               "Metal. The runner starts and stops the server for each benchmark run. One question can carry 2 to 64 "
               "options."),
    "clm": ("CLM-v0.1-8B", "Contrastive-LM", "Qwen/Qwen3-8B plus the Contrastive-LM/CLM-v0.1-8B heads (about 75 MB)",
            "https://huggingface.co/Contrastive-LM/CLM-v0.1-8B",
            "CLM scores labels with two small projection heads on top of a frozen Qwen3-8B encoder: one head for the "
            "text and one for each option. The score of a pair is the scaled cosine similarity of the two "
            "projections. The heads were trained on agent actions, not on topic labels."),
    "OpenJev": ("OpenJev with DiffusionGemma 26B-A4B", "OpenJev (server), Google (model)",
                "mlx-community/diffusiongemma-26B-A4B-it-4bit by default (OPENJEV_MLX_MODEL)",
                "https://github.com/razorback16/openjev",
                "OpenJev is a local server that accepts the same choice questions as the Jev API. In this project it "
                "runs a four-bit MLX build of DiffusionGemma 26B-A4B. The confidence value comes from the server and "
                "equals one minus the normalized entropy of the probabilities."),
    "djev": ("djev with DiffusionGemma 26B-A4B", "djev (server), Google (model)",
             "google/diffusiongemma-26B-A4B-it in BF16 (about 52 GB), downloaded on the remote GPU host",
             "https://github.com/Davipar/djev-dev",
             "djev serves DiffusionGemma 26B-A4B in BF16 with vLLM, inside Docker on a remote NVIDIA B200 host, and "
             "accepts the same choice questions as the Jev API. The texts you classify leave this machine, so check "
             "whether the task's data may be sent elsewhere before running it."),
    "tfidf_logreg": ("TF-IDF and logistic regression", "scikit-learn", "No pretrained weights. The model is fitted on "
                     "the task's train split.", "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression",
                     "Word features (1 and 2 word sequences) and character features (3 to 5 characters) are weighted "
                     "with TF-IDF and passed to a multinomial logistic regression. The regularization strength C is "
                     "chosen by 3-fold cross-validation on the train split."),
    "tfidf_svm": ("TF-IDF and linear SVM", "scikit-learn", "No pretrained weights. The model is fitted on the task's "
                  "train split.", "https://scikit-learn.org/stable/modules/svm.html#classification",
                  "The same TF-IDF features are passed to a linear support vector machine. An SVM produces margins, "
                  "not probabilities, so the calibrated entry converts them with sigmoid calibration fitted by 5-fold "
                  "cross-validation on the train split."),
    "tfidf_cnb": ("TF-IDF and Complement Naive Bayes", "scikit-learn", "No pretrained weights. The model is fitted on "
                  "the task's train split.", "https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes",
                  "TF-IDF features are passed to a Complement Naive Bayes classifier. It trains in seconds and gives "
                  "a quick reference score for the other models."),
    "embedsup": ("BGE-M3 with a supervised head", "BAAI (embeddings), scikit-learn (head)",
                 "bge-m3:567m (Ollama) plus a head fitted on the task's train split", "https://ollama.com/library/bge-m3",
                 "BGE-M3 embeds a sample of the training texts once, and a small classifier is fitted on those "
                 "vectors. The encoder itself is not changed."),
}

# Entry-specific replacements for (model, weights, official page) and one extra sentence.
OVERRIDES = {
    "nli_modernbert": ("ModernBERT-large zero-shot v2.0", "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
                       "https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0"),
    "naluri_279m": ("Naluri-279M encoder", "jerichosiahaya/Naluri-279M-encoder",
                    "https://huggingface.co/jerichosiahaya/Naluri-279M-encoder"),
    "naluri_308m": ("Naluri-308M encoder", "jerichosiahaya/Naluri-308M-encoder",
                    "https://huggingface.co/jerichosiahaya/Naluri-308M-encoder"),
    "d3_edge": ("Decision 3.0 edge", "vllm-sr/d3-edge", "https://huggingface.co/vllm-sr/d3-edge"),
    "d3_lite": ("Decision 3.0 lite", "vllm-sr/d3-lite", "https://huggingface.co/vllm-sr/d3-lite"),
    "d3_nano": ("Decision 3.0 nano", "vllm-sr/d3-nano", "https://huggingface.co/vllm-sr/d3-nano"),
}

VARIANT_NOTES = {
    "laya": "This entry does not pin a checkpoint. Each text goes to the English or the multilingual checkpoint "
            "according to its language.",
    "laya_typed": "This entry pins the typed-decisions checkpoint, a ModernBERT-large model with 421 million "
                  "parameters, and runs with a batch size of 16.",
    "laya_multilingual": "This entry pins the multilingual checkpoint, an mmBERT-base model with 322 million "
                         "parameters, and runs with a batch size of 8.",
    "nli": "This entry uses DeBERTa-v3-large in the dtype stored in the checkpoint.",
    "nli_modernbert": "This entry uses ModernBERT-large in float32 with a batch size of 8. The checkpoint is stored "
                      "in bfloat16, which moved probabilities by up to 0.04, so it is upcast to float32.",
    "naluri_279m": "This entry uses the 279M checkpoint, which is based on XLM-R.",
    "naluri_308m": "This entry uses the 308M checkpoint, which is based on mmBERT.",
    "d3_edge": "This entry uses the edge model, which has 0.59 billion parameters.",
    "d3_lite": "This entry uses the lite model.",
    "d3_nano": "This entry uses the nano model.",
    "tfidf_logreg": "This entry trains on the full train split.",
    "tfidf_logreg_4k": "This entry trains on the same sample as the embedding baselines: the task's "
                       "supervised_sample_per_label setting, which is 1,000 articles per label for AG News (4,000 in "
                       "total). The two methods can then be compared on equal training data.",
    "tfidf_svm": "This entry reports calibrated probabilities.",
    "tfidf_svm_raw": "This entry skips calibration. Its probabilities are a softmax over the SVM margins and are not "
                     "calibrated, so compare its accuracy rather than its confidence values.",
    "embedsup_centroid": "This entry computes one centroid per class. A temperature fitted on a held-out 20% of the "
                         "train sample turns the cosine similarities into probabilities, and the centroids are then "
                         "recomputed on the whole sample.",
    "embedsup_logreg": "This entry fits a logistic regression on the embeddings.",
}

# What each folder needs to run: engine, Python libraries (from requirements.txt), hardware, extra services.
ENGINE = {
    "jev": ("TypeSafe hosted API. Nothing runs locally.", "typesafe-sdk", "Any machine with internet access.",
            "A TypeSafe API key in .env (jev_api_key and jev_api_name)."),
    "laya": ("PyTorch on Apple MPS (CUDA or CPU also work).", "laya", "Apple Silicon Mac recommended.", None),
    "nli": ("PyTorch through the Hugging Face transformers zero-shot-classification pipeline.",
            "torch, transformers (4.48 or later), sentencepiece, protobuf", "Apple Silicon Mac recommended.", None),
    "modernbert_jev": ("PyTorch, with an optional ONNX Runtime backend.",
                       "torch, transformers (4.48 or later), onnxruntime, tokenizers. setup.sh also clones the "
                       "ModernBERT-JEV source into src/ at a pinned commit.", "Apple Silicon Mac recommended.", None),
    "gliclass": ("PyTorch through the gliclass package.", "torch, transformers, gliclass 0.1.20",
                 "Apple Silicon Mac recommended.", None),
    "verdict": ("PyTorch through the gliclass package.", "torch, transformers, gliclass 0.1.20",
                "Apple Silicon Mac recommended.", None),
    "embedsim": ("Ollama serves the bge-m3:567m embedding model. Python only sends HTTP requests.", "httpx",
                 "Any machine that can run Ollama.", "Ollama running at OLLAMA_HOST, with bge-m3:567m pulled."),
    "SemIf": ("MLX through the SemIf library.", "semif-phase1[mlx] (installed from GitHub at a pinned commit)",
              "Apple Silicon Mac required (MLX).", None),
    "nimble": ("MLX for inference. PyTorch and PEFT are installed for the Nimble source code.",
               "mlx 0.32.2, mlx-lm 0.31.3, transformers 5.17.0, torch 2.8.0, peft 0.21.0, accelerate 1.15.0, "
               "sentencepiece. setup.sh clones Nimble into src/ at a pinned commit.",
               "Apple Silicon Mac required (MLX).", None),
    "llm": ("Ollama serves the chat model. Python only sends HTTP requests to /api/chat.", "httpx",
            "A machine that can run a 27B model in Ollama. It can be another machine on the network (LLM_HOST).",
            "Ollama running at LLM_HOST (or OLLAMA_HOST), with the model tag pulled."),
    "decider": ("PyTorch on Apple MPS with MLX kernels, through the decider-ai package.",
                "decider-ai[metal] 1.8.0, pyarrow", "Apple Silicon Mac required for the Metal kernels.", None),
    "strands_decider": ("MLX through the strands-decider package.",
                        "strands-decider[mlx] (GitHub, pinned commit), mlx 0.32.3, mlx-lm 0.32.0, torch 2.7.1, "
                        "transformers 5.17.0, peft 0.21.0, pyarrow", "Apple Silicon Mac required (MLX).", None),
    "d1": ("PyTorch on Apple MPS.", "torch 2.7.1, torchvision 0.22.1, transformers 5.17.0, pillow",
           "Apple Silicon Mac recommended.", None),
    "d1_omni": ("PyTorch on Apple MPS.", "torch 2.7.1, torchvision 0.22.1, transformers 5.17.0",
                "Apple Silicon Mac recommended.", None),
    "naluri": ("PyTorch on Apple MPS.", "torch 2.7.1, transformers 5.17.0, safetensors, huggingface_hub",
               "Apple Silicon Mac recommended.", None),
    "d3": ("PyTorch on the CPU. MPS can be selected but is experimental.",
           "torch 2.7.1, torchvision 0.22.1, transformers 5.17.0, safetensors, accelerate, pillow, "
           "opencv-python-headless", "Any machine. CUDA is the upstream target.", None),
    "kev": ("MLX through the kev package.", "kev[mlx] (GitHub, pinned commit), pyarrow",
            "Apple Silicon Mac required (MLX).", None),
    "dexy": ("PyTorch on CUDA.", "torch 2.13.0, transformers 5.19.0, accelerate 1.12.0, safetensors, huggingface_hub",
             "An NVIDIA GPU. Apple MPS is not verified.", None),
    "plumb": ("MLX for the forward pass. The jevk5 runtime supplies the prompt and temperature.",
              "jevk5 (GitHub, v0.2.0), mlx 0.32.2, mlx-lm 0.31.3, transformers 5.17.0",
              "Apple Silicon Mac required (MLX).", None),
    "winnow": ("llama.cpp with Metal, in a separate server process. Python only sends HTTP requests.", "httpx",
               "Apple Silicon Mac.", "The Winnow server (classifiers/winnow/server/). The runner starts it for "
                                     "benchmark runs. For your own code, start it with server/serve.sh."),
    "clm": ("PyTorch on Apple MPS, in bfloat16.", "torch, transformers 5.17.0, numpy, huggingface_hub",
            "Apple Silicon Mac with room for an 8B model.", None),
    "OpenJev": ("MLX, in a separate OpenJev server process. Python only sends HTTP requests.", "httpx",
                "Apple Silicon Mac with room for a 26B four-bit model.",
                "The OpenJev server, started with classifiers/OpenJev/server/serve.sh (default port 8080)."),
    "djev": ("vLLM in a Docker container on a remote NVIDIA B200 host. Python only sends HTTP requests.", "httpx",
             "A remote NVIDIA B200 GPU host. This machine only runs the client.",
             "The remote server (infra/deploy_gpu_host.sh) and an SSH tunnel (infra/tunnel.sh)."),
    "tfidf_logreg": ("scikit-learn on the CPU.", "scikit-learn", "Any machine.", None),
    "tfidf_svm": ("scikit-learn on the CPU.", "scikit-learn", "Any machine.", None),
    "tfidf_cnb": ("scikit-learn on the CPU.", "scikit-learn", "Any machine.", None),
    "embedsup": ("Ollama for the embeddings, scikit-learn and NumPy on the CPU for the head.",
                 "httpx, numpy, scikit-learn", "Any machine that can run Ollama.",
                 "Ollama running at OLLAMA_HOST, with bge-m3:567m pulled."),
}

# The code that creates `clf` on each page. capture_samples.py runs exactly this code.
EXAMPLES = {
    "jev": "clf = JEVClassifier(labels, instruction)",
    "laya": "clf = LayaClassifier(labels, instruction)",
    "laya_typed": 'clf = LayaClassifier(labels, instruction, model="typed-decisions")',
    "laya_multilingual": 'clf = LayaClassifier(labels, instruction, model="multilingual")',
    "nli": "clf = NLIClassifier(labels)",
    "nli_modernbert": 'clf = NLIClassifier(labels, model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0")',
    "modernbert_jev": "clf = ModernBertJevClassifier(labels, instruction)",
    "gliclass": "clf = GLiClassClassifier(labels)",
    "verdict": "clf = VerdictClassifier(labels, instruction)",
    "embedsim": "clf = EmbedSimClassifier(labels)",
    "SemIf": "clf = SemIfClassifier(labels, instruction)",
    "nimble": "clf = NimbleClassifier(labels, instruction)",
    "llmqwen": 'clf = LLMClassifier(labels, instruction, model="qwen3.8:27b-q4_K_M")',
    "decider": "clf = DeciderClassifier(labels, instruction)",
    "strands_decider": "clf = StrandsDeciderClassifier(labels, instruction)",
    "d1": "clf = D1Classifier(labels, instruction)",
    "d1_omni": "clf = D1OmniClassifier(labels, instruction)",
    "naluri_279m": 'clf = NaluriClassifier(labels, instruction, variant="279m")',
    "naluri_308m": 'clf = NaluriClassifier(labels, instruction, variant="308m")',
    "d3_edge": 'clf = D3Classifier(labels, instruction, variant="edge")',
    "d3_lite": 'clf = D3Classifier(labels, instruction, variant="lite")',
    "d3_nano": 'clf = D3Classifier(labels, instruction, variant="nano")',
    "kev": "clf = KevClassifier(labels, instruction)",
    "dexy": "clf = DexyClassifier(labels, instruction)",
    "plumb": "clf = PlumbClassifier(labels, instruction)",
    "winnow": "clf = WinnowClassifier(labels, instruction)",
    "clm": "clf = CLMClassifier(labels, instruction)",
    "OpenJev": "clf = OpenJevClassifier(labels, instruction)",
    "djev": "clf = DjevClassifier(labels, instruction)",
    "tfidf_logreg": 'train = load_train_set()  # the task\'s train split\n'
                    'clf = TfidfLogRegClassifier().fit(train["text"].tolist(), train["label"].tolist())',
    "tfidf_logreg_4k": 'train = load_train_sample(per_class=1000, seed=0)  # 4,000 AG News articles\n'
                       'clf = TfidfLogRegClassifier().fit(train["text"].tolist(), train["label"].tolist())',
    "tfidf_svm": 'train = load_train_set()\n'
                 'clf = TfidfSVMClassifier().fit(train["text"].tolist(), train["label"].tolist())',
    "tfidf_svm_raw": 'train = load_train_set()\n'
                     'clf = TfidfSVMClassifier(calibration="none").fit(train["text"].tolist(), train["label"].tolist())',
    "tfidf_cnb": 'train = load_train_set()\n'
                 'clf = TfidfCNBClassifier().fit(train["text"].tolist(), train["label"].tolist())',
    "embedsup_centroid": 'clf = EmbedSupClassifier(method="centroid")\n'
                         'train = load_train_sample(per_class=1000, seed=0)\n'
                         'vectors, _, _ = clf.embed_train(train, seed=0)  # cached in results/cache/\n'
                         'clf.fit(vectors, train["label"].tolist())',
    "embedsup_logreg": 'clf = EmbedSupClassifier(method="logreg")\n'
                       'train = load_train_sample(per_class=1000, seed=0)\n'
                       'vectors, _, _ = clf.embed_train(train, seed=0)\n'
                       'clf.fit(vectors, train["label"].tolist())',
    # Trained from scratch / fine-tuned: trains on the train split, or loads models/<task>/ (scratch_sample shows it).
    "scratch_fasttext": "clf = trained()",
    "scratch_cnn": "clf = trained()",
    "scratch_bilstm": "clf = trained()",
    "finetune_modernbert": "clf = trained()",
    "finetune_deberta": "clf = trained()",
}

# Shortlist variants: a Banking77 example of both stages. capture_samples.py runs this code (it defines keep and p).
SHORTLIST_TEXT = "I ordered a new card two weeks ago and it still has not arrived."
SHORTLIST_CLASSES = {"SemIf": "SemIfClassifier", "verdict": "VerdictClassifier", "plumb": "PlumbClassifier",
                     "winnow": "WinnowClassifier", "llm": "LLMClassifier"}


def shortlist_example(app) -> str:
    cls = SHORTLIST_CLASSES[folder_of(app)]
    extra = ', model="qwen3.8:27b-q4_K_M"' if folder_of(app) == "llm" else ""
    return (f"from task import current_task\nfrom shortlist import Shortlister\nfrom classifier import {cls}\n\n"
            f"task = current_task()  # run with BENCHMARK_TASK=banking77\n"
            f'labels = {{s["id"]: s["description"] for s in task.labels}}  # all 77 intents\n'
            f"k = {app.shortlist}\n\n"
            f"shortlister = Shortlister(task, k)  # stage 1: embeds the 77 label texts with bge-m3 (Ollama)\n"
            f"clf = {cls}(dict(list(labels.items())[:k]), task.instruction{extra})  # stage 2, built for k labels\n\n"
            f'text = "{SHORTLIST_TEXT}"\n'
            f"keep = shortlister.top_k([text])[0]  # the k label IDs closest to the text\n"
            f"p = clf.predict_shortlisted([text], [{{l: labels[l] for l in keep}}])[0]\n"
            f"print(keep)\nprint(p.label, p.confidence)")


# Publishers given by their account name only (Hugging Face or GitHub); nothing else identifies them.
ACCOUNT_PUBLISHERS = {"verdict", "winnow", "naluri", "plumb", "dexy", "kev", "modernbert_jev"}

# How the CLI (python classifier.py "text") prints its result: a table, or the Prediction object.
REPR_CLI = {"d1_omni", "d3", "dexy", "kev", "naluri"}

# Adapters that talk to a server over HTTP, by request style.
CHOICE_API = {"jev", "OpenJev", "djev", "winnow"}

RELEASED = {"laya", "modernbert_jev", "gliclass", "verdict", "SemIf", "nimble", "llm", "decider", "strands_decider",
            "d1", "d1_omni", "naluri", "d3", "kev", "dexy", "plumb", "winnow", "clm", "OpenJev", "djev"}


def load_samples() -> dict:
    try:
        return json.loads(SAMPLES_FILE.read_text())
    except (OSError, ValueError):
        return {}


SAMPLES = load_samples()


def folder_of(app) -> str:
    return app.folder or app.name


def facts(app):
    model, maker, artifact, url, summary = FACTS[folder_of(app)]
    if app.name in OVERRIDES:
        model, artifact, url = OVERRIDES[app.name]
    return model, maker, artifact, url, summary


def group(app) -> str:
    if app.name in {p["name"] for p in SCRATCH}:  # neural networks trained from scratch: their own group
        return "scratch"
    if app.name in {p["name"] for p in FINETUNE}:  # pre-trained encoders, fine-tuned: their own group
        return "finetune"
    if app.shortlist:
        return "shortlist"
    if app.supervision == "supervised":
        return "supervised"
    if app.trained_on_tasks:
        return "in-domain"
    return "zero-shot"


GROUPS = [
    ("zero-shot", "Zero-shot models",
     "These models were not trained on the bundled tasks. They read the label names and descriptions at run time "
     "and choose among them."),
    ("in-domain", "Models trained on benchmark data",
     "The training data of these released models includes at least one bundled task. On those tasks the report "
     "marks their results as in-domain, not zero-shot."),
    ("supervised", "Classical supervised baselines",
     "These classifiers are fitted on the task's train split before they see any test text."),
    ("shortlist", "Two-stage shortlist variants",
     "BGE-M3 first reduces the label list to a fixed number of candidates. The model then chooses among those "
     "candidates. These entries exist for Banking77, which has more labels than the models accept."),
]
GROUP_TYPE = {"zero-shot": "Zero-shot", "in-domain": "In-domain checkpoint", "supervised": "Classical supervised",
              "shortlist": "Shortlist", "scratch": "Trained from scratch",
              "finetune": "Fine-tuned"}


def label_limit(app) -> str:
    if app.shortlist:
        return f"Top {app.shortlist} from BGE-M3"
    if app.max_labels:
        return f"Up to {app.max_labels}"
    return NO_HARD_LIMIT


NO_HARD_LIMIT = "No hard adapter limit"
TRAINED_LIMIT = "Labels of the train split"  # supervised models: the label set is whatever they were fitted on
PRACTICAL_LIMIT = ("There is no hard adapter limit on the number of labels. Practical capacity is still constrained by "
                   "context length, label-description truncation, memory, runtime and model quality, so validate large "
                   "label sets separately.")
LAYA_NOTE = ("The upstream model card warns that the shipped checkpoints are overconfident and recommends refitting a "
             "temperature per question type and option count before trusting the probabilities (mean ECE 0.466 to "
             "0.081 for <code>laya</code>, 0.314 to 0.106 for <code>laya-multilingual</code>, as reported on the "
             "card). This benchmark reports the adapter's <code>answer_confidence</code> without refitting; treat it "
             "as model-specific confidence unless it has been calibrated on your domain.")
LAYA_BUDGET = ("The card also warns that at its default prompt budget a 77-option question such as Banking77 leaves "
               "only about 3 to 4 tokens per label, and accuracy falls sharply. This adapter raises the budget to "
               "<code>max_len = 1024</code> and <code>head_max_len = 512</code> (about 6 tokens per label for 77 "
               "options), so its Banking77 results apply to that configuration only.")


def conf_badge(app) -> str:
    """The entry's confidence type, linked to its definition on the index."""
    label, text = CONFIDENCE_KINDS[confidence_kind(app)]
    return f'<a class="badge conf" href="index.html#confidence" title="{h(text)}">{h(label)}</a>'


CONF_TASK_BADGE = (f'<a class="badge conf" href="index.html#confidence" title="{h(CONFIDENCE_KINDS["task"][1])}">'
                   f'{h(CONFIDENCE_KINDS["task"][0])}</a>')


def probabilities_text(app) -> str:
    """The `probabilities` bullet: what the per-label values are for this entry (errata E8, E9)."""
    kind = confidence_kind(app)
    if app.shortlist:
        return ("a <code>dict</code> with every task label. Labels excluded by the shortlist are assigned 0 as "
                "placeholders, not model-assigned probabilities; the remaining values are the main model's normalized "
                "scores within the shortlist.")
    if kind == "normalized":
        return ("normalized scores for every label ID (a <code>dict</code>). They sum to 1 but are not calibrated "
                "probability estimates.")
    if kind == "separate":
        return "a <code>dict</code> from every label ID to the service's per-label score."
    return "a <code>dict</code> from every label ID to its probability estimate."


def confidence_caveat(app) -> str:
    if confidence_kind(app) == "separate":
        return (" The <code>confidence</code> field is a separate server score and should not be interpreted as the "
                "selected label's probability.")
    return ""


# Inline-SVG diagrams. Every one has a text equivalent next to it (figcaption or list); colours come from styles.css.
def _svg_box(x, y, w, hgt, lines, cls="bx") -> str:
    out = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{hgt}" rx="4"/>'
    top = y + hgt / 2 - (len(lines) - 1) * 8 + 4
    for i, line in enumerate(lines):
        out += f'<text x="{x + w / 2}" y="{top + i * 16}" text-anchor="middle">{h(line)}</text>'
    return out


def _svg_arrow(points, dashed=False, pre="") -> str:
    d = "M" + " L".join(f"{x} {y}" for x, y in points)
    return (f'<path class="ar{" dash" if dashed else ""}" d="{d}" '
            f'marker-end="url(#{pre}{"ahd" if dashed else "ah"})"/>')


def svg_defs(pre: str) -> str:
    """Arrowhead markers with ids unique to one diagram (several diagrams can share a page)."""
    return (f'<defs><marker id="{pre}ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path class="ah" d="M0 0 L10 5 L0 10 z"/></marker>'
            f'<marker id="{pre}ahd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path class="ahd" d="M0 0 L10 5 L0 10 z"/></marker></defs>')


def pipeline_diagram() -> str:
    """P1: how data flows through the benchmark, with the confidence-only calibration path dashed."""
    parts = [f'<svg viewBox="0 0 860 330" role="img" aria-labelledby="pipe-cap">{svg_defs('pipe-')}',
             _svg_box(10, 20, 150, 50, ["Task manifest", "train split"]),
             _svg_box(10, 200, 150, 50, ["Task manifest", "test split"]),
             _svg_box(220, 0, 190, 50, ["Supervised fitting", "(weights; supervised only)"]),
             _svg_box(220, 90, 190, 50, ["Confidence temperature", "(calibration only)"], "bx cal"),
             _svg_box(220, 200, 190, 50, ["Fixed stratified sample", "(same for every entry)"]),
             _svg_box(470, 10, 170, 50, ["Label names", "and descriptions"]),
             _svg_box(470, 150, 170, 60, ["Classifier adapter"], "bx key"),
             _svg_box(690, 150, 160, 60, ["pred, confidence,", "per-label scores"]),
             _svg_box(690, 260, 160, 50, ["Validation and", "report metrics"]),
             _svg_arrow([(160, 40), (190, 40), (190, 25), (218, 25)], pre='pipe-'),
             _svg_arrow([(160, 50), (190, 50), (190, 115), (218, 115)], dashed=True, pre='pipe-'),
             _svg_arrow([(160, 225), (218, 225)], pre='pipe-'),
             _svg_arrow([(410, 25), (440, 25), (440, 165), (468, 165)], pre='pipe-'),
             _svg_arrow([(410, 115), (430, 115), (430, 180), (468, 180)], dashed=True, pre='pipe-'),
             _svg_arrow([(410, 225), (440, 225), (440, 195), (468, 195)], pre='pipe-'),
             _svg_arrow([(555, 60), (555, 148)], pre='pipe-'),
             _svg_arrow([(640, 180), (688, 180)], pre='pipe-'),
             _svg_arrow([(770, 210), (770, 258)], pre='pipe-'),
             '<g class="legend"><line class="ar" x1="10" y1="300" x2="50" y2="300"/>'
             '<text x="58" y="304">data or fitting that can change the decision</text>'
             '<line class="ar dash" x1="10" y1="322" x2="50" y2="322"/>'
             '<text x="58" y="326">confidence calibration only (does not change the predicted label)</text></g>',
             "</svg>"]
    steps = ["The task manifest names the train and test splits, the labels and their descriptions.",
             "Supervised entries fit their weights on the train split. Some zero-shot entries fit only a confidence "
             "temperature on a labeled train sample (dashed): it changes confidence values, not the predicted label.",
             "The runner draws one fixed, stratified test sample, the same for every entry, from the test split only.",
             "Each adapter receives the texts with the label names and descriptions and returns the predicted label, a "
             "confidence value and a score for every label.",
             "The runner validates every result file and computes the report metrics from the predictions."]
    return ('<figure class="diagram">' + "".join(parts)
            + '<figcaption id="pipe-cap">How the benchmark runs every entry. Solid lines carry data or fitting that '
              'can change the decision; the dashed path is confidence calibration only.</figcaption>'
            + "<ol class=\"diagram-text\">" + "".join(f"<li>{x}</li>" for x in steps) + "</ol></figure>")


def temperature_diagram() -> str:
    """P3: scalar temperature scaling keeps the argmax."""
    parts = [f'<svg viewBox="0 0 760 130" role="img" aria-labelledby="temp-cap">{svg_defs('temp-')}',
             _svg_box(10, 20, 140, 50, ["label logits"]),
             _svg_box(200, 20, 150, 50, ["divide by one", "positive T"], "bx cal"),
             _svg_box(400, 20, 120, 50, ["softmax"]),
             _svg_box(570, 20, 180, 50, ["changed confidence", "values"]),
             _svg_arrow([(150, 45), (198, 45)], pre='temp-'), _svg_arrow([(350, 45), (398, 45)], pre='temp-'),
             _svg_arrow([(520, 45), (568, 45)], pre='temp-'),
             '<path class="ar" d="M80 70 C80 120, 660 120, 660 70" fill="none"/>',
             '<text x="370" y="114" text-anchor="middle">the label with the largest score is unchanged</text>',
             "</svg>"]
    return ('<figure class="diagram">' + "".join(parts)
            + '<figcaption id="temp-cap">Scalar temperature scaling: dividing every label logit by the same positive '
              'number T before the softmax changes how confident the output is, but not the order of the labels, so '
              'the predicted label stays the same. Other calibration methods (per-class scaling, isotonic, '
              'one-vs-rest) can change the order.</figcaption></figure>')


def flow_diagram(pre: str, stages: list, inputs: list, caption: str, steps: list[str]) -> str:
    """A left-to-right diagram: `stages` are (lines, cls) boxes joined by arrows; `inputs` are
    (stage index, lines, dashed) boxes above a stage with an arrow down into it. cls "key" = the model itself,
    "cal" = confidence calibration only (dashed). Rendered with a caption and the steps as a text equivalent."""
    n, gap, bw = len(stages), 34, 150
    width = 2 * 10 + 5 * bw + 4 * gap  # one width for every diagram, so text renders at the same size on every page
    left = (width - (n * bw + (n - 1) * gap)) / 2
    has_inputs = bool(inputs)
    y0 = 92 if has_inputs else 20
    bh = 64
    height = y0 + bh + (52 if any(c == "cal" for _, c in stages) else 24)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{pre}cap">{svg_defs(pre)}']
    xs = [left + i * (bw + gap) for i in range(n)]
    for i, (lines, cls) in enumerate(stages):
        parts.append(_svg_box(round(xs[i]), y0, round(bw), bh, lines, "bx " + cls if cls else "bx"))
        if i:
            parts.append(_svg_arrow([(round(xs[i - 1] + bw), y0 + bh / 2), (round(xs[i]) - 2, y0 + bh / 2)], pre=pre))
    for i, lines, dashed in inputs:
        parts.append(_svg_box(round(xs[i]), 10, round(bw), 50, lines, "bx cal" if dashed else "bx"))
        cx = round(xs[i] + bw / 2)
        parts.append(_svg_arrow([(cx, 60), (cx, y0 - 2)], dashed=dashed, pre=pre))
    if any(c == "cal" for _, c in stages):
        parts.append(f'<g class="legend"><rect class="bx cal" x="{left:.0f}" y="{height - 22}" width="26" height="14" rx="3"/>'
                     f'<text x="{left + 34:.0f}" y="{height - 11}">confidence calibration only: changes confidence values, '
                     'not the predicted label</text></g>')
    parts.append("</svg>")
    return ('<figure class="diagram">' + "".join(parts) + f'<figcaption id="{pre}cap">{caption}</figcaption>'
            + '<ol class="diagram-text">' + "".join(f"<li>{x}</li>" for x in steps) + "</ol></figure>")


# Which mechanism each adapter folder uses (checked against each adapter's docstring, 2026-11-19).
FAMILY = {
    "nli": "per-label", "modernbert_jev": "per-label",
    "laya": "joint", "gliclass": "joint", "verdict": "joint", "naluri": "joint", "d1": "joint", "d1_omni": "joint",
    "d3": "joint", "kev": "joint", "strands_decider": "joint",
    "SemIf": "codes", "nimble": "codes", "llm": "codes", "plumb": "codes", "winnow": "codes", "decider": "codes",
    "jev": "service", "OpenJev": "service", "djev": "service",
    "embedsim": "embedding", "clm": "embedding",
    "tfidf_logreg": "supervised", "tfidf_svm": "supervised", "tfidf_cnb": "supervised", "embedsup": "supervised",
}
FAMILY_TITLE = {
    "per-label": "Per-label scoring", "joint": "Decision model reading all options together",
    "codes": "Answer codes read from a language model", "service": "Choice service",
    "embedding": "Embedding similarity", "supervised": "Supervised classifier",
}
# Model-specific wording for the scoring box of the joint decision models.
JOINT_HEAD = {
    "laya": ["Laya encoder", "(scores each option)"], "gliclass": ["GLiClass encoder", "(text + all labels)"],
    "verdict": ["Verdict: ModernBERT", "+ GLiClass head"], "naluri": ["Naluri encoder", "+ decision head"],
    "d1": ["d1 System One", "native choice API"], "d1_omni": ["d1-omni System One", "native choice API"],
    "d3": ["Decision 3.0", "model + runtime"], "kev": ["Qwen3.5-0.8B", "+ pointer head"],
    "strands_decider": ["Qwen3.5-2B + LoRA", "+ pointer head"],
}
CODES_MODEL = {
    "SemIf": ["Qwen3.5-4B (MLX)", "no text generated"], "nimble": ["Nimble-9B (MLX)", "no text generated"],
    "llm": ["Qwen chat model", "reply = one letter"], "plumb": ["Plumb-4B (MLX)", "no text generated"],
    "winnow": ["Winnow-12B server", "(llama.cpp)"], "decider": ["Decider-4B", "no text generated"],
}
CODES_INPUT = {"nimble": ["text + options with", "one-token codes"], "llm": ["text + options", "lettered A to P"],
               "strands_decider": ["text + numbered", "options"]}


def _calibration_stage(app):
    """The calibration box for an app (lines, cls), or None when its scores are only normalized."""
    kind = confidence_kind(app)
    folder = folder_of(app)
    if kind == "task":
        return (["÷ temperature fitted", "on a train sample"], "cal")
    if kind == "release":
        if folder == "verdict":
            return (["÷ temperature for", "the option count"], "cal")
        return (["÷ release", "temperature"], "cal")
    return None


def family_diagram(app) -> str:
    """P4: how this entry turns a text into label scores, drawn from the family template with its specifics."""
    folder = folder_of(app)
    fam = FAMILY.get(folder)
    if not fam:
        return ""
    cal = _calibration_stage(app)
    limit = label_limit(app)
    model = facts(app)[0]
    out_box = (["one score per label", "→ highest wins"], "")
    cal_step = ("" if not cal else
                " The scores are then divided by one positive temperature, which changes the confidence values but "
                "not which label wins.")
    if fam == "per-label":
        if folder == "nli":
            first = (["text + one sentence", "per label"], "")
            mid = (["NLI model", "one pass per label"], "key")
            score = (["entailment score", "per label"], "")
            steps = ["Each label becomes a sentence such as “This news article is about Business”.",
                     f"{h(model)} reads the text with each sentence in a separate pass and estimates how strongly the "
                     "text supports it (entailment).",
                     "The entailment scores are normalized across the labels, so they sum to 1."]
        else:
            first = (["text + one candidate", "per label"], "")
            mid = (["ModernBERT + LoRA", "one pass per label"], "key")
            score = (["one logit", "per label"], "")
            steps = ["Each label description becomes a candidate paired with the text.",
                     f"{h(model)} scores every (text, candidate) pair on its own and returns one logit per label."
                     + cal_step, "A softmax across the labels gives scores that sum to 1."]
        stages = [first, mid, score] + ([cal] if cal else []) + [(["normalize: one", "score per label"], "")]
        caption = ("Per-label scoring: the model sees one label at a time, so the cost grows with the number of "
                   "labels, but there is no fixed label limit in the adapter.")
    elif fam == "joint":
        head = JOINT_HEAD.get(folder, [model[:20], "decision model"])
        stages = [(CODES_INPUT.get(folder, ["text, instruction", "and all labels"]), ""), (head, "key"),
                  (["score per option"], "")] + ([cal] if cal else []) + [(["softmax: one", "score per label"], "")]
        extra = (" Verdict also adds an “insufficient evidence” option; its share is dropped and the task labels "
                 "are renormalized." if folder == "verdict" else "")
        steps = [f"The text, the instruction and the labels (with their descriptions) form one choice question. "
                 f"Label limit: {h(limit)}.",
                 f"{h(model)} reads the question and gives each option a score; nothing is generated." + extra
                 + cal_step, "A softmax over the options gives one score per label; the highest one is the answer."]
        caption = ("Decision model: the text and every option are read together, and the model scores each option. "
                   "All options must fit in the model's input, which is why some entries have a label limit.")
    elif fam == "codes":
        stages = [(CODES_INPUT.get(folder, ["text + options", "with answer letters"]), ""),
                  (CODES_MODEL.get(folder, [model[:20], "language model"]), "key"),
                  (["logit of each", "answer code"], "")] + ([cal] if cal else []) \
            + [(["softmax: one", "score per label"], "")]
        steps = [f"The prompt lists the labels, each with a short answer code (a letter or token). Label limit: "
                 f"{h(limit)}.",
                 f"{h(model)} runs one forward pass to the answer position; the adapter reads the logit of every "
                 "answer code instead of letting the model write a reply." + cal_step,
                 "A softmax over the codes gives one score per label."]
        caption = ("Answer codes: a language model is prompted with coded options and the probabilities of the codes "
                   "are read directly. The number of codes the prompt format supports sets the label limit.")
    elif fam == "service":
        remote = app.remote
        stages = [(["choice question:", "text + options"], ""),
                  ((["Jev API", "(remote, paid)"] if remote else ["OpenJev server", "(local, MLX)"]), "key"),
                  (["probability per option", "+ own confidence score"], ""), out_box]
        steps = ["The adapter sends one /v1/systemone choice question with the text and the label descriptions."
                 + (" The text leaves this machine." if remote else " The server runs on this machine."),
                 "The service returns a probability for each option.",
                 "The confidence value is the service's own score"
                 + (" (1 minus the normalized entropy)" if folder == "OpenJev" else "")
                 + ", not the selected label's probability."]
        caption = "Choice service: the model runs behind an API; the adapter sends the question and reads the answer."
    elif fam == "embedding":
        if folder == "clm":
            stages = [(["text"], ""), (["Qwen3-8B (frozen)", "+ text / option heads"], "key"),
                      (["scaled cosine", "per option"], "")]
            enc = "the frozen Qwen3-8B encoder and two small projection heads, one for the text and one for options"
        else:
            stages = [(["text"], ""), (["BGE-M3 encoder", "(Ollama)"], "key"), (["cosine similarity", "per label"], "")]
            enc = "BGE-M3"
        stages += ([cal] if cal else []) + [(["softmax: one", "score per label"], "")]
        inputs = [(1, ["label descriptions", "(embedded too)"], False)]
        steps = [f"The text and every label description are turned into vectors by {enc}.",
                 "The adapter computes the cosine similarity between the text vector and each label vector."
                 + cal_step, "A softmax over the labels gives one score per label."]
        caption = ("Embedding similarity: no label is trained into the model; the label whose description lies "
                   "closest to the text wins.")
        return ("<h3>How it works</h3>" + flow_diagram("fam-", stages, inputs, caption, steps))
    else:  # supervised
        if folder == "embedsup":
            feats = (["BGE-M3 embedding", "(not changed)"], "")
            clf = (["class centroids" if "centroid" in app.name else "logistic", "regression" if "centroid" not in app.name else "(nearest wins)"], "key")
            fit = "a small classifier (class centroids or logistic regression) on the train texts' embeddings"
        else:
            feats = (["TF-IDF features", "words + characters"], "")
            name = {"tfidf_logreg": ["logistic", "regression"], "tfidf_svm": ["linear SVM", "(margins)"],
                    "tfidf_cnb": ["Complement", "Naive Bayes"]}[folder]
            clf = (name, "key")
            fit = "the TF-IDF vectorizer and the classifier on the task's train split"
        svm_cal = None
        if folder == "tfidf_svm":
            svm_cal = (["sigmoid calibration", "(train, 5-fold CV)"] if confidence_kind(app) == "task"
                       else ["softmax of margins", "(uncalibrated)"], "")
        cal_box = svm_cal or cal
        stages = [(["text"], ""), feats, clf] + ([cal_box] if cal_box else []) + [(["one score", "per label"], "")]
        inputs = [(2, ["train split", "(fitted once)"], False)]
        last = "Its scores become one value per label"
        if folder == "tfidf_svm" and confidence_kind(app) == "task":
            last = ("The SVM margins are converted by one sigmoid per label, fitted by 5-fold cross-validation on the "
                    "train split, then normalized. Unlike a single temperature, per-label calibration can occasionally "
                    "change which label has the highest score")
        elif folder == "tfidf_svm":
            last = "A softmax of the raw margins gives normalized scores that are not calibrated probabilities"
        elif cal_box and cal_box[1] == "cal":
            last += "; the temperature changes the confidence values, not the predicted label"
        steps = [f"Before any test text is seen, the runner fits {fit}.",
                 "Each test text is turned into the same features and passed to the fitted classifier.", last + "."]
        caption = ("Supervised classifier: fitted on labeled examples of this task, so the labels are fixed by "
                   "the train split.")
        return ("<h3>How it works</h3>" + flow_diagram("fam-", stages, inputs, caption, steps))
    return ("<h3>How it works</h3>" + flow_diagram("fam-", stages, [], caption, steps))


def trained_diagram(p, finetuned: bool) -> str:
    """P4 for the fine-tuned and from-scratch pages: training feeds the network; temperature is calibration only."""
    short = p["short"]
    if finetuned:
        stages = [(["text → sub-word", "tokens"], ""), ([f"{short}", "pre-trained, updated"], "key"),
                  (["classification", "head"], ""), (["÷ temperature fitted", "on held-out train"], "cal"),
                  (["softmax: one", "score per label"], "")]
        inputs = [(1, ["train split", "(fine-tuning)"], False)]
        steps = ["The text is split into sub-word tokens by the model's own tokenizer.",
                 f"The {h(short)} encoder, pre-trained on large amounts of text, turns the tokens into one text vector. "
                 "Fine-tuning on the task's train split adjusts these weights and trains the head.",
                 "The classification head gives one score per label; a temperature fitted on held-out train data "
                 "changes the confidence values, not the predicted label."]
        caption = (f"Fine-tuned encoder: {h(short)} starts from pre-trained weights, and the train split teaches it "
                   "the labels.")
    else:
        stages = [(["text → word vectors", "(random at start)"], ""), ([f"{short} encoder"], "key"),
                  (["linear head"], ""), (["÷ temperature fitted", "on held-out train"], "cal"),
                  (["softmax: one", "score per label"], "")]
        inputs = [(1, ["train split", "(trained from scratch)"], False)]
        steps = ["Each word gets a vector; at the start the vectors are random and training shapes them.",
                 f"The {h(short)} encoder turns the word vectors into one text vector, and a linear head gives one "
                 "score per label. All of it is learned from the task's train split.",
                 "A temperature fitted on held-out train data changes the confidence values, not the predicted label."]
        caption = (f"Network trained from scratch: every weight of the {h(short)} network is learned from this "
                   "task's train split.")
    return "<h3>How it works</h3>" + flow_diagram("fam-", stages, inputs, caption, steps)


def shortlist_diagram(k: int) -> str:
    """P4: the two-stage shortlist pipeline and where the zero placeholders come from."""
    parts = [f'<svg viewBox="0 0 860 150" role="img" aria-labelledby="sl-cap">{svg_defs('sl-')}',
             _svg_box(10, 30, 140, 50, ["All 77 labels", "and the text"]),
             _svg_box(190, 30, 170, 50, ["BGE-M3 similarity", "(embedding retrieval)"]),
             _svg_box(400, 30, 120, 50, [f"Top {k} labels"]),
             _svg_box(560, 30, 130, 50, ["Main model"], "bx key"),
             _svg_box(730, 10, 120, 40, [f"scores for {k}"]),
             _svg_box(730, 70, 120, 40, ["0 for the rest"], "bx cal"),
             _svg_arrow([(150, 55), (188, 55)], pre='sl-'), _svg_arrow([(360, 55), (398, 55)], pre='sl-'),
             _svg_arrow([(520, 55), (558, 55)], pre='sl-'), _svg_arrow([(690, 45), (728, 30)], pre='sl-'),
             _svg_arrow([(460, 80), (460, 120), (790, 120), (790, 112)], dashed=True, pre='sl-'),
             '<text x="560" y="140" text-anchor="middle">labels outside the shortlist: placeholder 0, never scored</text>',
             "</svg>"]
    return ("<h3>The two stages at a glance</h3><figure class=\"diagram\">" + "".join(parts)
            + f'<figcaption id="sl-cap">BGE-M3 keeps the {k} labels closest to the text; only those reach the main '
              f'model, which scores them. Every other label gets 0 in the output as a placeholder, not as a model '
              f'judgement. If the correct label is not kept, the entry cannot predict it.</figcaption></figure>')


def siblings(app) -> list:
    return [a for a in APPS if folder_of(a) == folder_of(app) and a.name != app.name]


def shortlist_entry(app):
    return next((a for a in APPS if a.shortlist and folder_of(a) == folder_of(app)), None)


# ---------------------------------------------------------------------------------------------------------------
# Page parts
# ---------------------------------------------------------------------------------------------------------------
FONTS = ""  # the fonts are local files in models/fonts/, declared in styles.css
# The mark: a scatter plot split by a decision line (circles on one side, squares on the other).
LOGO_SHAPES = ('<path class="ln" d="M4 3v17h17M2.3 4.7 4 3l1.7 1.7M19.3 18.3 21 20l-1.7 1.7M6.5 17.5 18.5 5.5"/>'
               '<circle class="pt" cx="8" cy="8.2" r="1.6"/><circle class="pt" cx="12.6" cy="4.8" r="1.5"/>'
               '<circle class="pt" cx="7.3" cy="12.6" r="1.4"/>'
               '<rect class="sq" x="15.8" y="9.4" width="2.8" height="2.8" rx=".5"/>'
               '<rect class="sq" x="12.2" y="13.9" width="3" height="3" rx=".5"/>'
               '<rect class="sq" x="17.3" y="14.3" width="2.7" height="2.7" rx=".5"/>'
               '<rect class="sq" x="9" y="16.2" width="2.5" height="2.5" rx=".5"/>')
LOGO = f'<svg class="mark" viewBox="0 0 24 24" aria-hidden="true">{LOGO_SHAPES}</svg>'
FAVICON = "data:image/svg+xml," + quote(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><style>.ln{fill:none;stroke:#914508;stroke-width:1.8;'
    'stroke-linecap:round;stroke-linejoin:round}.pt{fill:#fff0dd;stroke:#914508;stroke-width:1.2}.sq{fill:#e8922f;'
    f'stroke:#914508;stroke-width:1.2}}</style>{LOGO_SHAPES}</svg>')
CHART_ICON = ('<svg class="ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 2v12h12M5 11V8M8 11V5M11 11V7"/>'
              '</svg>')


def head(title: str, description: str) -> str:
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="description" content="{h(description)}"><title>{h(title)}</title>{FONTS}'
            f'<link rel="icon" href="{FAVICON}">'
            f'<link rel="stylesheet" href="styles.css">{theme_toggle.HEAD_JS}</head><body>')


def topbar(current: str = "") -> str:
    def nav(href, label, key):
        cur = ' aria-current="page"' if key == current else ""
        return f'<a href="{href}"{cur}>{label}</a>'
    return (f'<header class="topbar"><div class="wrap"><a class="brand" href="index.html">{LOGO}<span>Classification'
            f'</span></a><nav class="topnav">{nav("index.html", "Models", "models")}'
            f'{nav("index.html#intro", "Introduction", "intro")}</nav>'
            f'<div class="actions"><span class="actions-label">Reports</span>{report_links()}'
            f'{theme_toggle.BUTTON}</div></div></header>{theme_toggle.BODY_JS}')


def report_links() -> str:
    """Links that open the two HTML reports. Same style for both, so neither looks selected."""
    return "".join(f'<a class="btn report" href="../reports/{task}.html" title="Open the {name} performance report">'
                   f'{CHART_ICON}<span>{name}</span><span class="arrow" aria-hidden="true">&#8599;</span></a>'
                   for task, name in (("ag_news", "AG News"), ("banking77", "Banking77")))


def footer() -> str:
    return (f'<footer class="site"><div class="wrap"><span>{LOGO}Classification benchmark, model guide</span>'
            f'<span>Generated by <code>models/generate_pages.py</code> from <code>runner/apps.py</code></span>'
            f'</div></footer></body></html>')


def note(kind: str, tag: str, body: str) -> str:
    return f'<div class="note {kind}"><span class="tag">{h(tag)}</span>{body}</div>'


def kv_table(rows: list[tuple[str, str]]) -> str:
    return '<table class="kv"><tbody>' + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</tbody></table>"


# ---------------------------------------------------------------------------------------------------------------
# Model page sections
# ---------------------------------------------------------------------------------------------------------------
def section_about(app) -> str:
    model, maker, artifact, url, summary = facts(app)
    folder = folder_of(app)
    paras = []  # the summary is the page's lede, directly above
    reg = (f"In this project the model is registered as <code>{h(app.name)}</code>, and its code is in "
           f"<code>classifiers/{h(folder)}/</code>.")
    if app.args:
        reg += f" This entry passes <code>{h(' '.join(app.args))}</code> to the evaluator."
    others = siblings(app)
    if others:
        reg += " Other entries that use the same folder: " + ", ".join(
            f'<a href="{h(a.name)}.html"><code>{h(a.name)}</code></a>' for a in others) + "."
    paras.append(f"<p>{reg}</p>")
    if app.name in VARIANT_NOTES:
        paras.append(f"<p>{h(VARIANT_NOTES[app.name])}</p>")

    limits = []
    if app.shortlist:
        limits.append(f"This is a two-stage entry. BGE-M3 first compares the text with every label description and "
                      f"keeps the {app.shortlist} closest labels. The model then chooses among those {app.shortlist}. "
                      f"If the correct label is not in the shortlist, the model cannot choose it, so the shortlist's "
                      f"recall is the highest accuracy this entry can reach.")
    elif app.max_labels:
        limit = (f"The model accepts at most {app.max_labels} labels in one question. AG News has 4 labels and "
                 f"Banking77 has 77")
        if app.max_labels < 77:
            sl = shortlist_entry(app)
            limit += (f", so Banking77 cannot be run directly. Use <a href=\"{h(sl.name)}.html\"><code>{h(sl.name)}"
                      f"</code></a> for Banking77." if sl else ", so Banking77 cannot be run with this model.")
        else:
            limit += ", so both tasks fit."
        limits.append(limit)
    elif app.supervision != "supervised":
        limits.append(PRACTICAL_LIMIT + (" " + LAYA_BUDGET if folder == "laya" else ""))
    if app.remote:
        limits.append("The input text is sent to a remote service. The runner skips tasks whose manifest does not "
                      "allow remote processing.")
    if app.cost == "paid":
        limits.append(f"Each request is billed. The runner refuses samples larger than {PAID_MAX_N:,} items unless "
                      f"you pass <code>--allow-paid</code>.")
    paras += [f"<p>{x}</p>" for x in limits]
    if app.shortlist:
        base = next((a for a in APPS if folder_of(a) == folder and not a.shortlist and not a.args), None)
        if base:
            paras.append(family_diagram(base).replace(
                "<h3>How it works</h3>", "<h3>How the main model works</h3><p>After the shortlist stage (see "
                "<a href=\"#io\">Sending data and reading results</a>), the main model works as below, on the "
                f"{app.shortlist} kept labels only.</p>", 1))
    else:
        paras.append(family_diagram(app))
    if folder == "laya":
        paras.append(note("warn", "Calibration", f"<p>{LAYA_NOTE}</p>"))

    trained = ", ".join(app.trained_on_tasks) if app.trained_on_tasks else (
        "Fitted on the task's train split" if app.supervision == "supervised" else "None of the bundled tasks")
    publisher = h(maker)
    if folder in ACCOUNT_PUBLISHERS:
        publisher += ('<br><span class="aside">This is the name of the account that published the model on Hugging '
                      'Face or GitHub. The link below was checked on 2026-11-18; who runs the account was not.</span>')
    if folder == "djev":
        paras.append("<p>djev cannot run on a Mac. Its deploy script needs an NVIDIA B200 with CUDA, Docker and the "
                     "NVIDIA Container Toolkit, because the BF16 weights alone are about 52 GB. To run the same model "
                     "on a Mac, use <a href=\"OpenJev.html\"><code>OpenJev</code></a>: it serves DiffusionGemma "
                     "26B-A4B in four-bit MLX and accepts the same requests.</p>")
    rows = [("Model", h(model)), ("Publisher", publisher), ("Weights or service", h(artifact)),
            ("Official page", f'<a href="{h(url)}">{h(url.replace("https://", ""))}</a>'),
            ("Runs on", h(app.runs_on)), ("Label limit", h(label_limit(app))),
            ("Confidence value", h(app.confidence) + " " + conf_badge(app)),
            ("Trained on benchmark data", h(trained))]
    return "".join(paras) + kv_table(rows)


def section_requirements(app) -> str:
    folder = folder_of(app)
    engine, libs, hardware, service = ENGINE[folder]
    ollama = "No."
    if app.needs == "ollama" or app.extra_ollama or (app.ollama_model and folder != "llm"):
        tag = (app.ollama_model or app.extra_ollama)[1]
        ollama = f"Yes. Ollama must be running and the tag <code>{h(tag)}</code> must be pulled."
    if folder == "llm":
        ollama = ("Yes. The chat model runs in Ollama at <code>LLM_HOST</code> (default <code>OLLAMA_HOST</code>). "
                  "Pull it with <code>ollama pull qwen3.8:27b-q4_K_M</code>.")
        if app.shortlist:
            ollama += " The shortlist also needs <code>bge-m3:567m</code> on <code>OLLAMA_HOST</code>."
    elif app.shortlist:
        ollama = ("Yes, for the shortlist only. Ollama must be running at <code>OLLAMA_HOST</code> with "
                  "<code>bge-m3:567m</code> pulled. The model itself does not use Ollama.")
    rows = [("Inference engine", h(engine)), ("Python libraries", h(libs) + ". Every entry also installs "
            "pandas and httpx for the evaluator, and most install python-dotenv to read <code>.env</code>."),
            ("Hardware", h(hardware)), ("Ollama", ollama),
            ("Other services", h(service) if service else "None."),
            ("Python environment", f"Its own virtual environment in <code>classifiers/{h(folder)}/.venv</code>, "
             f"created by <code>setup.sh</code>. Entries do not share packages.")]
    return ("<p>This table lists what must be present before the model can classify a text. The exact versions are "
            f"in <code>classifiers/{h(folder)}/requirements.txt</code>, and <code>setup.sh</code> installs them.</p>"
            + kv_table(rows))


def section_download(app) -> str:
    _, _, _, url, _ = facts(app)
    folder = folder_of(app)
    pins = ("" if folder in {"tfidf_logreg", "tfidf_svm", "tfidf_cnb", "jev"} else
            f" The setup script pins exact revisions in <code>classifiers/{h(folder)}/setup.sh</code> or "
            f"<code>classifier.py</code>. Use the setup script instead of downloading the latest revision by hand, "
            f"because a different revision can give different results.")
    out = [f'<p>The official page is <a href="{h(url)}">{h(url.replace("https://", ""))}</a>.{pins}</p>',
           code(f"./run setup {app.name}", "shell")]
    if folder in {"tfidf_logreg", "tfidf_svm", "tfidf_cnb"}:
        out.append("<p>There are no pretrained weights. The setup script creates the virtual environment and installs "
                   "scikit-learn. The model is fitted on the task's train split the first time you run it, and the "
                   "fit is cached in <code>results/cache/</code>.</p>")
    elif folder == "OpenJev":
        out.append("<p>For this entry, <code>./run setup</code> installs only the client. The model runs in a separate "
                   "server. The server setup clones the OpenJev repository into <code>server/src/</code> and downloads "
                   "the MLX model named by <code>OPENJEV_MLX_MODEL</code>.</p>"
                   + code("classifiers/OpenJev/server/setup.sh   # once\nclassifiers/OpenJev/server/serve.sh   "
                          "# start the server, leave it running", "shell"))
    elif folder == "djev":
        out.append("<p>For this entry, <code>./run setup</code> installs only the client. The server is deployed to a "
                   "remote GPU host, and the client reaches it through an SSH tunnel at <code>DJEV_URL</code>.</p>"
                   + code("classifiers/djev/infra/deploy_gpu_host.sh   # once, deploys the server\n"
                          "classifiers/djev/infra/tunnel.sh            # opens the tunnel to 127.0.0.1:8000", "shell"))
    elif folder == "winnow":
        out.append("<p>The setup script also prepares the llama.cpp server in <code>server/</code> and downloads the "
                   "Q8_0 GGUF file. The runner starts and stops the server during benchmark runs.</p>")
    elif folder == "jev":
        out.append("<p>Nothing is downloaded. The setup script installs the TypeSafe SDK. Put your key in "
                   "<code>.env</code> as <code>jev_api_key</code> (with <code>jev_api_name</code>).</p>")
    else:
        out.append(f"<p>The setup script creates <code>classifiers/{h(folder)}/.venv</code>, installs "
                   f"<code>requirements.txt</code>, downloads the pinned weights, and runs the adapter's checks where it "
                   f"has them (probe texts, and for some entries a comparison of GPU and CPU scores).</p>")
    if app.ollama_model or app.extra_ollama:
        tags = sorted({t[1] for t in (app.ollama_model, app.extra_ollama) if t})
        out.append("<p>Ollama models are pulled with " + ", ".join(f"<code>ollama pull {h(t)}</code>" for t in tags)
                   + ". The setup script does this when Ollama is running.</p>")
    return "".join(out)


def section_install(app) -> str:
    return (code(f"cd /path/to/Classification\nrunner/setup.sh        # the runner's own environment\n"
                 f"cp .env.example .env    # then edit the settings you need\n./run list\n./run setup {app.name}",
                 "shell")
            + "<p>The reference platform is macOS on Apple Silicon. Requirements for memory, GPU, Ollama, servers and "
              "API keys differ between entries, and <code>./run list</code> states what is missing for each one. If a "
              "check fails during setup, fix the cause instead of skipping the check. Keep API keys in "
              "<code>.env</code>. Do not put them in HTML files or commit them.</p>")


def sample_block(app) -> str:
    s = SAMPLES.get(app.name)
    if not s:
        return ""
    width = max(len(k) for k in s["probabilities"])
    if folder_of(app) in REPR_CLI and not app.shortlist:
        # These demos print the Prediction object: labels in task order (values rounded here to 4 decimals).
        order = [k for k in AG_LABELS if k in s["probabilities"]] or list(s["probabilities"])
        probs = ", ".join(f"'{k}': {s['probabilities'][k]}" for k in order)
        printed = f"Prediction(label='{s['label']}', confidence={s['confidence']}, probabilities={{{probs}}})"
    else:
        printed = (f"label:      {s['label']}\nconfidence: {s['confidence']:.2f}\n"
                   + "\n".join(f"  {k:<{max(width, 10)}} {v:.3f}" for k, v in s["probabilities"].items()))
    return printed


def section_io(app) -> str:
    folder = folder_of(app)
    out = []
    if app.shortlist:
        base = next(a for a in APPS if folder_of(a) == folder and not a.shortlist and not a.args)
        out.append(f"<h3>How the two stages work</h3><p>For each text, BGE-M3 (through Ollama) ranks all labels by "
                   f"cosine similarity between the text and the label descriptions. The {app.shortlist} closest labels "
                   f"are passed to the model as an ordinary label set. The output dictionary still has every task "
                   f"label: labels excluded by the shortlist are assigned 0 as placeholders, and the remaining values are "
                   f"the main model's normalized scores within the shortlist. "
                   f"The input and output formats of the model itself are described on the <a href=\"{h(base.name)}"
                   f".html\"><code>{h(base.name)}</code> page</a>.</p>")
        out.append(shortlist_diagram(app.shortlist))
        out.append("<h3>Calling it from Python</h3><p>The benchmark evaluator does both stages for you. To do them in "
                   "your own code, build a <code>Shortlister</code> from <code>shared/shortlist.py</code>, then pass "
                   "each text's shortlisted labels to <code>predict_shortlisted</code>, which every shortlist-capable "
                   "classifier has.</p>")
        out.append(code(f"cd classifiers/{folder}\nBENCHMARK_TASK=banking77 PYTHONPATH=../../shared "
                        f".venv/bin/python my_script.py", "shell"))
        out.append(code(shortlist_example(app), "python"))
        s = SAMPLES.get(app.name)
        if s:
            top = list(s["probabilities"].items())[:5]
            printed = (f"shortlist ({len(s['shortlist'])} labels): " + ", ".join(s["shortlist"]) + "\n\n"
                       f"label:      {s['label']}\nconfidence: {s['confidence']:.2f}\n"
                       + "\n".join(f"  {k:<28} {v:.3f}" for k, v in top))
            hit = "card_arrival" in s["shortlist"]
            out.append(f"<h3>Sample output</h3><p>A real result from this project, recorded on {h(s['date'])} with the "
                       f"Banking77 labels and the text <code>{h(SHORTLIST_TEXT)}</code>, whose correct intent is "
                       f"<code>card_arrival</code>. The shortlist {'kept' if hit else 'did not keep'} the correct "
                       f"label. The first line lists the labels the model chose from, most similar first. The table "
                       f"shows the five most probable labels. Both stages together took "
                       f"{round(s['load_s'] + s['predict_s'], 1)} seconds, including loading the model.</p>")
            out.append(code(printed, "output"))
        out.append("<h3>Running it on the benchmark</h3>" + code(f"./run {app.name} --task banking77 --n 200", "shell"))
        out.append("<p>The run metadata records the shortlist's recall: the share of texts whose correct label was "
                   "kept, measured on the calibration split and on the test sample. That share is the highest "
                   "accuracy the entry can reach.</p>")
        out.append("<h3>Benchmark output</h3>" + benchmark_output())
        return "".join(out)

    supervised = app.supervision == "supervised"
    out.append("<h3>Input format</h3>")
    if supervised:
        out.append("<p>A supervised classifier learns its labels from the training data, so you do not pass a label "
                   "list. The labels and the train split come from the task manifest selected by "
                   "<code>BENCHMARK_TASK</code> (default <code>ag_news</code>). To classify, you pass one string, or a "
                   "list of strings to <code>predict_many</code>.</p>")
    else:
        out.append("<ul><li><b>Labels</b>: a Python <code>dict</code> that maps each label ID to a short description. "
                   "The ID is what the model returns. The description is what the model reads, so it should say "
                   "clearly what belongs in that label.</li>")
        if "instruction" in EXAMPLES[app.name]:
            out.append("<li><b>Instruction</b>: one question that tells the model what decision to make, for example "
                       f"<code>{h(AG_INSTRUCTION)}</code>.</li>")
        else:
            out.append("<li><b>Instruction</b>: this adapter does not take one. It reads only the text and the "
                       "labels.</li>")
        out.append("<li><b>Text</b>: one string per call to <code>predict</code>, or a list of strings for "
                   "<code>predict_many</code>, which batches them.</li></ul>")
        out.append("<p>The benchmark builds the labels from the task manifest in <code>data/tasks/</code>. For your own "
                   "labels, pass your own dictionary. Use IDs without spaces, because the evaluator writes one column "
                   "named <code>p_&lt;id&gt;</code> per label.</p>")

    out.append("<h3>Calling it from Python</h3>")
    out.append(f"<p>Run Python from the entry's folder with its own environment. <code>PYTHONPATH</code> must point at "
               f"<code>shared/</code>, because the adapter imports the task loader from there.</p>")
    out.append(code(f"cd classifiers/{folder}\nPYTHONPATH=../../shared .venv/bin/python my_script.py", "shell"))
    cls_line = EXAMPLES[app.name]
    imports = "from classifier import " + cls_line.split("clf = ")[-1].split("(")[0].split(".")[0] if not supervised \
        else "from classifier import *  # the class and the train loaders"
    if supervised:
        body = f"{imports}\n\n{cls_line}\n\n"
    else:
        labels = "labels = {\n" + "".join(f'    "{k}": "{v}",\n' for k, v in AG_LABELS.items()) + "}\n"
        instr = f'instruction = "{AG_INSTRUCTION}"\n' if "instruction" in cls_line else ""
        body = f"{imports}\n\n{labels}{instr}\n{cls_line}\n\n"
    body += (f'p = clf.predict("{SAMPLE_TEXT}")\nprint(p.label)          # the label ID with the highest score\n'
             f"print(p.confidence)     # a number from 0 to 1\nprint(p.probabilities)  # one score per label ID\n\n"
             f'many = clf.predict_many(["First text.", "Second text."])  # Predictions, in order')
    out.append(code(body, "python"))
    if folder in {"tfidf_logreg", "tfidf_svm", "tfidf_cnb"}:
        out.append("<p>This example fits with the default hyperparameters. The benchmark evaluator instead chooses them "
                   "by cross-validation on the train split and stores the choice in "
                   f"<code>classifiers/{h(folder)}/hyperparameters.json</code>.</p>")

    out.append("<h3>Output format</h3>")
    out.append("<p><code>predict</code> returns a <code>Prediction</code> object with three fields:</p>"
               "<ul><li><code>label</code>: the chosen label ID. It is always a label with the highest score.</li>"
               "<li><code>confidence</code>: a number from 0 to 1. For this entry it is defined as: "
               f"{h(app.confidence)}." + confidence_caveat(app) + "</li><li><code>probabilities</code>: "
               + probabilities_text(app)
               + (f" They sum to 1 within {app.prob_sum_tol}, because the engine rounds them." if app.prob_sum_tol > 1e-3
                  else "")
               + f" Confidence type: {conf_badge(app)}.</li></ul>")

    out.append("<h3>Sample output</h3>")
    printed = sample_block(app)
    cli = f'cd classifiers/{folder}\nPYTHONPATH=../../shared .venv/bin/python classifier.py "{SAMPLE_TEXT}"'
    if app.args and folder not in {"tfidf_logreg"}:
        cli_note = (" The command-line demo always uses the folder's default model, so for this entry the Python call "
                    "above is the reliable way to reproduce the sample.")
    else:
        cli_note = ""
    if printed:
        s = SAMPLES[app.name]
        total = round(s["load_s"] + s["predict_s"], 1)
        out.append(f"<p>This is a real result from this project, recorded on {h(s['date'])} on an Apple Silicon Mac with "
                   f"the AG News labels and the text <code>{h(SAMPLE_TEXT)}</code>. Creating the classifier and making "
                   f"the first prediction took {total} seconds in total. Most of that is loading the model, which some "
                   f"adapters do on the first prediction rather than in the constructor. Later predictions are faster. "
                   f"The command-line demo prints the same values in this form:{cli_note}</p>")
        if app.name == "llmqwen":
            out.append("<p>This sample was recorded with Ollama on the same Mac (<code>LLM_HOST=http://127.0.0.1:11434"
                       "</code>). With <code>LLM_HOST</code> pointing at another machine, the request is the same.</p>")
        out.append(code(cli, "shell"))
        out.append(code(printed, "output"))
    else:
        reason = {"jev": "it is a paid API", "djev": "it sends text to a remote GPU host",
                  "dexy": "it needs an NVIDIA GPU and is not set up on this machine",
                  "llmqwen": "its Ollama host was not reachable when the samples were recorded",
                  "winnow": "its server was not running when the samples were recorded",
                  "OpenJev": "its server was not running when the samples were recorded",
                  "nimble": "the merged checkpoint for the pinned revision was missing (run its setup.sh again)",
                  "clm": "loading the 8B encoder and making one prediction took longer than the 20-minute limit"}.get(app.name,
                  "it could not be run when the samples were recorded")
        out.append(f"<p>No sample was recorded for this entry, because {h(reason)}. The output has the same structure "
                   f"as every other entry. Run the command-line demo to see a result:</p>")
        out.append(code(cli, "shell"))
    out.append(wire_format(app))
    out.append("<h3>Benchmark output</h3>")
    out.append(benchmark_output())
    return "".join(out)


def benchmark_output() -> str:
    return ('<p>The benchmark evaluator writes one prediction CSV per run to the entry\'s results folder, in the format '
            'described in <a href="index.html#shared">What every entry shares</a>.</p>')


def choice_response(app, by_name: bool) -> str:
    s = SAMPLES.get(app.name) or SAMPLES.get("OpenJev")
    names = {"world": "World", "sports": "Sports", "business": "Business", "sci_tech": "Science/Technology"}
    if s:
        probs = s["probabilities"]
        choice, conf = s["label"], s["confidence"]
    else:
        probs, choice, conf = {"business": 0.9, "world": 0.05, "sci_tech": 0.03, "sports": 0.02}, "business", 0.9
    key = (lambda k: names.get(k, k)) if by_name else (lambda k: k)
    lines = ",\n".join(f'        "{key(k)}": {v}' for k, v in probs.items())
    return ('{\n  "answers": {\n    "label": {\n'
            f'      "choice": "{key(choice)}",\n      "confidence": {conf},\n      "probabilities": {{\n{lines}\n'
            '      }\n    }\n  },\n  "usage": {"input_tokens": ..., "output_tokens": ...}\n}')


def wire_format(app) -> str:
    folder = folder_of(app)
    criteria = ",\n".join(f'        "{k}": "{v}"' for k, v in AG_LABELS.items())
    if folder in {"OpenJev", "djev", "winnow"}:
        url = {"OpenJev": "http://127.0.0.1:8080", "djev": "http://127.0.0.1:8000 (the tunnel)",
               "winnow": "http://127.0.0.1:8091"}[folder]
        model = {"OpenJev": '  "model": "openjev-latest",\n', "winnow": '  "model": "Winnow-12B",\n', "djev": ""}[folder]
        crit = criteria
        if folder == "winnow":
            names = {"world": "World", "sports": "Sports", "business": "Business", "sci_tech": "Science/Technology"}
            crit = ",\n".join(f'        "{names[k]}": "{v}"' for k, v in AG_LABELS.items())
        extra = ',\n  "options": {"seed": 0, "samples": 1}' if folder == "djev" else ""
        body = (f'POST {url.split(" ")[0]}/v1/systemone\nContent-Type: application/json\n'
                f'Authorization: Bearer <key>   (only when an API key is set)\n\n{{\n{model}  "state": "{SAMPLE_TEXT}",\n'
                f'  "questions": {{\n    "label": {{\n      "type": "choice",\n      "instructions": "{AG_INSTRUCTION}",\n'
                f'      "criteria": {{\n{crit}\n      }}\n    }}\n  }}{extra}\n}}')
        health = {"OpenJev": "GET /v1/models", "djev": "GET /ready", "winnow": "GET /health"}[folder]
        named = (" Winnow keys the options by label name (for example <code>Science/Technology</code>), and the adapter "
                 "maps the names back to label IDs." if folder == "winnow" else "")
        return ("<h3>What goes over the wire</h3>"
                f"<p>The Python class sends one HTTP request per text to the server at {h(url)}. The request is a "
                f"\"choice question\" in the Jev format: the text goes in <code>state</code>, and the labels go in "
                f"<code>criteria</code> as ID and description pairs.{named} Before a run, the adapter checks the "
                f"server with <code>{h(health)}</code>.</p>" + code(body, "http")
                + "<p>The server replies with the chosen option, its confidence and the probability of every option"
                + (" (the numbers below are from this entry's recorded sample):</p>" if app.name in SAMPLES else
                   ". No sample was recorded for this entry, so the numbers below only show the structure:</p>")
                + code(choice_response(app, folder == "winnow"), "json")
                + "<p>Failed requests (connection errors, HTTP 429 and server errors) are retried with a growing "
                  "delay. You can call the server from any language with the same JSON.</p>")
    if folder == "jev":
        return ("<h3>What goes over the wire</h3><p>The adapter uses the TypeSafe SDK instead of raw HTTP. The call "
                "below is what <code>JEVClassifier.predict</code> does. The text goes in <code>state</code>, and each "
                "label ID with its description goes in <code>criteria</code>.</p>"
                + code('from typesafe_sdk import TypeSafeClient, Choice\n\n'
                       'with TypeSafeClient(api_key=KEY, model="jev-latest") as client:\n'
                       '    response = client.system_one(\n'
                       f'        state="{SAMPLE_TEXT}",\n'
                       '        questions={"label": Choice(instructions=instruction, criteria=labels)},\n'
                       '    )\n\nanswer = response.choices["label"]\n'
                       'answer.choice         # the chosen label ID\nanswer.confidence     # Jev\'s own score\n'
                       'answer.probabilities  # one probability per label ID, rounded to 2 decimals', "python")
                + "<p>The model name comes from <code>JEV_MODEL</code> in <code>.env</code> "
                  "(default <code>jev-latest</code>). <code>predict_many</code> sends up to 16 requests at once, and the "
                  "SDK retries HTTP 429 responses.</p>")
    if folder == "llm":
        body = ('POST http://LLM_HOST/api/chat\n\n{\n  "model": "qwen3.8:27b-q4_K_M",\n  "stream": false,\n'
                '  "think": false,\n  "logprobs": true,\n  "top_logprobs": 20,\n'
                '  "options": {"temperature": 0, "seed": 0, "num_ctx": <LLM_NUM_CTX>},\n'
                '  "format": {"type": "object", "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},'
                ' "required": ["answer"]},\n  "messages": [\n    {"role": "system", "content": "<fixed system prompt>"},\n'
                f'    {{"role": "user", "content": "{AG_INSTRUCTION}\\n\\nText: {SAMPLE_TEXT}\\n\\nOptions:\\n'
                'A) World: International news, ...\\nB) Sports: ...\\nC) Business: ...\\nD) Science/Technology: ..."}\n'
                '  ]\n}')
        return ("<h3>What goes over the wire</h3><p>The adapter sends one Ollama chat request per text. The options are "
                "lettered, and a JSON schema limits the reply to one of those letters. The request asks for the "
                "log-probabilities of the top 20 tokens.</p>" + code(body, "http")
                + "<p>Ollama replies with <code>message.content</code> such as <code>{\"answer\": \"C\"}</code> and a "
                  "<code>logprobs</code> list. The adapter takes the log-probabilities of the letters at the answer "
                  "token, converts them to probabilities, and normalizes them over the options. If the most probable "
                  "letter differs from the written one, the most probable letter is used.</p>")
    if folder in {"embedsim", "embedsup"}:
        return ("<h3>What goes over the wire</h3><p>Only the embeddings come from Ollama. The adapter sends texts in "
                "batches to the embedding endpoint and does the rest of the work in Python.</p>"
                + code('POST http://OLLAMA_HOST/api/embed\n\n{"model": "bge-m3:567m", "input": ["' + SAMPLE_TEXT
                       + '", "..."]}\n\nresponse: {"embeddings": [[0.0123, -0.0456, ...], ...]}   '
                       '# one 1024-number vector per input', "http")
                + ("<p>For embedsim the label descriptions are embedded with the template "
                   "<code>{name} news: {description}</code> from the task manifest, and each text gets "
                   "<code>softmax(cosine / T)</code> over the labels.</p>" if folder == "embedsim" else
                   "<p>Training embeddings are cached in <code>results/cache/</code> together with the model tag and "
                   "its Ollama digest. A re-pulled model is embedded again.</p>"))
    return ("<h3>What goes over the wire</h3><p>Nothing. The model runs inside the Python process, so there is no "
            "server or HTTP request. The only network access is the one-time download during setup.</p>")


SHARED_DATA = ("<p>Every entry receives the same data in the same way, which is what makes the results comparable.</p><ol>"
            "<li><code>shared/task.py</code> loads the task manifest, <code>data/tasks/ag_news.json</code> or "
            "<code>data/tasks/banking77.json</code>.</li>"
            "<li>On first use it downloads the CSV files named in the manifest, if they are not present, and verifies "
            "their SHA-256 checksums.</li>"
            "<li>The raw columns are converted to exactly two: <code>text</code> and <code>label</code>. For AG News the "
            "text is the title and the description joined as <code>{title}. {description}</code>. Banking77 uses its "
            "text column.</li>"
            "<li>The runner draws the same label-balanced test sample for every entry, chosen by <code>--n</code> and "
            "<code>--seed</code>.</li>"
            "<li>The adapter receives the label IDs, names, descriptions and the task instruction. Supervised models "
            "fit on the train split. Zero-shot models fit nothing that changes their decision. A few (confidence type "
            "<i>task-calibrated</i>) fit one confidence temperature on a fixed labeled sample drawn from the train "
            "split: their accuracy remains zero-shot, but their confidence and calibration metrics are "
            "task-calibrated.</li></ol>"
            "<table><thead><tr><th>Task</th><th>Labels</th><th>Text</th><th>Calibration sample</th>"
            "<th>Supervised sample</th></tr></thead><tbody>"
            "<tr><td>AG News</td><td>4 topics</td><td>title and description</td><td>100 train rows per label</td>"
            "<td>1,000 rows per label</td></tr>"
            "<tr><td>Banking77</td><td>77 intents</td><td>customer message</td><td>5 train rows per label</td>"
            "<td>30 rows per label</td></tr></tbody></table>"
            "<p>To use your own data, copy a manifest in <code>data/tasks/</code>, give the train and test files or URLs "
            "with their checksums, map every raw label to an ID, and write a short description for each label that "
            "separates it from the others. Only single-label classification is supported.</p>"
            "<h3>Benchmark output</h3>"
            "<p>The benchmark evaluator writes a CSV file of predictions to the entry's results folder. It has one row "
            "per test text and these columns: <code>sample_id</code>, <code>text</code>, <code>label</code> (the true "
            "label), <code>pred</code>, <code>confidence</code>, and one <code>p_&lt;label&gt;</code> column per label. "
            "<code>runner/validate.py</code> checks that the rows match the expected sample, that every label is "
            "known, that each row's probabilities sum to 1, and that <code>pred</code> has the highest "
            "probability.</p>"
            "<h3>Running options</h3>"
            "<p><code>--n 0</code> runs the whole test split. <code>--seed N</code> draws a different but reproducible "
            "sample. <code>--dry-run</code> prints the resolved command without running it. Options for the adapter "
            "itself go after <code>--</code>. <code>./run all</code> runs entries and then rebuilds the report; "
            "<code>./run report</code> rebuilds it from results that already exist.</p>"
            "<h3>Reading the results</h3>"
            "<p>Accuracy tells you how often the top label was correct, not which labels a model confuses; the report "
            "adds a confusion matrix, per-class scores, calibration, latency, memory and selective prediction. Compare "
            "two runs only when the task, sample size, seed and model revision are the same. A high confidence value "
            "does not guarantee a correct answer.</p>"
            "<h3>Common problems</h3><ul>"
            "<li><code>./run list --task ag_news</code> shows whether each entry is ready and, if not, why.</li>"
            "<li>Run <code>./run setup &lt;name&gt;</code> again after changing Python or dependencies.</li>"
            "<li><code>.env.example</code> lists each entry's host, model, device, key and context settings.</li>"
            "<li><code>ModuleNotFoundError: No module named 'task'</code> (or <code>'training'</code>) when running "
            "<code>classifier.py</code> directly means <code>PYTHONPATH=../../shared</code> is missing.</li>"
            "<li>Out-of-memory errors, truncated options and unreachable services are setup problems, not measures of "
            "model quality.</li></ul>")


def approach_text(app) -> str:
    """The entry's comparison category on each task (the same categories as the score reports)."""
    by_task = {t: app.approach_on(t) for t in ("ag_news", "banking77")}
    if len(set(by_task.values())) == 1:
        return by_task["ag_news"]
    return "; ".join(f"{v} on {TASK_NAMES[t]}" for t, v in by_task.items())


TASK_NAMES = {"ag_news": "AG News", "banking77": "Banking77"}


def section_data(app) -> str:
    """What this entry learns from each task's data; the shared preparation steps are on the index page."""
    kind = app.supervision
    if app.trained_on_tasks:
        seen = " and ".join(TASK_NAMES.get(t, t) for t in app.trained_on_tasks)
        every = set(app.trained_on_tasks) >= set(TASK_NAMES)
        fits = (f"Nothing is fitted for the benchmark. The released checkpoint already saw {seen} data in its own "
                "training, so the reports list it as an in-domain checkpoint on "
                + ("both bundled tasks." if every else "that task and as zero-shot on the other."))
    elif kind == "supervised":
        fits = "It is fitted on the task's train split before any test text is classified."
    elif confidence_kind(app) == "task":
        fits = ("Its decision uses only the label descriptions. One confidence temperature is fitted on the fixed "
                "labeled calibration sample from the train split, so its accuracy is zero-shot but its confidence "
                "figures are task-calibrated.")
    else:
        fits = "Nothing is fitted on task data: it decides from the label names and descriptions alone."
    limit = (f" It accepts at most {app.max_labels} labels per request, so Banking77 (77 labels) runs only through a "
             "shortlist entry." if app.max_labels and app.max_labels < 77 and not app.shortlist else "")
    if app.shortlist:
        limit = (f" An embedding model first narrows the labels to {app.shortlist} candidates, and this entry chooses "
                 "among them.")
    return (f"<p>Approach in the score reports: <b>{h(approach_text(app))}</b>. {fits}{limit}</p>"
            '<p>The data preparation, sampling and calibration sample sizes are the same for every entry: see '
            '<a href="index.html#shared">What every entry shares</a>.</p>')


def section_run(app) -> str:
    task = "banking77" if app.shortlist else "ag_news"
    cmds = f"./run {app.name} --task {task} --n 200"
    if not app.shortlist and not (app.max_labels and app.max_labels < 77):
        cmds += f"\n./run {app.name} --task banking77 --n 200"
    return (code(cmds, "shell")
            + "<h3>Inside the full benchmark</h3>"
            + code(f"./run all --task {task} --only {app.name}\n./run report --task {task}", "shell")
            + '<p>Options such as <code>--n</code>, <code>--seed</code> and <code>--dry-run</code> are described in '
              '<a href="index.html#shared">What every entry shares</a>.</p>')


def section_training(app) -> str:
    folder = folder_of(app)
    if app.supervision == "supervised":
        if folder == "embedsup":
            detail = ("The BGE-M3 encoder is not changed. The evaluator embeds a fixed, stratified sample of the train "
                      "split and fits either class centroids or a logistic regression on the vectors. For the centroid "
                      "method, the temperature is fitted on a held-out part of that sample.")
        else:
            detail = ("The evaluator builds the TF-IDF features and fits the estimator on the train split. The "
                      "hyperparameters are chosen by 3-fold stratified cross-validation on the train split only, and the "
                      "fit is cached together with a fingerprint of the data and the scikit-learn version.")
        return (note("train", "Training", f"<p>Training is required, and the evaluator does it automatically. {detail}</p>")
                + "<ol><li>Keep the labels and examples in the task manifest and its train file.</li>"
                  "<li>Run the benchmark command. Fitting happens before any test text is classified.</li>"
                  f"<li>The chosen hyperparameters are in <code>classifiers/{h(folder)}/hyperparameters.json</code>, and "
                  "cached fits are in <code>results/cache/</code>.</li>"
                  "<li>After the data changes, run again. The fingerprint prevents an old fit from being reused. Delete "
                  "the cache only when you want to force a new fit.</li></ol>"
                + "<p>Never tune on <code>data/*_test.csv</code>. Choose settings with cross-validation or a validation "
                  "part of the train split, and evaluate the final choice once on the test split.</p>")
    if app.trained_on_tasks:
        tasks = " and ".join(f"<code>{h(x)}</code>" for x in app.trained_on_tasks)
        return (note("warn", "In-domain", f"<p>The released checkpoint was trained on data from {tasks}. Its results "
                                          f"on those tasks are not zero-shot results.</p>")
                + "<p>You do not need to fine-tune it for this benchmark. During setup, an overlap check compares our test "
                  "texts with the model's training data and writes <code>results/overlap.json</code>. Test texts that "
                  "also appear in the dataset's own train split do not count, because the supervised baselines see them "
                  "too. If other test texts appear in the training data, or the check has not run, the runner skips the "
                  "task.</p>"
                + "<p>For a new domain, first try clearer label descriptions. If you retrain, use the model author's "
                  "own training code, keep this project's test split out of training, prefer LoRA or another "
                  "parameter-efficient method for large models, choose epochs and learning rate on validation data, "
                  "calibrate on a separate part of the train split, and register the result as a new entry so its "
                  "results stay separate from the release.</p>")
    if folder in RELEASED:
        return (note("", "No fine-tuning", "<p>This adapter uses a released, already trained checkpoint and applies it "
                                           "to the label set. No training is part of the benchmark.</p>")
                + "<p>If the results are poor on your data, first improve the label names and descriptions in the "
                  "manifest. Fine-tuning is possible but is outside this adapter. If you do it, use the training "
                  "procedure from the official project page, keep the test data out of training, use a train and "
                  "validation split, prefer a parameter-efficient adapter for models with billions of parameters, "
                  "calibrate probabilities on held-out train data, and register the new checkpoint as a separate "
                  "entry. Do not replace the pinned release, because earlier results could then no longer be "
                  "reproduced.</p>")
    return (note("", "No fine-tuning", "<p>The method is zero-shot. Only the label text and, where used, a calibration "
                                       "temperature fitted on train data are adapted.</p>")
            + "<p>Write distinct label descriptions before trying a different model. Where this entry uses a "
              "temperature, it is one positive scalar applied uniformly to all label logits: it changes the "
              "normalized confidence values but not which label has the largest score. If you need a model trained on your domain, add it as a separate entry "
              "and evaluate it on a test split it has never seen.</p>")


def section_results(app) -> str:
    extra = ""
    if app.trained_on_tasks:
        extra = (f" On {' and '.join(TASK_NAMES.get(t, t) for t in app.trained_on_tasks)} the reports list it as an "
                 "in-domain checkpoint, not as zero-shot.")
    return (f"<p>This entry's confidence is <strong>{h(app.confidence)}</strong>.{confidence_caveat(app)}"
            + extra + "</p>"
            + ("<p>It runs on a remote service, so its timings include the network and hardware outside the test "
               "machine.</p>" if app.remote else "")
            + '<p>How to read accuracy, confidence and the other measures is described once in '
              '<a href="index.html#shared">What every entry shares</a>.</p>'
            f'<p class="reports">{report_links()}</p>')


def section_trouble(app) -> str:
    items = [f"<code>./run {h(app.name)} --dry-run</code> shows the arguments the runner passes to this entry.",
             f"Run <code>./run setup {h(app.name)}</code> again after changing Python or dependencies."]
    if app.max_labels and app.max_labels < 77:
        items.append("If Banking77 is refused because it has too many labels, use the <code>_shortlist</code> entry "
                     "when one exists.")
    if app.needs in {"openjev_server", "djev_server", "winnow_server"} or folder_of(app) in {"OpenJev", "djev", "winnow"}:
        items.append("\"Cannot reach\" or \"not ready\" means the server is not running or is still loading the model.")
    if app.needs == "ollama" or app.ollama_model or app.extra_ollama:
        items.append("\"Ollama not reachable\" means Ollama is not running at the configured host. \"not pulled\" "
                     "means the model tag must be pulled first.")
    return ("<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
            + '<p>Problems common to every entry are listed in <a href="index.html#shared">What every entry '
              'shares</a>.</p>')


SECTIONS = [
    ("about", "About this model", section_about),
    ("requirements", "What you need", section_requirements),
    ("download", "Getting the model", section_download),
    ("install", "Installing from a clean checkout", section_install),
    ("io", "Sending data and reading results", section_io),
    ("data", "What it learns from the task", section_data),
    ("run", "Running the benchmark", section_run),
    ("training", "Training and adaptation", section_training),
    ("results", "Its results and confidence", section_results),
    ("trouble", "When something goes wrong", section_trouble),
]


def model_page(app, index: int) -> str:
    model, maker, _, _, summary = facts(app)
    title = app.display_title
    toc = "".join(f'<li><a href="#{sid}"><span>{i:02d}</span>{h(name)}</a></li>'
                  for i, (sid, name, _) in enumerate(SECTIONS, 1))
    body = "".join(f'<section id="{sid}"><p class="secnum">{i:02d}</p><h2>{h(name)}</h2>{fn(app)}</section>'
                   for i, (sid, name, fn) in enumerate(SECTIONS, 1))
    meta = [("Registry name", f"<code>{h(app.name)}</code>"), ("Approach", h(approach_text(app))),
            ("Runs on", h(app.runs_on)), ("Label limit", h(label_limit(app))), ("Confidence", conf_badge(app))]
    meta_html = "".join(f'<div><span class="num">{i:03d}</span><span class="k">{k}</span><span class="v">{v}</span></div>'
                        for i, (k, v) in enumerate(meta, 1))
    prev_app = APPS[index - 1] if index > 0 else None
    next_app = APPS[index + 1] if index + 1 < len(APPS) else None
    pager = '<nav class="pager wrap" aria-label="Previous and next model">'
    pager += (f'<a href="{h(prev_app.name)}.html"><small>Previous</small>{h(prev_app.display_title)}</a>' if prev_app
              else '<a href="index.html"><small>Back</small>All models</a>')
    pager += (f'<a class="next" href="{h(next_app.name)}.html"><small>Next</small>{h(next_app.display_title)}</a>'
              if next_app else '<a class="next" href="index.html"><small>Back</small>All models</a>')
    pager += "</nav>"
    return (head(f"{app.name} model guide", f"Guide to {title}: requirements, input and output, setup, running and "
                                            f"results in the Classification benchmark.")
            + topbar()
            + f'<main><div class="pagehead"><div class="wrap"><p class="crumb"><a href="index.html">Models</a>'
              f'<span>/</span><span>{index + 1:03d}</span><span>/</span><span>{h(app.name)}</span></p>'
              f'<h1>{h(title)}</h1><p class="lede">{h(summary)}</p></div></div>'
            + f'<div class="band"><div class="wrap meta">{meta_html}</div></div>'
            + f'<div class="wrap doc"><aside class="toc"><p class="toc-title">Contents</p><ol>{toc}</ol></aside>'
              f'<article>{body}</article></div>{pager}</main>'
            + footer())


# ---------------------------------------------------------------------------------------------------------------
# Neural networks trained from scratch: their own page layout and their own index group
# ---------------------------------------------------------------------------------------------------------------
# An entry is "implemented" when it
# is in runner/apps.py and "planned" otherwise; both kinds get the page below (not model_page) and sit in one index
# group. When one is implemented, add its EXAMPLES line, record a sample (models/capture_samples.py <name>) and fill
# in its "measured" dict.
SCRATCH_GROUP = ("scratch", "Trained from scratch",
                 "Small neural networks with no pre-trained weights, trained on the task's train split.")
PLANNED_NOTE = " Entries marked Planned are designed but not yet implemented."


def is_implemented(p) -> bool:
    return p["name"] in {a.name for a in APPS}

FASTTEXT_CODE = '''class FastText(nn.Module):
    """Average the vectors of the text's words and word pairs; a linear head scores the labels."""

    def __init__(self, rows, n_labels, dim=64, dropout=0.0):
        super().__init__()
        self.n_labels = n_labels
        self.emb = nn.Embedding(rows, dim, padding_idx=0)   # words + hashed bigram buckets
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, n_labels)

    def forward(self, x):                                   # x: (batch, length) ids, 0 = padding
        mask = (x != 0).unsqueeze(-1).float()
        mean = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.out(self.drop(mean))                    # (batch, labels) logits'''

CNN_CODE = '''class TextCNN(nn.Module):
    """Each filter learns a phrase pattern; max-pooling keeps its strongest match anywhere in the text."""

    def __init__(self, rows, n_labels, dim=128, filters=100, widths=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.n_labels, self.min_len = n_labels, max(widths)
        self.emb = nn.Embedding(rows, dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(dim, filters, w) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(widths), n_labels)

    def forward(self, x):
        if x.size(1) < self.min_len:                        # a text shorter than the widest window
            x = F.pad(x, (0, self.min_len - x.size(1)))
        lengths = (x != 0).sum(1, keepdim=True)             # real tokens; padding sits only at the end
        e = self.emb(x).transpose(1, 2)                     # (batch, dim, length)
        pooled = []
        for conv, w in zip(self.convs, (c.kernel_size[0] for c in self.convs)):
            h = F.relu(conv(e))                             # (batch, filters, windows)
            # keep only the windows the text would have on its own (window 0 always)
            start = torch.arange(h.size(2), device=x.device).unsqueeze(0)
            outside = start > (lengths - w).clamp(min=0)
            h = h.masked_fill(outside.unsqueeze(1), float("-inf"))
            pooled.append(h.max(dim=2).values)              # strongest match per filter
        return self.out(self.drop(torch.cat(pooled, dim=1)))'''

BILSTM_CODE = '''class BiLSTM(nn.Module):
    """A reader with memory, in both directions."""

    def __init__(self, rows, n_labels, dim=128, hidden=128, layers=1, dropout=0.3):
        super().__init__()
        self.n_labels = n_labels
        self.emb = nn.Embedding(rows, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, n_labels)   # last fwd + last bwd + max-pool fwd + max-pool bwd

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.drop(self.emb(x)), lengths, batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)                         # padding is skipped, never read
        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=float("-inf"))
        pooled = out.max(dim=1).values
        last = torch.cat([h[-2], h[-1]], dim=1)                 # final forward and backward states
        return self.out(self.drop(torch.cat([last, pooled], dim=1)))'''

SCRATCH = [
    {
        "name": "scratch_fasttext", "title": "fastText-style bag of embeddings, trained from scratch",
        "short": "fastText-style", "runs_on": "Local, PyTorch CPU", "device": "CPU",
        "env": "SCRATCH_FASTTEXT_DEVICE", "class": "FastText",
        "summary": "The simplest neural text classifier: every word gets a learned vector, the vectors of the text's "
                   "words and word pairs are averaged, and one linear layer scores the labels. It learns everything "
                   "from the task's train split, with no pre-trained weights. Planned, not implemented yet.",
        "diagram": "my    card   has    not   arrived  yet       + pairs: my_card, card_has, has_not, not_arrived, ...\n"
                   " |     |      |      |      |       |\n"
                   "[v1]  [v2]   [v3]   [v4]   [v5]    [v6]      <- one learned vector per word (and per word pair)\n"
                   "  \\_____\\______\\__ average __/_______/\n"
                   "                    |\n"
                   "              [text vector]  ->  linear head  ->  card_arrival",
        "how": "<p>Look up the vector of every word, add the vectors of adjacent word pairs (bigrams), and take the "
               "average. That average is the text vector, and a single linear layer turns it into one score per label. "
               "There is nothing else: no layers in between, no notion of position.</p>"
               "<p><b>Why word pairs.</b> Averaging ignores word order, so \"not arrived\" and \"arrived, not\" would "
               "look the same. Adding a vector for each adjacent pair (\"not_arrived\") restores short phrases at "
               "almost no cost. There are too many possible pairs to give each its own row, so each pair is hashed into "
               "one of a fixed number of buckets (200,000 for AG News, 50,000 for Banking77), the hashing trick from "
               "the fastText paper. Two rare pairs may share a bucket; with enough buckets this rarely matters.</p>"
               "<p><b>The name.</b> It comes from Facebook's fastText library (Joulin et al., 2016, \"Bag of Tricks "
               "for Efficient Text Classification\"), which made this approach popular. This entry reimplements the idea "
               "in about ten lines of PyTorch; it does not use the fastText library.</p>"
               "<p><b>An analogy.</b> It judges a text by its bag of keywords, like TF-IDF, except that it learns that "
               "similar words should count similarly: after training, \"stock\", \"shares\" and \"earnings\" have "
               "nearby vectors.</p>",
        "strengths": ["Fastest to train and to run; trains in minutes on the CPU.",
                      "Almost nothing to tune.",
                      "Very strong on topic classification such as AG News, where the words alone give the topic away."],
        "weaknesses": ["Sees no word order beyond adjacent pairs.",
                       "Cannot tell \"the card arrived but the PIN did not\" from \"the PIN arrived but the card did "
                       "not\".",
                       "The bigram table makes it the largest of the three on disk (about 68 MB for AG News)."],
        "code": FASTTEXT_CODE,
        "settings": [("Embedding size", "64", "64"), ("Bigram hash buckets", "200,000", "50,000"),
                     ("Dropout", "0", "0"), ("Batch size", "128", "32"), ("Peak learning rate", "5e-3", "5e-3"),
                     ("Max epochs (patience)", "8 (2)", "60 (6)"), ("Max tokens", "100", "48"),
                     ("Min word count", "2", "1")],
        "expected": ("~92-92.5%", "~84-88%"),
        "time": ("~30-60 s", "~3-5 min", "under 1 min", "under 1 s"),
        "size": ("~17M parameters, ~68 MB", "~3.4M parameters, ~14 MB"),
        # Measured 2026-11-18 on an Apple M5 (CPU), seed 42 test samples, models trained with the recipe above.
        "measured": {
            "date": "2026-11-18",
            "accuracy": [("Common tier, 1,000 texts", "91.7% (macro F1 0.917)", "88.0% (macro F1 0.879)"),
                         ("Whole test split", "91.0% (6,913 / 7,600)", "88.5% (2,726 / 3,080)"),
                         ("Validation accuracy at the best epoch", "91.2%", "88.4%")],
            "training": [("Train texts (fit + validation)", "108,000 + 12,000", "9,000 + 1,003"),
                         ("Vocabulary", "45,528 words", "2,297 words"),
                         ("Parameters", "15.7M", "3.4M"),
                         ("Epochs run (best)", "4 (2): early stop", "27 (21): early stop"),
                         ("Seconds per epoch", "14", "0.9"),
                         ("Training time", "58 s", "25 s"),
                         ("Temperature T", "1.103", "1.222"),
                         ("Saved model", "60 MB", "13 MB")],
            "speed": [("Batch of 256", "about 45,000 texts/s", "about 87,000 texts/s"),
                      ("One text at a time", "0.04 ms per text", "0.04 ms per text"),
                      ("Loading the saved model", "0.1 s", "0.02 s")],
            "device": "One AG News epoch took 14.1 s on the CPU and 12.9 s on MPS with identical losses, so the "
                      "default stays the CPU.",
            "notes": "On AG News the network fits the train set fast: validation accuracy peaks at epoch 2 while "
                     "training loss keeps falling, and early stopping keeps epoch 2. On the whole test split it lands "
                     "1.7 points below tfidf_logreg (91.0% against 92.7%). On Banking77 it does better than expected: 88% on the common tier, above "
                     "every zero-shot and in-domain decision model there (best: d1, 86.5%), but 3.5 points below "
                     "tfidf_logreg on the same 1,000 texts (91.5%). Every one of the 77 labels is predicted at least "
                     "once.",
        },
    },
    {
        "name": "scratch_cnn", "title": "TextCNN, trained from scratch",
        "short": "CNN", "runs_on": "Local, PyTorch MPS", "device": "MPS (Apple GPU), CPU fallback",
        "env": "SCRATCH_CNN_DEVICE", "class": "TextCNN",
        "summary": "A convolutional network for text in the style of Kim (2014): filters slide over windows of a few "
                   "words and learn to fire on phrases, and max-pooling keeps each filter's strongest match anywhere "
                   "in the text. It learns everything from the task's train split, with no pre-trained weights. "
                   "Planned, not implemented yet.",
        "diagram": "my  card  has  not  arrived  yet\n"
                   "\\____window 1___/\n"
                   "     \\___window 2____/\n"
                   "           \\____window 3_____/\n"
                   "                \\_____window 4_____/       <- a 3-word filter slides along the text\n"
                   "\n"
                   "filter \"has not arrived\":  0.1  0.4  2.7  0.3   -> max-pool -> 2.7\n"
                   "filter \"card declined\":    0.2  0.0  0.1  0.0   -> max-pool -> 0.2\n"
                   "...300 filters in all       -> [300 numbers]   -> linear head -> card_arrival",
        "how": "<p>A <b>filter</b> is a small learned pattern that looks at a window of consecutive words (2 to 5 of "
               "them) and outputs a score for how well the window matches. The filter slides along the text, scoring "
               "every window. One filter may learn to fire on \"has not arrived\", another on \"card was declined\", "
               "another on \"won the match\". Nobody tells it which phrases to look for; training finds the ones that "
               "help predict the labels.</p>"
               "<p><b>Max-pooling.</b> For each filter only its highest score anywhere in the text is kept. That "
               "answers the question \"did this phrase appear anywhere?\", wherever it was. With 100 filters for each "
               "of three window widths, every text becomes 300 numbers, and number <i>k</i> means \"how strongly did "
               "phrase pattern <i>k</i> appear?\". A linear layer turns those into label scores.</p>"
               "<p><b>The name.</b> It is the same idea as the convolutional networks used for images, where the filter "
               "slides over pixels; here it slides over words. The design follows Kim (2014), \"Convolutional Neural "
               "Networks for Sentence Classification\".</p>"
               "<p><b>Padding.</b> Texts in a batch are padded to the same length, which creates extra windows that run "
               "past a text's last word. Classified alone, the text would not have those windows, and they can win the "
               "max-pooling, so they are masked: only the windows that end inside the text are kept (window 0 always, "
               "for texts shorter than the window). A text then gets the same answer alone or in a batch. The setup "
               "check caught exactly this: masking only windows that <i>start</i> in the padding, as first planned, "
               "left differences of up to 0.1 in the logits.</p>",
        "strengths": ["Captures short phrases and local negation (\"not arrived\").",
                      "Still fast: the filters run in parallel, so it suits the GPU.",
                      "A good middle step between the bag of words and a sequence reader."],
        "weaknesses": ["Cannot connect words further apart than its widest window.",
                       "More settings than fastText-style: filter widths, number of filters, dropout."],
        "code": CNN_CODE,
        "settings": [("Embedding size", "128", "128"), ("Filters x widths", "100 x 3, 4, 5", "100 x 2, 3, 4"),
                     ("Dropout", "0.5", "0.5"), ("Batch size", "64", "32"), ("Peak learning rate", "1e-3", "2e-3"),
                     ("Max epochs (patience)", "6 (2)", "40 (5)"), ("Max tokens", "100", "48"),
                     ("Min word count", "2", "1")],
        "expected": ("~91-92%", "~85-89%"),
        "time": ("~1-2 min", "~5-10 min", "~1 min", "~1 s"),
        "size": ("~8.5M parameters, ~34 MB", "~0.5M parameters, ~2 MB"),
        # Measured 2026-11-18 on an Apple M5 (MPS), seed 42 test samples, models trained with the recipe above.
        "measured": {
            "date": "2026-11-18",
            "accuracy": [("Common tier, 1,000 texts", "91.5% (macro F1 0.915)", "89.4% (macro F1 0.894)"),
                         ("Whole test split", "90.9% (6,910 / 7,600)", "88.4% (2,723 / 3,080)"),
                         ("Validation accuracy at the best epoch", "91.3%", "89.1%")],
            "training": [("Train texts (fit + validation)", "108,000 + 12,000", "9,000 + 1,003"),
                         ("Vocabulary", "45,528 words", "2,297 words"),
                         ("Parameters", "6.0M", "0.43M"),
                         ("Epochs run (best)", "6 (6): the limit", "20 (15): early stop"),
                         ("Seconds per epoch", "21 (MPS)", "1.5 (MPS)"),
                         ("Training time", "127 s", "32 s"),
                         ("Temperature T", "1.433", "1.959"),
                         ("Saved model", "23 MB", "1.7 MB")],
            "speed": [("Batch of 256", "about 20,000 texts/s", "about 27,000 texts/s"),
                      ("One text at a time", "0.6 ms per text", "1.2 ms per text"),
                      ("Loading the saved model", "0.1 s", "0.03 s")],
            "device": "One AG News epoch took 20.5 s on MPS and 144.5 s on the CPU, so the CNN is the one model of "
                      "the three where the Apple GPU pays off (7x); the default is MPS. One text at a time costs more "
                      "than on the CPU-only models because every call goes to the GPU and back.",
            "notes": "On AG News the CNN ends level with the fastText-style model (91.5% against 91.7% on the common "
                     "tier, 90.9% against 91.0% on the whole split): phrases add little to a topic task that keywords "
                     "already solve. Its best epoch was the last one allowed (6), with validation accuracy still "
                     "creeping up (91.28% to 91.31%), so one or two more epochs might add a little. On Banking77 it "
                     "is the best of the from-scratch models on the common tier, 89.4%, ahead of every decision "
                     "model there (best: d1, 86.5%), though 2.1 points below tfidf_logreg on the same texts (91.5%). On the "
                     "whole Banking77 test split it scores 88.4%. Every one of the 77 labels is predicted at least "
                     "once.",
        },
    },
    {
        "name": "scratch_bilstm", "title": "BiLSTM, trained from scratch",
        "short": "BiLSTM", "runs_on": "Local, PyTorch CPU",
        "device": "CPU (measured: packed LSTMs gain nothing on the Apple GPU)",
        "env": "SCRATCH_BILSTM_DEVICE", "class": "BiLSTM",
        "summary": "A bidirectional long short-term memory network: it reads the text one word at a time, forwards and "
                   "backwards, carrying a memory vector that summarises what it has read so far. It learns everything "
                   "from the task's train split, with no pre-trained weights. Planned, not implemented yet.",
        "diagram": "forwards:   my -> card -> has -> not -> arrived -> yet   -> memory: \"a card, negated, arrival\"\n"
                   "backwards:  yet -> arrived -> not -> has -> card -> my   -> memory: \"arrival, negated, a card\"\n"
                   "\n"
                   "[final forward memory | final backward memory | max over every step]\n"
                   "            -> one text vector -> linear head -> card_arrival",
        "how": "<p>An <b>LSTM</b> reads the text one word at a time and keeps a running memory vector. At each word, "
               "learned gates decide what to remember, what to forget and what to update. After \"my card has not\" "
               "its memory holds something like \"about a card, negated\"; after \"arrived\" it can combine the two. "
               "That is how it understands word order and relationships across a sentence.</p>"
               "<p><b>Bidirectional.</b> Two LSTMs run: one left to right, one right to left. Every word is then seen "
               "with both its left and its right context. This entry joins the final memory of each direction with the "
               "maximum of every memory value over all positions, and a linear layer scores the labels.</p>"
               "<p><b>The name.</b> Long Short-Term Memory (Hochreiter and Schmidhuber, 1997) was designed to keep "
               "information over long sequences, which earlier recurrent networks forgot. BiLSTMs were the standard "
               "for text before Transformers took over around 2018.</p>"
               "<p><b>Padding.</b> The texts are packed by length, so padding is never read, and a text gets the same "
               "answer alone or in a batch.</p>",
        "strengths": ["Understands word order and longer-range relationships, such as \"I ordered a new card two "
                      "weeks ago and it still hasn't come\".",
                      "The most expressive of the three."],
        "weaknesses": ["Slowest: it must read word by word and cannot process all positions at once.",
                       "More fiddly to train, and on keyword-driven tasks such as AG News the extra power buys little."],
        "code": BILSTM_CODE,
        "settings": [("Embedding size", "128", "128"), ("Hidden size per direction", "128", "128"),
                     ("Layers", "1", "1"), ("Dropout", "0.3", "0.3"), ("Batch size", "64", "32"),
                     ("Peak learning rate", "1e-3", "2e-3"), ("Max epochs (patience)", "6 (2)", "40 (5)"),
                     ("Max tokens", "100", "48"), ("Min word count", "2", "1")],
        "expected": ("~91-92%", "~85-89%"),
        "time": ("~2-5 min", "~15-30 min", "~2-3 min", "~1-3 s"),
        "size": ("~8.6M parameters, ~35 MB", "~0.6M parameters, ~2.5 MB"),
        # Measured 2026-11-18 on an Apple M5 (CPU), seed 42 test samples, models trained with the recipe above.
        "measured": {
            "date": "2026-11-18",
            "accuracy": [("Common tier, 1,000 texts", "92.3% (macro F1 0.923)", "89.1% (macro F1 0.891)"),
                         ("Whole test split", "91.6% (6,965 / 7,600)", "88.9% (2,738 / 3,080)"),
                         ("Validation accuracy at the best epoch", "92.0%", "90.9%")],
            "training": [("Train texts (fit + validation)", "108,000 + 12,000", "9,000 + 1,003"),
                         ("Vocabulary", "45,528 words", "2,297 words"),
                         ("Parameters", "6.1M", "0.60M"),
                         ("Epochs run (best)", "6 (6): the limit", "20 (15): early stop"),
                         ("Seconds per epoch", "105 (CPU)", "3.6 (CPU)"),
                         ("Training time", "630 s (10.5 min)", "71 s"),
                         ("Temperature T", "1.513", "1.561"),
                         ("Saved model", "23 MB", "2.3 MB")],
            "speed": [("Batch of 256", "about 7,300 texts/s", "about 20,000 texts/s"),
                      ("One text at a time", "0.7 ms per text", "0.3 ms per text"),
                      ("Loading the saved model", "0.04 s", "0.004 s")],
            "device": "One AG News epoch took 102 s on the CPU and 109 s on MPS: packed LSTMs gain nothing on the "
                      "Apple GPU, so the default was changed from the planned MPS to the CPU.",
            "notes": "The BiLSTM is the most accurate of the three from-scratch models on AG News: 92.3% on the "
                     "common tier and 91.6% on the whole split, against 91.7% and 91.0% for the fastText-style "
                     "model, about 0.6 points, which is within the noise of a 1,000-text sample but consistent on "
                     "the 7,600-text split. Reading in order helps a little on a topic task, at about 10 times the "
                     "training time. Its best epoch was the last one allowed (6), with validation accuracy still "
                     "rising slightly. On Banking77 it scores 89.1% on the common tier and 88.9% on the whole split, "
                     "the best of the three on the whole split, 2.4 points below tfidf_logreg on the common tier (91.5%) "
                     "and ahead of every decision model there (best: d1, 86.5%). On AG News it is in the report's "
                     "leading tier with tfidf_logreg (93.2%) and laya_typed (92.7%). Every one of the 77 labels "
                     "is predicted at least once.",
        },
    },
]


def table(head_cells: list[str], rows: list[list[str]]) -> str:
    return ("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in head_cells) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>")


def ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{h(x)}</li>" for x in items) + "</ul>"


SCRATCH_VS_FINETUNE = table(
    ["", "Training from scratch (this entry)", "Fine-tuning"],
    [["Starting weights", "Random", "Pre-trained (ModernBERT has already read about 2 trillion tokens)"],
     ["What the model knows at the start", "Nothing, not even that \"card\" and \"cards\" are related",
      "Grammar, word meanings, a lot of world knowledge"],
     ["What our data teaches it", "Everything", "Only the labels"],
     ["Data needed", "A lot", "Little: Banking77's ~130 examples per label is enough"],
     ["Typical AG News accuracy", "~91-92.5%", "~94.5-95.5%"],
     ["Typical Banking77 accuracy", "Clearly lower (too few examples per label)", "~93-94%"]])

SPECTRUM = table(
    ["Kind", "What is trained", "Entries here"],
    [["Classic machine learning from scratch", "Word weights on TF-IDF features, from our data only",
      "<code>tfidf_logreg</code>, <code>tfidf_svm</code>, <code>tfidf_cnb</code>"],
     ["Small neural network from scratch", "Every weight, starting from random",
      "<code>scratch_fasttext</code>, <code>scratch_cnn</code>, <code>scratch_bilstm</code>"],
     ["Frozen pre-trained features and a trained head", "Only a small classifier on top of vectors from a model "
      "that is never changed", "<code>embedsup_centroid</code>, <code>embedsup_logreg</code> (bge-m3)"],
     ["Full fine-tuning", "Every weight of a pre-trained model, plus a new head",
      "<code>finetune_modernbert</code>, <code>finetune_deberta</code>; released checkpoints fine-tuned by their "
      "authors: <code>decider</code>, <code>kev</code>, <code>strands_decider</code>"],
     ["Parameter-efficient fine-tuning (LoRA)", "Small adapter matrices added to a frozen pre-trained model",
      "Releases by their authors: <code>kev</code>, <code>modernbert_jev</code>"]])

THREE_COMPARED = table(
    ["", "fastText-style", "CNN", "BiLSTM"],
    [["Sees word order?", "No, only adjacent pairs", "Locally, within 2-5 word windows", "Yes, the whole sentence"],
     ["Mental model", "Bag of words", "Phrase detector", "Reader with memory"],
     ["Speed", "Fastest", "Fast", "Slower"],
     ["Lines of PyTorch", "~10", "~20", "~20"],
     ["Best at", "Topics (AG News)", "Topics plus key phrases", "Meaning that depends on word order"]])


def scratch_status(p) -> str:
    n = h(p["name"])
    m = p.get("measured") if is_implemented(p) else None
    lead = ""
    if not is_implemented(p):
        lead = note("warn", "Planned", "<p>This entry is designed but not yet implemented; it cannot be run.</p>")
    rows = [("Registry name", f"<code>{n}</code>"), ("Folder", f"<code>classifiers/{n}/</code>"),
            ("Architecture", f"<code>{h(p['class'])}</code> in <code>model.py</code>"),
            ("Pre-trained weights", "None. Every weight starts random and is learned from the task's train split."),
            ("Training data", "The task's train split: a stratified 90% to fit, 10% for validation"),
            ("Trained models", f"<code>classifiers/{n}/models/&lt;task&gt;/</code>"),
            ("Shared code", "<code>shared/textnn.py</code>: tokenizer, vocabulary, training loop, calibration, model "
                            "cache"),
            ("Device", h(p["device"]) + f" (<code>{h(p['env'])}</code>)"),
            ("Confidence", "max p after temperature scaling on the validation split")]
    if m:
        rows.append(("Accuracy, 1,000 test texts", f"AG News {h(m['accuracy'][0][1])}; Banking77 "
                                                   f"{h(m['accuracy'][0][2])}"))
    return lead + kv_table(rows) + (f"<p>The code is in <code>classifiers/{n}/</code> and <code>shared/textnn.py</code>; "
                                     "the measurements are in the reports linked at the top of this page.</p>")


def scratch_sample(p) -> str:
    """The recorded demo output (models/samples.json), for an implemented entry."""
    s = SAMPLES.get(p["name"])
    if not s:
        return ""
    width = max(len(k) for k in s["probabilities"])
    printed = (f"label:      {s['label']}\nconfidence: {s['confidence']:.2f}\n"
               + "\n".join(f"  {k:<{max(width, 10)}} {v:.3f}" for k, v in s["probabilities"].items()))
    return ("<h3>A recorded prediction</h3>"
            + f"<p>Recorded on {h(s['date'])} with the AG News model, through the same call the benchmark uses:</p>"
            + code(f"from classifier import trained\nclf = trained()   # trains on the train split, or loads "
                   f"models/ag_news/\np = clf.predict({SAMPLE_TEXT!r})", "python")
            + code(printed, "output")
            + "<p>Every prediction has the same three parts as every other entry: the label, the confidence (the "
              "largest probability) and a probability for every label.</p>")


def scratch_background(p) -> str:
    return ("<p>This entry is <b>trained from scratch</b>. It is not fine-tuned. Fine-tuning starts from a model that "
            "has already been pre-trained on huge amounts of text and only teaches it the labels. Training from "
            "scratch starts from random numbers, and our train split has to teach the model everything: which words "
            "matter, which words are similar, and which label they point to.</p>"
            + SCRATCH_VS_FINETUNE
            + "<h3>Where it sits among the entries</h3>"
            + "<p>The collection covers most of the range between \"learn everything from our data\" and \"use what a "
              "large model already knows\". The three from-scratch entries fill the one row that was empty:</p>"
            + SPECTRUM
            + "<p>Training a large language model from scratch is rarely worth it for classification; it only pays "
              "off with millions of examples in an unusual domain. A <i>small</i> network from scratch is different: "
              "it trains in minutes, it teaches how neural classifiers work, and it is a fair reference for what "
              "pre-training adds.</p>")


def scratch_how(p) -> str:
    return (trained_diagram(p, finetuned=False) + "<p>All three from-scratch networks share the same front and back. Each word gets a learned vector of "
            "numbers (its <b>embedding</b>); at the start the vectors are random, and during training words that help "
            "predict the same label drift together. At the end, a linear <b>head</b> turns one text vector into one "
            "score per label. What differs is the <b>encoder</b> in the middle: how many word vectors become one text "
            "vector. The running example is a Banking77 message, \"my card has not arrived yet\", label "
            "<code>card_arrival</code>.</p>"
            + code(p["diagram"], "diagram") + p["how"]
            + "<h3>Strengths</h3>" + ul(p["strengths"]) + "<h3>Weaknesses</h3>" + ul(p["weaknesses"]))


def scratch_code(p) -> str:
    return (f"<p>The whole architecture of this entry is one PyTorch module in "
            f"<code>classifiers/{h(p['name'])}/model.py</code>. Its input is a batch of token ids (0 means padding) "
            f"and its output is one raw score (logit) per label. Everything else (tokenizing, training, calibration, "
            f"saving) is shared in <code>shared/textnn.py</code>.</p>"
            + code(p["code"], "python")
            + f"<p>The other files are in the repository: <code>shared/textnn.py</code>, and <code>classifier.py</code>, "
              f"<code>evaluate.py</code>, <code>checks.py</code> and <code>setup.sh</code> in "
              f"<code>classifiers/{h(p['name'])}/</code>.</p>")


def scratch_text(p) -> str:
    bigram = (" For this entry, the ids of adjacent word pairs are then appended: each pair is hashed into one of the "
              "bigram buckets, whose rows sit after the vocabulary in the same embedding table."
              if p["name"] == "scratch_fasttext" else "")
    return ("<p>A network only understands numbers, so every text goes through the same steps before training and "
            "before classification:</p><ol>"
            "<li><b>Tokenize.</b> Lowercase the text and split it into words and punctuation: \"My card hasn't "
            "arrived!\" becomes <code>my card hasn't arrived !</code>.</li>"
            "<li><b>Build a vocabulary, from the train split only.</b> Every word seen at least twice (once for "
            "Banking77) gets an integer id. Id 0 is padding and id 1 is \"unknown\", used for any word never seen in "
            "training. AG News gives roughly 60,000-70,000 words; Banking77 roughly 2,000-3,000.</li>"
            f"<li><b>Turn each text into ids</b> and cut it to a maximum length (100 tokens for AG News, 48 for "
            f"Banking77).{bigram}</li>"
            "<li><b>Pad.</b> Texts in one batch are padded with id 0 to the same length. The model masks the "
            "padding, so a text gets the same answer alone or in a batch.</li></ol>"
            "<p>The benchmark's own data preparation is unchanged: AG News text is the title and description joined as "
            "<code>{title}. {description}</code>, and Banking77 uses the message text. The test split is never used "
            "for any of the steps above.</p>")


def scratch_training(p) -> str:
    rows = [[h(k), h(a), h(b)] for k, a, b in p["settings"]]
    return (note("train", "Training", "<p>Training is required and will run automatically on the first benchmark "
                                      "run for each task. Later runs load the saved model in about a second.</p>")
            + "<ol><li><b>Split the train split</b>: a stratified 90% trains the network; the other 10% (seed 0) is "
              "the validation set. The test split is never touched.</li>"
              "<li><b>Start from random weights</b> (seed 0).</li>"
              "<li><b>Each step</b>: take a batch of texts, compute the logits, measure the cross-entropy loss "
              "against the true labels, backpropagate, and let AdamW update every weight. The learning rate rises "
              "for the first 10% of steps and then falls (one-cycle schedule); gradients are clipped at norm 1.</li>"
              "<li><b>After each epoch</b>, measure validation accuracy. Keep the best epoch; stop early when it has "
              "not improved for a few epochs (patience).</li>"
              "<li><b>Calibrate</b>: fit one temperature T on the validation set, so that softmax(logits / T) gives "
              "honest probabilities. It changes confidence, never the chosen label.</li>"
              f"<li><b>Save</b> the weights and a JSON file with the vocabulary, settings, temperature and the "
              f"per-epoch history in <code>classifiers/{h(p['name'])}/models/&lt;task&gt;/</code>.</li></ol>"
            + "<h3>Settings per task</h3>"
            + table(["Setting", "AG News", "Banking77"], rows)
            + "<p>Banking77 differs because its texts are short (about 12 words), its vocabulary is small, and each "
              "epoch is only about 280 steps, so it needs many more epochs and keeps rare words.</p>"
            + "<h3>Overfitting, and how to see it</h3>"
            + "<p>The saved history has training loss and validation loss per epoch. Training loss keeps falling; "
              "when validation loss starts rising while training loss falls, the network is memorising the training "
              "texts instead of learning the task. Early stopping keeps the epoch before that happens. Dropout and "
              "the vocabulary cut-off also help.</p>"
            + "<p>Never tune on <code>data/*_test.csv</code>. Every choice above is made on the validation part of "
              "the train split, and the test split is evaluated once, by the harness.</p>")


def scratch_results(p) -> str:
    ag, b77 = p["expected"]
    epoch, ag_full, b77_full, classify = p["time"]
    ag_size, b77_size = p["size"]
    m = p.get("measured") if is_implemented(p) else None
    if m:
        rows = lambda pairs: [[h(k), h(a), h(b)] for k, a, b in pairs]
        common_ag, common_b77 = m["accuracy"][0][1].split(" (")[0], m["accuracy"][0][2].split(" (")[0]
        return (f"<p>Measured on {h(m['date'])} on an Apple M5, with the recipe in "
                f"<a href=\"#training\">How it is trained</a>. The test samples are the benchmark's own (seed 42), "
                f"the same texts every other entry is scored on.</p>"
                + "<h3>Accuracy</h3>" + table(["", "AG News", "Banking77"], rows(m["accuracy"]))
                + f"<p>{h(m['notes'])}</p>"
                + "<h3>Compared with the other entries</h3>"
                + table(["", "AG News", "Banking77"],
                        [["<b>This entry, common tier (measured)</b>", f"<b>{common_ag}</b>", f"<b>{common_b77}</b>"],
                         ["This entry, as expected in the plan", ag, b77],
                         ["<code>tfidf_logreg</code>, common tier (measured)", "93.2%", "91.5%"],
                         ["<code>tfidf_logreg</code>, whole AG News test split", "92.7%", ""],
                         ["Best decision model on the common tier", "92.7% (<code>laya_typed</code>)",
                          "86.5% (<code>d1</code>)"],
                         ["Fine-tuned encoder (literature)", "~94.5-95.5%", "~93-94%"]])
                + "<p>With 1,000 test texts, the 95% confidence interval is about ±1.5 to 2 points, so differences "
                  "smaller than that are not reliable. The report's McNemar test says which ones are.</p>"
                + "<h3>Training, measured</h3>" + table(["", "AG News", "Banking77"], rows(m["training"]))
                + f"<p>{h(m['device'])} The whole epoch history (training loss, validation loss and accuracy per "
                  f"epoch) is in each model's JSON file.</p>"
                + "<h3>Speed, measured</h3>" + table(["", "AG News", "Banking77"], rows(m["speed"]))
                + "<p>For comparison, the decision models in this benchmark take from about 10 ms to over a second "
                  "per text. A model this small answers in well under a millisecond on the CPU.</p>"
                + scratch_sample(p))
    return ("<p>These are targets from the literature and from the baselines in this project, not measurements. "
            "They will be replaced by the measured results once the entry runs.</p>"
            + table(["", "AG News", "Banking77"],
                    [[f"<b>This entry (expected)</b>", f"<b>{ag}</b>", f"<b>{b77}</b>"],
                     ["<code>tfidf_logreg</code> (3-fold CV on train)", "92.2%", "89.9%"],
                     ["Best decision model on the common tier", "92.7% (<code>laya_typed</code>)",
                      "86.5% (<code>d1</code>)"],
                     ["Fine-tuned encoder (literature)", "~94.5-95.5%", "~93-94%"]])
            + "<p><b>AG News</b> has 120,000 articles, plenty to learn word meanings from scratch, and the topic is "
              "mostly given away by keywords. Expect to land near the TF-IDF baseline.</p>"
              "<p><b>Banking77</b> is the hard case: about 130 short messages per label is too little for a network "
              "to learn word meanings by itself. A word like \"refund\" may appear only a handful of times. Landing "
              "below <code>tfidf_logreg</code> there is a plausible, informative result, not a bug. It is the case "
              "for pre-training.</p>"
            + "<h3>Time on an Apple M5 (estimates)</h3>"
            + table(["AG News, per epoch", "AG News, full training", "Banking77, full training",
                     "Classifying 1,000 texts"], [[epoch, ag_full, b77_full, classify]])
            + "<p>Training is paid once per task and recipe; later runs load the saved model. For tiny models, "
              "especially the fastText-style one, the CPU is often as fast as the Apple GPU (PyTorch "
              "<code>mps</code>), because the cost of moving data to the GPU outweighs the arithmetic.</p>"
            + "<h3>Size</h3>"
            + table(["AG News", "Banking77"], [[ag_size, b77_size]])
            + "<p>The vocabulary dominates the size. For comparison, ModernBERT-base has 149M parameters.</p>")


def scratch_measured_comparison() -> str:
    """Measured accuracy and training time of the implemented from-scratch entries, side by side."""
    done = [x for x in SCRATCH if is_implemented(x) and x.get("measured")]
    if not done:
        return ""

    def pick(x, i, col):
        return h(x["measured"]["accuracy"][i][col].split(" (")[0])

    def train(x, col):
        return h(next(v for k, *v in x["measured"]["training"] if k == "Training time")[col - 1])

    rows = [[f'<a href="{x["name"]}.html">{h(x["short"])}</a>', pick(x, 0, 1), pick(x, 1, 1), train(x, 1),
             pick(x, 0, 2), pick(x, 1, 2), train(x, 2)] for x in done]
    rows.append(["<code>tfidf_logreg</code>, for reference", "93.2%", "92.7%", "about 1 min",
                 "91.5%", "", "seconds"])
    return ("<h3>Measured side by side</h3>"
            + table(["", "AG News, 1,000", "AG News, whole split", "AG News, training",
                     "Banking77, 1,000", "Banking77, whole split", "Banking77, training"], rows)
            + "<p>On AG News, word order buys little: the BiLSTM leads the bag of embeddings by about half a point, "
              "for about ten times the training time, and none of the three beats the TF-IDF baseline (the BiLSTM "
              "is in the report's bootstrap leading tier with it). On Banking77 the three land within 1.4 points of each "
              "other, 2 to 3.5 points below TF-IDF, and above every decision model in this benchmark. The remaining "
              "gap to a fine-tuned encoder (about 94-95% and 93-94%) is what pre-training buys.</p>")


def scratch_compare(p) -> str:
    links = ", ".join(f'<a href="{x["name"]}.html">{h(x["short"])}</a>' for x in SCRATCH if x is not p)
    return ("<p>The three from-scratch networks answer the same question in three ways: how do many word vectors become one "
            f"text vector? This page is the {h(p['short'])} one; the others are {links}.</p>"
            + THREE_COMPARED
            + scratch_measured_comparison()
            + "<p>A sensible order to learn them is fastText-style first (the whole loop end to end, in minutes), then "
              "the same model on Banking77 to see the small-data problem, then the CNN and the BiLSTM to see whether "
              "word order helps and by how much.</p>")


def scratch_transformer(p) -> str:
    return ("<p><b>No.</b> A Transformer is not required for classification, and trained from scratch it is often not "
            "the best choice.</p><ul>"
            "<li>Fine-tuned Transformers such as ModernBERT and DeBERTa beat everything on this page, but they win "
            "because of <b>pre-training</b> on trillions of words, not because of the architecture itself.</li>"
            "<li>A Transformer from scratch has to learn which words relate to which (attention) on top of what the "
            "words mean, so it needs more data. On 120,000 AG News articles a small one usually lands about where "
            "the CNN or BiLSTM does, around 91-92%, while being slower and harder to tune. On Banking77's 10,000 "
            "messages it would likely do worse than the simpler models.</li>"
            "<li>AG News is mostly a keyword problem. A bag of embeddings already gets about 92.5%, and most of the "
            "remaining errors are genuinely ambiguous articles, such as a business story about a technology "
            "company.</li></ul>"
            "<p>The practical rule: for a small model from scratch, use fastText-style or a CNN. For the best "
            "accuracy, do not train a Transformer from scratch; fine-tune a pre-trained one.</p>")


def scratch_reproduce(p) -> str:
    n = p["name"]
    tree = (f"classifiers/{n}/\n"
            f"├── model.py          the architecture ({p['class']})\n"
            f"├── classifier.py     per-task settings (CONFIGS) and the classifier; runnable demo\n"
            f"├── evaluate.py       benchmark adapter: train or load, then classify the test sample\n"
            f"├── checks.py         setup checks: overfit 64 texts, batch-size invariance, device vs CPU\n"
            f"├── setup.sh, requirements.txt\n"
            f"├── models/<task>/    model_<digest>.pt (weights) and .json (vocabulary, settings, T, epoch history)\n"
            f"└── results/          predictions_*.csv, run_*.json\n"
            f"\nshared/textnn.py      tokenizer, vocabulary, batching, training loop, temperature, model cache")
    return ("<ol class=\"steps\">"
            f"<li><b>Set up the environment.</b> Creates <code>.venv</code>, installs PyTorch and runs the setup checks "
            f"on both tasks.{code(f'./run setup {n}', 'shell')}</li>"
            f"<li><b>Train and evaluate.</b> The first run per task trains the network on the train split and saves it; "
            f"later runs load the saved model.{code(f'./run {n} --task ag_news --n 1000\n./run {n} --task banking77 --n 1000', 'shell')}</li>"
            "<li><b>Inspect the training.</b> Open <code>models/&lt;task&gt;/model_&lt;digest&gt;.json</code>: "
            "<code>history</code> has training loss, validation loss and validation accuracy per epoch, and "
            "<code>best_epoch</code> the epoch that was kept. Validation loss rising while training loss falls is "
            "overfitting; early stopping keeps the epoch before it.</li>"
            f"<li><b>Try it on one text.</b>{code(f'cd classifiers/{n}\nPYTHONPATH=../../shared .venv/bin/python classifier.py \"Stocks rallied after the central bank decision.\"', 'shell')}</li>"
            "<li><b>Experiment.</b> Change a recipe in <code>CONFIGS</code>, or pass options after <code>--</code>: "
            "<code>--epochs</code>, <code>--lr</code>, <code>--train-batch-size</code>, <code>--train-seed</code>, "
            "<code>--device</code>. A changed setting changes the model's fingerprint, so a new model is trained "
            "and kept beside the old one. Choose settings on validation accuracy, never on the test split.</li>"
            "</ol>"
            + "<h3>Files</h3>" + code(tree, "tree")
            + "<p><code>&lt;digest&gt;</code> fingerprints the app, every setting, the labels, the train data and the "
              "torch and shared-code versions.</p>")


SCRATCH_SECTIONS = [
    ("overview", "Overview", scratch_status),
    ("concepts", "Concepts: training from scratch vs fine-tuning", scratch_background),
    ("how", "How the model works", scratch_how),
    ("text", "From text to numbers", scratch_text),
    ("training", "Training procedure", scratch_training),
    ("reproduce", "Reproducing it, step by step", scratch_reproduce),
    ("code", "The architecture in code", scratch_code),
    ("results", "Results, time and size", scratch_results),
    ("compare", "Comparison with the other two", scratch_compare),
    ("transformer", "Design note: why not a small Transformer", scratch_transformer),
]


def scratch_number(p) -> int:
    """Registered entries keep their registry number; planned ones are numbered after every registered entry."""
    names = [a.name for a in APPS]
    if p["name"] in names:
        return names.index(p["name"]) + 1
    planned = [x["name"] for x in SCRATCH if not is_implemented(x)]
    return len(APPS) + planned.index(p["name"]) + 1


def scratch_page(p, index: int) -> str:
    number = scratch_number(p)
    state = "implemented" if is_implemented(p) else "planned"
    toc ="".join(f'<li><a href="#{sid}"><span>{i:02d}</span>{h(name)}</a></li>'
                  for i, (sid, name, _) in enumerate(SCRATCH_SECTIONS, 1))
    body = "".join(f'<section id="{sid}"><p class="secnum">{i:02d}</p><h2>{h(name)}</h2>{fn(p)}</section>'
                   for i, (sid, name, fn) in enumerate(SCRATCH_SECTIONS, 1))
    kind = "Supervised, trained from scratch" + (" (planned)" if state == "planned" else "")
    meta = [("Registry name", f"<code>{h(p['name'])}</code>"), ("Type", kind),
            ("Runs on", h(p["runs_on"])), ("Label limit", TRAINED_LIMIT),
            ("Confidence", CONF_TASK_BADGE)]
    meta_html = "".join(f'<div><span class="num">{i:03d}</span><span class="k">{k}</span><span class="v">{v}</span></div>'
                        for i, (k, v) in enumerate(meta, 1))
    prev_p = SCRATCH[index - 1] if index > 0 else None
    next_p = SCRATCH[index + 1] if index + 1 < len(SCRATCH) else None
    pager = '<nav class="pager wrap" aria-label="Previous and next model">'
    pager += (f'<a href="{h(prev_p["name"])}.html"><small>Previous</small>{h(prev_p["title"])}</a>' if prev_p
              else '<a href="index.html#g-scratch"><small>Back</small>All models</a>')
    pager += (f'<a class="next" href="{h(next_p["name"])}.html"><small>Next</small>{h(next_p["title"])}</a>'
              if next_p else '<a class="next" href="index.html#g-scratch"><small>Back</small>All models</a>')
    pager += "</nav>"
    summary = p["summary"]
    if state == "implemented":  # the summaries were written for the planned state
        summary = summary.replace(" Planned, not implemented yet.", "")
    return (head(f"{p['name']} model guide", f"Guide to the {state} {p['title']}: how it works, its code, training, "
                                             f"results and time in the Classification benchmark.")
            + topbar()
            + f'<main><div class="pagehead"><div class="wrap"><p class="crumb"><a href="index.html">Models</a>'
              f'<span>/</span><span>{number:03d}</span><span>/</span><span>{h(p["name"])}</span>'
              f'{"<span>/</span><span>planned</span>" if state == "planned" else ""}</p>'
              f'<h1>{h(p["title"])}</h1><p class="lede">{h(summary)}</p></div></div>'
            + f'<div class="band"><div class="wrap meta">{meta_html}</div></div>'
            + f'<div class="wrap doc"><aside class="toc"><p class="toc-title">Contents</p><ol>{toc}</ol></aside>'
              f'<article>{body}</article></div>{pager}</main>'
            + footer())


# ---------------------------------------------------------------------------------------------------------------
# Pre-trained encoders, fine-tuned: their own page layout and their own index group
# ---------------------------------------------------------------------------------------------------------------
# Each entry is its own folder (classifiers/<name>/) sharing
# shared/finetune.py. Fill in "measured" after the runs; record a sample with models/capture_samples.py <name>.
FINETUNE_GROUP = ("finetune", "Fine-tuned encoders",
                  "Released pre-trained encoders, fine-tuned on the task's train split.")

FT_LOOP_CODE = '''for epoch in range(1, cfg.max_epochs + 1):
    model.train()
    for _, ids, mask, y in batches(fit_x, fit_y, cfg.batch_size, pad_id, rng):   # length-bucketed, shuffled
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=cfg.bf16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        loss = F.cross_entropy(logits.float(), y.to(device))   # the loss itself in float32
        opt.zero_grad(set_to_none=True)
        loss.backward()                                         # gradients reach every encoder weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                                              # AdamW, peak lr 2e-5 to 5e-5
        sched.step()                                            # linear warm-up (6%), then linear decay
    val_acc = accuracy(logits_for(model, val_x, ...), val_y)    # keep the best epoch, stop early otherwise'''

FINETUNE = [
    {
        "name": "finetune_modernbert", "title": "ModernBERT-base, fine-tuned", "short": "ModernBERT-base",
        "runs_on": "Local, PyTorch MPS", "env": "FINETUNE_MODERNBERT_DEVICE",
        "repo": "answerdotai/ModernBERT-base", "rev": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "url": "https://huggingface.co/answerdotai/ModernBERT-base",
        "summary": "Answer.AI's ModernBERT-base, a 149-million-parameter encoder pre-trained on about 2 trillion tokens, "
                   "fine-tuned on the task's train split. A new linear head is added and every weight "
                   "is trained gently, so the model keeps what it learned in pre-training and learns our labels.",
        "backbone": [("Model", "ModernBERT-base (Answer.AI, LightOn and partners, December 2024)"),
                     ("Size", "149M parameters (22 layers, hidden size 768), about 600 MB in float32"),
                     ("Pre-training", "about 2 trillion tokens of English text and code; context of 8,192 tokens"),
                     ("Tokenizer", "BPE, 50,368 sub-word tokens: every word can be spelled from pieces, so there is no "
                                   "unknown word"),
                     ("Why it", "Modern design (rotary positions, alternating local and global attention, GeGLU), "
                                "fast, and already pinned and used on MPS in this project (modernbert_jev, "
                                "nli_modernbert)")],
        "recipe": [("Max tokens", "160", "64"), ("Batch size", "32", "32"), ("Peak learning rate", "3e-5", "5e-5"),
                   ("Max epochs (patience)", "3 (1)", "15 (3)")],
        "lengths": "ModernBERT tokens per text, measured on 3,000 train texts: AG News mean 55 (99th percentile 126, "
                   "max 299); Banking77 mean 16 (99th percentile 51, max 93).",
        "speed_probe": "Measured before implementing (fixed 128 and 64 tokens, batch 32): 24 and 47 texts/s in float32, "
                       "35 and 70 texts/s with bfloat16 autocast, so training uses bfloat16 (about 1.5x faster).",
        "checks": "Setup checks (both tasks): 149,664,077 parameters with the AG News head; fits 32 texts in 15 short "
                  "epochs (1.000); batch 1 vs 16 differ by at most 1.7e-05 in the logits; MPS vs CPU by at most "
                  "4.3e-05.",
        "measured": {
            "date": "2026-11-18",
            "accuracy": [("Common tier, 1,000 texts", "95.0% (macro F1 0.950)", "93.5% (macro F1 0.935)"),
                         ("Whole test split", "94.8% (7,205 / 7,600)", "93.6% (2,883 / 3,080)"),
                         ("Validation accuracy at the best epoch", "94.7%", "93.7%")],
            "training": [("Fit + validation texts", "108,000 + 12,000", "9,000 + 1,003"),
                         ("Epochs run (best)", "3 (3): the recipe's limit", "14 (11): early stop"),
                         ("Seconds per epoch", "about 1,440 (79 texts/s)", "62 (about 150 texts/s)"),
                         ("Fine-tuning time", "73 min", "14.6 min"),
                         ("Pilot on 40,000 texts", "30 min; validation 93.8%", ""),
                         ("Temperature T", "2.297", "1.571"),
                         ("Saved model", "571 MB", "571 MB")],
            "speed": [("Batch of 64", "about 91 texts/s", "about 247 texts/s"),
                      ("One text at a time", "10.8 ms per text", "12.1 ms per text"),
                      ("Loading the saved model", "0.9 s", "0.8 s")],
            "notes": "ModernBERT-base is the most accurate model in the benchmark on AG News's common tier: 95.0%, "
                     "1.8 points above tfidf_logreg (93.2%) and 2.7 above the best from-scratch network, and inside the "
                     "94.5-95.5% published for fine-tuned base encoders. On the whole AG News split it scores 94.8% "
                     "against tfidf_logreg's 92.7%. Using all 108,000 texts instead of a 40,000-text pilot bought about "
                     "one point (validation 93.8% to 94.7%) for 2.4 times the time; validation loss rose in epoch 3, so "
                     "a fourth epoch would not help. On Banking77 it reaches 93.5% (93.6% on the whole split), 2 points "
                     "above tfidf_logreg, predicting all 77 labels.",
        },
    },
    {
        "name": "finetune_deberta", "title": "DeBERTa-v3-base, fine-tuned", "short": "DeBERTa-v3-base",
        "runs_on": "Local, PyTorch MPS", "env": "FINETUNE_DEBERTA_DEVICE",
        "repo": "microsoft/deberta-v3-base", "rev": "de19fe7db5162df5f3d8f0b41321c0267288fd74",
        "url": "https://huggingface.co/microsoft/deberta-v3-base",
        "summary": "Microsoft's DeBERTa-v3-base, a 184-million-parameter encoder (86M in the layers, 98M in its large "
                   "vocabulary), fine-tuned on the task's train split. A new linear head is added and "
                   "every weight is trained gently. It is a long-standing strong choice for classification.",
        "backbone": [("Model", "DeBERTa-v3-base (Microsoft, 2021)"),
                     ("Size", "184M parameters: 12 layers of hidden size 768 (86M) and a 128,000-token embedding table "
                              "(98M); about 740 MB in float32"),
                     ("Pre-training", "ELECTRA-style replaced-token detection on 160 GB of text"),
                     ("Tokenizer", "SentencePiece, 128,000 sub-word tokens (spm.model): no unknown words"),
                     ("Revision", "refs/pr/14, Hugging Face's official SFconvertbot conversion of the original weights "
                                  "to safetensors. The main revision ships only a Python pickle (pytorch_model.bin), "
                                  "which this entry never downloads"),
                     ("Why it", "Disentangled attention (content and position kept apart) makes it one of the most "
                                "accurate base-size encoders for classification; a second, independent backbone to "
                                "compare with ModernBERT")],
        "recipe": [("Max tokens", "160", "64"), ("Batch size", "32", "32"), ("Peak learning rate", "2e-5", "4e-5"),
                   ("Max epochs (patience)", "3 (1)", "15 (3)")],
        "lengths": "DeBERTa's tokenizer splits text into a similar number of pieces as ModernBERT's, so the same "
                   "limits (160 and 64 tokens) keep nearly every text whole.",
        "speed_probe": "bfloat16 training was checked against float32 on Banking77 before the real runs: both fit the "
                       "32-text check set perfectly after 15 epochs, so DeBERTa also trains with bfloat16 autocast.",
        "checks": "Setup checks (both tasks): 184,425,220 parameters with the AG News head; fits 32 texts in 15 short "
                  "epochs (1.000; 8 epochs were too few for 32 Banking77 labels with one text each); batch 1 vs 16 "
                  "differ by at most 1.7e-06 in the logits; MPS vs CPU by at most 2.2e-06.",
        "measured": {
            "date": "2026-11-18",
            "accuracy": [("Common tier, 1,000 texts", "94.4% (macro F1 0.944)", "94.2% (macro F1 0.942)"),
                         ("Whole test split", "94.9% (7,211 / 7,600)", "93.2% (2,870 / 3,080)"),
                         ("Validation accuracy at the best epoch", "94.6%", "93.9%")],
            "training": [("Fit + validation texts", "108,000 + 12,000", "9,000 + 1,003"),
                         ("Epochs run (best)", "3 (3): the recipe's limit", "15 (12): early stop"),
                         ("Seconds per epoch", "about 1,810 (about 60 texts/s)", "90 (about 100 texts/s)"),
                         ("Fine-tuning time", "92 min", "22.5 min"),
                         ("Temperature T", "1.410", "1.599"),
                         ("Saved model", "704 MB", "704 MB")],
            "speed": [("Batch of 64", "about 52 texts/s", "about 199 texts/s"),
                      ("One text at a time", "22.1 ms per text", "26.9 ms per text"),
                      ("Loading the saved model", "1.0 s", "1.0 s")],
            "notes": "DeBERTa-v3-base is the most accurate model in the benchmark on Banking77's common tier: 94.2%, "
                     "2.7 points above tfidf_logreg (91.5%), with all 77 labels predicted; on the whole Banking77 split "
                     "it scores 93.2%. On AG News it scores 94.4% on the common tier and 94.9% on the whole split, the "
                     "best whole-split result in the benchmark, 2.2 points above tfidf_logreg (92.7%). It and "
                     "ModernBERT-base trade places between the 1,000-text sample and the whole split on both tasks, "
                     "so they are effectively tied; DeBERTa costs about 1.3-1.5 times the fine-tuning time and twice "
                     "the time per text. Its validation loss was still falling at epoch 3 on AG News (unlike "
                     "ModernBERT's), so a fourth epoch might add a little.",
        },
    },
]


def ft_implemented(p) -> bool:
    return p["name"] in {a.name for a in APPS}


def ft_status(p) -> str:
    n = h(p["name"])
    m = p.get("measured") if ft_implemented(p) else None
    lead = "" if ft_implemented(p) else note("warn", "Planned", "<p>Designed but not yet implemented.</p>")
    rows = [("Registry name", f"<code>{n}</code>"), ("Folder", f"<code>classifiers/{n}/</code>"),
            ("Pre-trained backbone", f'<a href="{h(p["url"])}"><code>{h(p["repo"])}</code></a>, revision '
                                     f'<code>{h(p["rev"][:7])}</code>'),
            ("What is trained", "Every weight of the encoder, at a small learning rate, plus a new linear head"),
            ("Training data", "The task's train split: a stratified 90% to fit, 10% for validation"),
            ("Fine-tuned models", f"<code>classifiers/{n}/models/&lt;task&gt;/&lt;digest&gt;/</code>"),
            ("Shared code", "<code>shared/finetune.py</code>, with the splits and calibration of "
                            "<code>shared/textnn.py</code>"),
            ("Device", f"MPS (<code>{h(p['env'])}</code>); bfloat16 training, float32 classification"),
            ("Confidence", "max p after temperature scaling on the validation split")]
    if m:
        rows.append(("Accuracy, 1,000 test texts", f"AG News {h(m['accuracy'][0][1])}; Banking77 "
                                                   f"{h(m['accuracy'][0][2])}"))
    return lead + kv_table(rows) + (f"<p>The code is in <code>classifiers/{n}/</code> and <code>shared/finetune.py</code>; "
                                     "the measurements are in the reports linked at the top of this page.</p>")


def ft_background(p) -> str:
    return ("<p>This entry is <b>fine-tuned</b>, not trained from scratch. It starts from an encoder that has already "
            "learned grammar, word meanings and a lot of world knowledge from huge amounts of text. Our train split "
            "only has to teach it the labels. Pre-training is why it can learn the labels from a modest train split, and "
            "on both tasks here it scores above the networks trained from scratch (see the comparison below).</p>"
            + SCRATCH_VS_FINETUNE.replace("(this entry)", "")
            + "<h3>Where it sits among the entries</h3>" + SPECTRUM)


def ft_how(p) -> str:
    return (trained_diagram(p, finetuned=True) + code("my card has not arrived yet\n"
                 "        |  the model's own tokenizer (sub-word pieces, never \"unknown\")\n"
                 "[CLS] my card has not arrived yet [SEP]\n"
                 "        |  pre-trained encoder: every token attends to every other token,\n"
                 "        |  layer after layer (this is what fine-tuning adjusts, gently)\n"
                 "  one vector for the whole text\n"
                 "        |  new linear head (random at the start, one output per label)\n"
                 "  77 scores  ->  softmax(scores / T)  ->  card_arrival", "diagram")
            + "<ol><li><b>Load the pre-trained encoder</b> at its pinned revision and put a new, randomly initialised "
              "linear layer on top, with one output per label.</li>"
              "<li><b>Tokenize</b> each text with the model's own tokenizer and feed it through the encoder; the head "
              "reads the encoder's text vector.</li>"
              "<li><b>Cross-entropy loss</b> against the true label; backpropagate through the head <i>and</i> the "
              "whole encoder.</li>"
              "<li><b>AdamW with a tiny learning rate</b> (2e-5 to 5e-5, about a hundred times smaller than for the "
              "from-scratch networks), rising for the first 6% of steps and then falling to zero. Large steps would "
              "erase what pre-training taught: catastrophic forgetting.</li>"
              "<li><b>After each epoch</b>, measure validation accuracy and keep the best epoch; then fit one "
              "temperature T so the probabilities are honest.</li></ol>"
              "<p>Steps 3 to 5 are the same loop the from-scratch networks use. Only the model, the tokenizer and the "
              "learning rate change.</p>")


def ft_backbone(p) -> str:
    return kv_table([(h(k), h(v)) for k, v in p["backbone"]]) + f"<p>{h(p['checks'])}</p>"


def ft_code(p) -> str:
    return ("<p>The heart of <code>shared/finetune.py</code>, shared by both fine-tuned entries:</p>"
            + code(FT_LOOP_CODE, "python")
            + "<p>Two details make it work well on an Apple GPU. Batches are <b>length-bucketed</b>: texts of similar "
              "length are trained together, so there is little padding, while the order stays random. Training runs "
              "under <b>bfloat16 autocast</b>, about 1.5 times faster than float32; classification runs in float32, so "
              "a text gets the same answer alone or in a batch and on the CPU.</p>"
            + f"<p>Every file's full code is in <code>shared/finetune.py</code> and "
              f"<code>classifiers/{h(p['name'])}/</code>.</p>")


def ft_text(p) -> str:
    return ("<p>Unlike the from-scratch networks, which build their own vocabulary from the train split, a fine-tuned "
            "model must use the tokenizer it was pre-trained with. It splits rare words into known pieces, so there "
            "is no \"unknown word\": a typo or a new product name still reaches the model as something it can read.</p>"
            f"<p>{h(p['lengths'])}</p>"
            "<p>The benchmark's data preparation is unchanged: AG News text is <code>{title}. {description}</code>, "
            "Banking77 is the message. The test split is never used for training or for any choice.</p>")


def ft_training(p) -> str:
    rows = [[h(k), h(a), h(b)] for k, a, b in p["recipe"]] + [
        ["Warm-up", "6% of steps, then linear decay", "same"], ["Weight decay", "0.01 (not on biases and norms)", "same"],
        ["Training precision", "bfloat16 autocast; loss in float32", "same"],
        ["Fit / validation", "90% / 10% of the train split (stratified, seed 0)", "same"]]
    return (note("train", "Training", "<p>Fine-tuning runs automatically on the first benchmark run per task and is "
                                      "saved; later runs load it. On an M5 it takes minutes for Banking77 and over an "
                                      "hour for AG News, so run it plugged in.</p>")
            + table(["Setting", "AG News", "Banking77"], rows)
            + f"<p>{h(p['speed_probe'])}</p>"
            + "<p>Options for experiments (each trains and keeps a separate model and results folder): "
              "<code>--train-limit 40000</code> (a quick pilot on part of the data), <code>--train-seed 1</code> "
              "(another seed, to see the spread), <code>--epochs</code>, <code>--lr</code> and <code>--fp32</code>.</p>")


def ft_results(p) -> str:
    m = p.get("measured")
    if not m:
        return ("<p>Measured results will appear here once the runs finish. Expected from published results for "
                "fine-tuned base-size encoders: about 94.5-95.5% on AG News and 92.5-94% on Banking77, against "
                "<code>tfidf_logreg</code>'s 93.2% and 91.5% on the same 1,000 texts.</p>")
    rows = lambda pairs: [[h(k), h(a), h(b)] for k, a, b in pairs]
    return (f"<p>Measured on {h(m['date'])} on an Apple M5, on the benchmark's own test samples (seed 42).</p>"
            + "<h3>Accuracy</h3>" + table(["", "AG News", "Banking77"], rows(m["accuracy"]))
            + f"<p>{h(m['notes'])}</p>"
            + "<h3>Fine-tuning, measured</h3>" + table(["", "AG News", "Banking77"], rows(m["training"]))
            + "<h3>Speed, measured</h3>" + table(["", "AG News", "Banking77"], rows(m["speed"]))
            + scratch_sample(p))


def ft_compare(p) -> str:
    rows = []
    for x in SCRATCH + FINETUNE:
        m = x.get("measured")
        if m and (is_implemented(x) if x in SCRATCH else ft_implemented(x)):
            kind = "from scratch" if x in SCRATCH else "fine-tuned"
            rows.append([f'<a href="{x["name"]}.html">{h(x["short"])}</a> ({kind})',
                         h(m["accuracy"][0][1].split(" (")[0]), h(m["accuracy"][0][2].split(" (")[0])])
    rows.append(["<code>tfidf_logreg</code>", "93.2%", "91.5%"])
    rows.append(["Best decision model", "92.7% (<code>laya_typed</code>)", "86.5% (<code>d1</code>)"])
    return ("<p>The entries trained from scratch and the fine-tuned entries, on the same 1,000 test texts "
            "(seed 42):</p>"
            + table(["", "AG News", "Banking77"], rows)
            + "<p>The from-scratch networks learn word meanings from our data alone and land at or below TF-IDF. The "
              "fine-tuned encoders bring their pre-training; the difference between the two groups is what "
              "pre-training buys.</p>"
            + "<h3>Are the gaps real? Exact McNemar tests on the whole test splits</h3>"
            + table(["Pair (whole split)", "AG News (7,600)", "Banking77 (3,080)"],
                    [["ModernBERT-base vs <code>tfidf_logreg</code>", "284 vs 122 texts only one gets right; "
                      "p = 6e-16", "130 vs 56; p = 6e-08"],
                     ["DeBERTa-v3-base vs <code>tfidf_logreg</code>", "281 vs 113; p = 1e-17", "136 vs 75; p = 3e-05"],
                     ["ModernBERT-base vs DeBERTa-v3-base", "118 vs 124; p = 0.75", "79 vs 66; p = 0.32"],
                     ["BiLSTM (from scratch) vs <code>tfidf_logreg</code>", "138 vs 216; p = 4e-05 (worse)",
                      "82 vs 153; p = 4e-06 (worse)"]])
            + "<p>Both fine-tuned encoders beat TF-IDF beyond any doubt on both tasks, and they are tied with each "
              "other. The from-scratch networks, tied with TF-IDF on the 1,000-text sample, are significantly worse "
              "than it on the whole splits. On 1,000 texts alone, only DeBERTa on AG News is not yet significant "
              "against TF-IDF (p = 0.13): a 1,000-text sample needs a gap of about 2 points.</p>"
            + "<h3>Three seeds on Banking77 (whole split)</h3>"
            + table(["", "Seed 0", "Seed 1", "Seed 2", "Mean ± sd"],
                    [["ModernBERT-base", "93.60%", "93.38%", "93.51%", "<b>93.50 ± 0.11</b>"],
                     ["DeBERTa-v3-base", "93.18%", "93.34%", "93.21%", "<b>93.25 ± 0.09</b>"],
                     ["<code>tfidf_logreg</code>", "91.20%", "", "", "deterministic"]])
            + "<p>Fine-tuning again with a different seed (a different validation split, head initialisation and "
              "batch order) moves accuracy by about 0.1 point, so the results are stable, and both models stay about "
              "2.3 points above TF-IDF on every seed.</p>"
            + "<h3>What the remaining errors are</h3>"
            + "<p>Reading the 30 most confident errors per task: on AG News about a third are label noise (\"Bryant "
              "Makes First Appearance at Trial\" labelled Science/Technology, \"Argentina Beats U.S. Men's "
              "Basketball Team\" labelled World) and most of the rest are business stories about technology companies, "
              "which fit either label; business and sci_tech account for 207 of ModernBERT's 395 errors. On Banking77 "
              "about a third are label noise (\"whats your exchange rate\" labelled exchange_charge) and most of the "
              "rest are overlapping intents such as card_arrival and card_delivery_estimate. Only a handful are clear "
              "model mistakes, and the two fine-tuned models share about two thirds of their errors. At this level, "
              "better labels or label descriptions would help more than a bigger model.</p>")


def ft_reproduce(p) -> str:
    n = p["name"]
    tree = (f"classifiers/{n}/\n"
            f"├── classifier.py     backbone pin (REPO, REVISION), per-task recipes (CONFIGS), FineTunedEncoder; demo\n"
            f"├── evaluate.py       benchmark adapter: fine-tune or load, then classify the test sample\n"
            f"├── checks.py         setup checks: backbone size, fit 32 texts, batch-size invariance, MPS vs CPU\n"
            f"├── setup.sh, requirements.txt\n"
            f"├── models/base/      the pinned pre-trained backbone (safetensors)\n"
            f"├── models/<task>/    <digest>/model.safetensors + meta.json (epoch history, T, timings)\n"
            f"└── results/          predictions_*.csv, run_*.json; seed<N>/ and train<N>/ for experiments\n"
            f"\nshared/finetune.py    batches, training loop, model cache, classifier (uses shared/textnn.py)")
    setup = code(f"./run setup {n}", "shell")
    b77 = code(f"./run {n} --task banking77 --n 1000", "shell")
    pilot = code(f"cd classifiers/{n}\nPYTHONPATH=../../shared .venv/bin/python evaluate.py --task ag_news "
                 f"--train-limit 40000", "shell")
    ag = code(f"./run {n} --task ag_news --n 1000", "shell")
    seed = code(f"cd classifiers/{n}\nPYTHONPATH=../../shared .venv/bin/python evaluate.py --task banking77 --n 0 "
                f"--train-seed 1", "shell")
    return ('<ol class="steps">'
            f"<li><b>Set up the environment.</b> Creates <code>.venv</code>, installs PyTorch and transformers, "
            f"downloads the pinned backbone and runs the setup checks.{setup}</li>"
            f"<li><b>Fine-tune and evaluate Banking77 first.</b> It takes minutes and confirms the whole pipeline. "
            f"The fine-tuned model is saved, and later runs load it.{b77}</li>"
            f"<li><b>Optionally, run a pilot on part of AG News</b> to check the learning curve and the time per epoch "
            f"before the full run. It is saved as a separate model with its own results folder.{pilot}</li>"
            f"<li><b>Fine-tune AG News.</b> This takes over an hour on an Apple M5; use mains power.{ag}</li>"
            "<li><b>Inspect the training.</b> <code>models/&lt;task&gt;/&lt;digest&gt;/meta.json</code> holds the "
            "per-epoch training loss, validation loss and validation accuracy, the epoch kept and the temperature. "
            "Validation loss rising while training loss falls is the sign to stop.</li>"
            f"<li><b>Check stability.</b> Fine-tune again with another seed; each seed is saved separately, and a "
            f"spread of about 0.1 point is normal.{seed}</li>"
            "</ol>"
            + "<h3>Files</h3>" + code(tree, "tree")
            + "<p><code>&lt;digest&gt;</code> fingerprints the backbone revision, every setting, the labels, the train "
              "data and the torch, transformers and shared-code versions.</p>")


FT_SECTIONS = [
    ("overview", "Overview", ft_status),
    ("concepts", "Concepts: fine-tuning vs training from scratch", ft_background),
    ("how", "How fine-tuning works", ft_how),
    ("backbone", "The pre-trained backbone", ft_backbone),
    ("text", "From text to tokens", ft_text),
    ("training", "Training procedure", ft_training),
    ("reproduce", "Reproducing it, step by step", ft_reproduce),
    ("code", "The training loop in code", ft_code),
    ("results", "Results, time and size", ft_results),
    ("compare", "Comparison and error analysis", ft_compare),
]


def ft_number(p) -> int:
    names = [a.name for a in APPS]
    return names.index(p["name"]) + 1 if p["name"] in names else len(APPS) + FINETUNE.index(p) + 1


def finetune_page(p, index: int) -> str:
    toc = "".join(f'<li><a href="#{sid}"><span>{i:02d}</span>{h(name)}</a></li>'
                  for i, (sid, name, _) in enumerate(FT_SECTIONS, 1))
    body = "".join(f'<section id="{sid}"><p class="secnum">{i:02d}</p><h2>{h(name)}</h2>{fn(p)}</section>'
                   for i, (sid, name, fn) in enumerate(FT_SECTIONS, 1))
    meta = [("Registry name", f"<code>{h(p['name'])}</code>"), ("Type", "Supervised, fine-tuned"),
            ("Runs on", h(p["runs_on"])), ("Label limit", TRAINED_LIMIT), ("Confidence", CONF_TASK_BADGE)]
    meta_html = "".join(f'<div><span class="num">{i:03d}</span><span class="k">{k}</span><span class="v">{v}</span></div>'
                        for i, (k, v) in enumerate(meta, 1))
    prev_p = FINETUNE[index - 1] if index > 0 else None
    next_p = FINETUNE[index + 1] if index + 1 < len(FINETUNE) else None
    pager = '<nav class="pager wrap" aria-label="Previous and next model">'
    pager += (f'<a href="{h(prev_p["name"])}.html"><small>Previous</small>{h(prev_p["title"])}</a>' if prev_p
              else '<a href="index.html#g-finetune"><small>Back</small>All models</a>')
    pager += (f'<a class="next" href="{h(next_p["name"])}.html"><small>Next</small>{h(next_p["title"])}</a>'
              if next_p else '<a class="next" href="index.html#g-finetune"><small>Back</small>All models</a>')
    pager += "</nav>"
    state = "implemented" if ft_implemented(p) else "planned"
    return (head(f"{p['name']} model guide", f"Guide to {p['title']}: how fine-tuning works, the backbone, training, "
                                             f"results and time in the Classification benchmark.")
            + topbar()
            + f'<main><div class="pagehead"><div class="wrap"><p class="crumb"><a href="index.html">Models</a>'
              f'<span>/</span><span>{ft_number(p):03d}</span><span>/</span><span>{h(p["name"])}</span>'
              f'{"<span>/</span><span>planned</span>" if state == "planned" else ""}</p>'
              f'<h1>{h(p["title"])}</h1><p class="lede">{h(p["summary"])}</p></div></div>'
            + f'<div class="band"><div class="wrap meta">{meta_html}</div></div>'
            + f'<div class="wrap doc"><aside class="toc"><p class="toc-title">Contents</p><ol>{toc}</ol></aside>'
              f'<article>{body}</article></div>{pager}</main>'
            + footer())


# ---------------------------------------------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------------------------------------------
REPO_URL = "https://github.com/humpreycogay/Classification"  # the public Git repository; shown in the clone command of "Get started" (placeholder while empty)

GET_STARTED = (
    "<p>Everything on this site can be reproduced from the repository: the setup scripts download each model, "
    "the benchmark datasets are downloaded on first use and checked against a pinned checksum, and the runner "
    "writes the same reports you see here from your own results.</p>"
    "<p><b>Requirements.</b> macOS on Apple Silicon is the reference machine (MPS / MLX GPU); many PyTorch entries "
    "also run on CUDA or CPU. Python 3.12 or newer (SemIf, Nimble and the OpenJev server need 3.12 specifically). "
    "Some entries also need <a href=\"https://ollama.com/download\">Ollama</a>, an API key or a remote GPU server; "
    "each model page lists what it needs.</p>"
    + code(f"git clone {REPO_URL or '<repository URL>'} Classification\n"
           "cd Classification\n"
           "runner/setup.sh              # the runner's own environment (once)\n"
           "cp .env.example .env         # model choices, devices and API keys; the defaults work for local entries\n"
           "./run list                   # every classifier: ready to run, or what it still needs\n"
           "./run setup tfidf_svm        # one entry: virtualenv, dependencies, model download\n"
           "./run tfidf_svm              # 200 AG News articles; --task banking77, --n 1000, --dry-run\n"
           "./run report                 # rebuild reports/ag_news.html from your results", "shell")
    + "<p>To classify a single text with an entry that is set up:</p>"
    + code("cd classifiers/tfidf_svm\n"
           "PYTHONPATH=../../shared .venv/bin/python classifier.py \"Stocks rallied after the central bank decision.\"",
           "shell")
    + "<p><code>./run all</code> runs every entry that is set up and builds the comparison report. "
      "<code>tfidf_svm</code> is a good first entry: it needs no GPU and runs in under a minute.</p>")

INTRO = [
    ("start", "Get started", GET_STARTED),
    ("what", "What a classifier does",
     "<p>A text classifier reads a piece of text and assigns it one label from a fixed list. The list is decided in "
     "advance. In this project there are two lists: four news topics for AG News (World, Sports, Business, "
     "Science/Technology) and 77 customer intents for Banking77, such as <code>card_arrival</code> or "
     "<code>lost_or_stolen_card</code>.</p>"
     "<p>Every classifier here returns the same three things for each text: the chosen label, a confidence value "
     "between 0 and 1, and a score for every label in the list (a probability estimate for most entries; see "
     "<a href=\"#confidence\">What the confidence numbers mean</a>). Because the output is the same, the models "
     "can be compared on the same test texts with the same measures.</p>"),
    ("system-one", "What a System One model is",
     "<p>Many of the models in this guide describe themselves as System One models. The name comes from psychology, "
     "where System 1 refers to fast, automatic judgments and System 2 to slow, step-by-step reasoning. In this field "
     "it names a model built to make one decision quickly. It receives a situation (the text, called the state) and a "
     "closed set of options, and it returns one probability per option, usually from a single forward pass.</p>"
     "<p>A System One model does not write an explanation, and it does not generate an answer word by word as a chat "
     "model does. This makes it faster and cheaper per decision, and its output can always be parsed: it is one of "
     "the given options, never free text. TypeSafe's Jev API defined the request format, called "
     "<code>/v1/systemone</code>, in which a choice question carries an instruction and a set of criteria (the "
     "options with descriptions). OpenJev, djev and Winnow serve the same format, and models such as d1, Decider, "
     "Nimble, Kev and CLM were trained for this kind of decision.</p>"
     "<p>A chat model can be used in the same way, as the <code>llmqwen</code> entry shows, by listing the options with "
     "letters and reading the probability of each letter. The difference is that a System One model is trained "
     "for the decision, while a chat model is prompted into it.</p>"
     "<h3>Other ways to classify text</h3>"
     "<p>System One models are one approach among several. Text classification is an older field than these models, "
     "and the other approaches remain useful:</p><ul>"
     "<li><b>Classic machine learning.</b> Word and character features weighted with TF-IDF, passed to logistic "
     "regression, a linear SVM or Naive Bayes. These train in seconds on a CPU and need labelled examples.</li>"
     "<li><b>Embeddings with a small classifier.</b> An embedding model turns each text into a vector, and a simple "
     "classifier is trained on those vectors.</li>"
     "<li><b>Fine-tuned custom models.</b> An encoder such as BERT or ModernBERT, trained on your own labelled data "
     "for your own labels. When enough labelled data exists, this is often the most accurate and cheapest option to "
     "run.</li>"
     "<li><b>Other zero-shot methods.</b> Natural language inference models and prompted chat models, which need no "
     "labelled data.</li></ul>"
     "<p>This project focuses on Jev-style decision models because comparing them is its purpose. The other "
     "approaches are included as reference points: the TF-IDF and embedding baselines are trained on each task's "
     "train split, and on AG News they are among the most accurate entries. If your task has labelled data, compare "
     "against them before choosing a System One model.</p>"),
    ("pipeline", "How the benchmark runs a model",
     "<p>Every entry goes through the same steps, so their results can be compared on the same texts. The diagram "
     "separates the paths that can change a model's decision from the path that only calibrates its confidence. The "
     "score reports link here from their type notes.</p>" + pipeline_diagram()),
    ("shared", "What every entry shares", SHARED_DATA),
    ("kinds", "How the models differ",
     "<p><b>Zero-shot</b> models decide without fitting their weights or label ranking on task examples. They read "
     "the label descriptions and decide. Changing the labels needs no training, so the quality of the descriptions "
     "matters. Some use labeled training examples solely to fit a confidence temperature; their accuracy remains "
     "zero-shot, but their confidence and calibration metrics are task-calibrated (see "
     "<a href=\"#confidence\">What the confidence numbers mean</a>).</p>"
     "<p><b>In-domain checkpoints</b> are released models whose training data includes a bundled task. Their "
     "scores on that task are reported separately, because they have seen similar texts before.</p>"
     "<p><b>Classical supervised</b> baselines, such as TF-IDF with logistic regression or embeddings with a linear "
     "classifier, are fitted on the task's own train split. They show what a small, conventional model reaches with "
     "labelled data.</p>"
     "<p><b>Fine-tuned encoders</b> are pre-trained language models adapted to the task's train split, and models "
     "trained <b>from scratch</b> learn only from that split, starting from random weights.</p>"
     "<p>The score reports use these five approaches to group results, because a ranking across them compares "
     "different amounts of task data as much as different models. Entries that run on a <b>remote</b> service are "
     "flagged separately, because their timings include the network.</p>"
     "<p><b>Shortlist</b> variants handle tasks with more labels than a model accepts. An embedding model first keeps "
     "the most similar labels, and the main model chooses among them.</p>"),
    ("confidence", "What the confidence numbers mean",
     "<p>Every entry returns a confidence value and a score per label, but these numbers do not all mean the same "
     "thing. Four concepts are easy to mix up:</p>"
     "<dl class=\"concepts\">"
     "<dt>Raw score or margin</dt><dd>Useful for ordering labels, for example an SVM margin or a similarity. Not a "
     "probability, and not on a 0 to 1 scale.</dd>"
     "<dt>Normalized score</dt><dd>Non-negative and sums to 1 over the labels, for example a softmax of raw scores. "
     "It looks like a probability but may still be badly calibrated.</dd>"
     "<dt>Calibrated probability estimate</dt><dd>Checked against data so that, on similar texts, a stated 0.8 is "
     "right about 80% of the time. Calibration holds only for data like the data it was checked on.</dd>"
     "<dt>Separate confidence score</dt><dd>A number a service computes on its own terms, for example 1 minus the "
     "normalized entropy. Useful for ranking predictions, but not the selected label's probability.</dd></dl>"
     "<p>Each model page shows one of these confidence types as a badge, and the score reports use the same terms:</p>"
     + table(["Confidence type", "Meaning", "Entries"],
             [[h(label), h(text), ", ".join(f'<a href="{h(a.name)}.html"><code>{h(a.name)}</code></a>'
                                            for a in APPS if confidence_kind(a) == key) or "–"]
              for key, (label, text) in CONFIDENCE_KINDS.items()])
     + "<p>Many entries scale their scores with a temperature. In this benchmark it is always one positive scalar, "
       "so it changes the confidence values but never the predicted label:</p>"
     + temperature_diagram()
     + "<p>None of these types is a guarantee on your data. Before gating on a threshold, check calibration and "
       "choose the threshold on a validation set from your own domain.</p>"),
    ("choose", "How to choose a model",
     "<p>The table lists every entry. To narrow it down for a deployment, answer these questions in order. Each "
     "answer points to a group in the table, not to a single best model: the score reports show how the entries in "
     "a group compare on each task.</p>"
     "<ul class=\"tree\">"
     "<li><b>Do you have labeled data for your task?</b><ul>"
     "<li>Yes, and you want the smallest and fastest option: <a href=\"#g-supervised\">supervised baselines</a> "
     "(TF-IDF, embeddings).</li>"
     "<li>Yes, and you can train an encoder: <a href=\"#g-finetune\">fine-tuned encoders</a>; the "
     "<a href=\"#g-scratch\">from-scratch networks</a> are a cheaper reference.</li>"
     "<li>No: <a href=\"#g-zero-shot\">zero-shot models</a>. Check <a href=\"#g-in-domain\">models trained on "
     "benchmark data</a> separately: their scores on those tasks are in-domain.</li></ul></li>"
     "<li><b>Does your task have more labels than the model accepts?</b> See the Labels column; use the "
     "<a href=\"#g-shortlist\">shortlist variants</a>, whose accuracy is capped by the shortlist's recall.</li>"
     "<li><b>May the text leave the machine?</b> If not, filter the table for <i>Local</i>; only entries marked as "
     "remote send text to a service.</li>"
     "<li><b>Do you need meaningful confidence, for example to auto-accept predictions?</b> Prefer entries whose "
     "<a href=\"#confidence\">confidence type</a> is calibrated, and validate the calibration and the threshold on "
     "your own data.</li>"
     "<li><b>Memory, latency or cost limits?</b> Each score report ranks the entries by time per item, interactive "
     "latency, memory and cost, and shows the accuracy-versus-speed frontier.</li></ul>"),
    ("scratch", "Training from scratch or fine-tuning",
     "<p>Two words get mixed up when people talk about building their own classifier. <b>Training from scratch</b> "
     "starts from random weights, so the task's train split must teach the model everything: which words matter, "
     "which are similar, and which label they point to. <b>Fine-tuning</b> starts from a model that was already "
     "pre-trained on huge amounts of text and only teaches it the labels. With limited labeled data, fine-tuning a "
     "suitable pretrained encoder often trains more efficiently and performs better than training a comparable "
     "neural encoder from random initialization; training a small network from scratch is cheap, fast, and the best way to learn how "
     "neural classifiers work.</p>"
     + SPECTRUM
     + "<p>The three entries in the <a href=\"#g-scratch\">Trained from scratch</a> group are small neural "
       "networks trained from random weights. They "
       "differ only in how many word vectors become one text vector: a <a href=\"scratch_fasttext.html\">"
       "fastText-style</a> average, a <a href=\"scratch_cnn.html\">CNN</a> that detects phrases, and a "
       "<a href=\"scratch_bilstm.html\">BiLSTM</a> that reads in order. A small Transformer is deliberately not among "
       "them: trained from scratch on data this size it gains little over these, and Transformers pay off when they "
       "are pre-trained and then fine-tuned.</p>"
     + "<p>The <a href=\"#g-finetune\">Fine-tuned encoders</a> group takes that route: two released pre-trained "
       "encoders, <a href=\"finetune_modernbert.html\">ModernBERT-base</a> and <a href=\"finetune_deberta.html\">"
       "DeBERTa-v3-base</a>, each fine-tuned on the task's train split. Their pages explain the concepts, the "
       "training procedure and how to reproduce it step by step.</p>"),
    ("read", "How to use this guide",
     "<p>The table below lists every runnable entry in <code>runner/apps.py</code>, grouped by kind. Each name links to "
     "a page that covers the same ten topics in the same order: what the model is, what it needs, how to get and "
     "install it, how to send it data and read its output (with a recorded sample), how the benchmark prepares data, "
     "how to run it, how to train or adapt it, how to read the results, and what to check when something fails.</p>"
     "<p>Several entries share one folder and differ only in arguments, for example the three Laya checkpoints or the "
     "two SVM calibrations. Each still has its own page and its own results.</p>"
     "<p>The last two groups are trained by this project on the task's train split: small neural networks "
     "<b>trained from scratch</b>, and released pre-trained encoders <b>fine-tuned</b> on the task. Their pages follow "
     "a different outline from the others: concepts, training procedure, and step-by-step reproduction. Entries marked "
     "<b>Planned</b> are designed but not yet implemented.</p>"
     "<p>Two further models are described in <a href=\"#worth-knowing\">Worth knowing</a>. They were not "
     "benchmarked, so they have no page of their own.</p>"),
    ("worth-knowing", "Worth knowing: models not benchmarked",
     "<p>The two models below are relevant to this comparison, and an adapter for each exists in "
     "<code>classifiers/</code>. They were not benchmarked, because their hardware requirements are beyond the "
     "Apple Silicon Mac with 32 GB of memory used for every other result. Without a run there are no accuracy, speed "
     "or memory figures, no recorded sample, and no verified setup steps, so this guide does not give them a model "
     "page.</p>"
     + table(["Model", "What it is", "Why it was not run"], [
         ['<a href="https://huggingface.co/Patrick-18/Dexy-Decision-9B">Dexy-Decision-9B</a><br>'
          '<span class="aside">Patrick-18 &middot; <code>classifiers/dexy/</code></span>',
          "An experimental 9B decision checkpoint. A verbalizer head scores 2 to 256 options in one pass and returns "
          "a probability for each, with the release's own temperature.",
          "The release is built and verified for Linux with an NVIDIA GPU (CUDA). Its BF16 weights are 17.9 GB, more "
          "than half of the Mac's memory, and it has not been verified on Apple MPS. The setup script refuses to "
          "download it unless <code>DEXY_ALLOW_EXPERIMENTAL=1</code> is set."],
         ['<a href="https://github.com/Davipar/djev-dev">djev</a> with DiffusionGemma 26B-A4B<br>'
          '<span class="aside">djev (server), Google (model) &middot; <code>classifiers/djev/</code></span>',
          "A decision server that runs DiffusionGemma 26B-A4B in BF16 with vLLM and answers the same "
          "<code>/v1/systemone</code> choice questions as the Jev API.",
          "The BF16 weights alone are about 52 GB. The deploy script needs a Linux host with an NVIDIA B200 GPU, "
          "Docker, the NVIDIA Container Toolkit and about 128 GB of RAM, reached through an SSH tunnel. No such host "
          "was available."],
     ])
     + "<p>Dexy is worth studying as the largest open decision checkpoint in this family. It shows how a "
       "verbalizer head turns a 9B language model into an option scorer. djev is the reference way to serve "
       "DiffusionGemma for decisions. The same model can be measured on a Mac through "
       "<a href=\"OpenJev.html\"><code>OpenJev</code></a>, which serves it in four-bit MLX, accepts the same "
       "requests, and has results in the reports. The difference between the two is the precision: four-bit "
       "weights instead of BF16.</p>"
     "<p>Both entries remain in <code>runner/apps.py</code>. With the right hardware, they run like any other "
       "entry, and their results would appear in the reports.</p>"),
]


def index_page() -> str:
    counts = {g: sum(1 for a in APPS if group(a) == g) for g, _, _ in GROUPS}
    counts[SCRATCH_GROUP[0]] = len(SCRATCH)
    counts[FINETUNE_GROUP[0]] = len(FINETUNE)
    band = "".join(f'<a href="#g-{g}"><span class="num">{i:03d}</span><span class="k">{h(name)}</span>'
                   f'<span class="v">{counts[g]} entries</span></a>'
                   for i, (g, name, _) in enumerate(GROUPS + [SCRATCH_GROUP, FINETUNE_GROUP], 1))
    intro_toc = "".join(f'<li><a href="#{sid}"><span>{i:02d}</span>{h(name)}</a></li>'
                        for i, (sid, name, _) in enumerate(INTRO, 1))
    table_num = f"{len(INTRO) + 1:02d}"
    intro = "".join(f'<section id="{sid}"><p class="secnum">{i:02d}</p><h2>{h(name)}</h2>{body}</section>'
                    for i, (sid, name, body) in enumerate(INTRO, 1))
    column_header = ('<tr class="column-row"><th scope="col">No.</th><th scope="col">Registry name</th>'
                     '<th scope="col">Model</th><th scope="col">Runs on</th><th scope="col">Labels</th>'
                     '<th scope="col"><span class="sr">Guide</span></th></tr>')
    rows = []
    number = {a.name: i for i, a in enumerate(APPS, 1)}
    for g, name, desc in GROUPS:
        members = [a for a in APPS if group(a) == g]
        rows.append(f'<tbody id="g-{g}" class="grp"><tr class="group-row"><th colspan="6" scope="colgroup">'
                    f'<span class="gname">{h(name)}</span><span class="gdesc">{h(desc)}</span></th></tr>'
                    f'{column_header}')
        for a in members:
            model = facts(a)[0]
            search = " ".join([a.name, a.display_title, model, a.runs_on, GROUP_TYPE[g]]).lower()
            rows.append(f'<tr data-search="{h(search)}"><td class="num">{number[a.name]:03d}</td>'
                        f'<td class="name"><a href="{h(a.name)}.html">{h(a.name)}</a></td>'
                        f'<td>{h(a.display_title)}</td><td>{h(a.runs_on)}</td><td>{h(label_limit(a))}</td>'
                        f'<td class="go"><a href="{h(a.name)}.html" aria-label="Read the {h(a.name)} guide">Read guide</a>'
                        f'</td></tr>')
        rows.append("</tbody>")
    g, name, desc = SCRATCH_GROUP
    if any(not is_implemented(p) for p in SCRATCH):
        desc += PLANNED_NOTE
    rows.append(f'<tbody id="g-{g}" class="grp"><tr class="group-row"><th colspan="6" scope="colgroup">'
                f'<span class="gname">{h(name)}</span><span class="gdesc">{h(desc)}</span></th></tr>'
                f'{column_header}')
    for p in SCRATCH:  # implemented and planned together
        state = "implemented" if is_implemented(p) else "planned"
        badges = '<span class="badges"><span class="badge planned">Planned</span></span>' if state == "planned" else ""
        search = " ".join([p["name"], p["title"], p["short"], p["runs_on"], state,
                           "supervised trained from scratch"]).lower()
        rows.append(f'<tr data-search="{h(search)}"><td class="num">{scratch_number(p):03d}</td>'
                    f'<td class="name"><a href="{h(p["name"])}.html">{h(p["name"])}</a></td>'
                    f'<td>{h(p["title"])}{badges}</td>'
                    f'<td>{h(p["runs_on"])}</td><td>{TRAINED_LIMIT}</td>'
                    f'<td class="go"><a href="{h(p["name"])}.html" aria-label="Read the {h(p["name"])} guide">'
                    f'Read guide</a></td></tr>')
    rows.append("</tbody>")
    g, name, desc = FINETUNE_GROUP
    rows.append(f'<tbody id="g-{g}" class="grp"><tr class="group-row"><th colspan="6" scope="colgroup">'
                f'<span class="gname">{h(name)}</span><span class="gdesc">{h(desc)}</span></th></tr>'
                f'{column_header}')
    for p in FINETUNE:
        badges = "" if ft_implemented(p) else '<span class="badges"><span class="badge planned">Planned</span></span>'
        search = " ".join([p["name"], p["title"], p["short"], p["repo"], p["runs_on"],
                           "supervised fine-tuned"]).lower()
        rows.append(f'<tr data-search="{h(search)}"><td class="num">{ft_number(p):03d}</td>'
                    f'<td class="name"><a href="{h(p["name"])}.html">{h(p["name"])}</a></td>'
                    f'<td>{h(p["title"])}{badges}</td>'
                    f'<td>{h(p["runs_on"])}</td>'
                    f'<td>{TRAINED_LIMIT}</td>'
                    f'<td class="go"><a href="{h(p["name"])}.html" aria-label="Read the {h(p["name"])} guide">'
                    f'Read guide</a></td></tr>')
    rows.append("</tbody>")
    table = '<div class="table-wrap"><table class="catalog">' + "".join(rows) + "</table></div>"
    script = ("<script>const q=document.querySelector('#search'),rows=[...document.querySelectorAll('tr[data-search]')],"
              "groups=[...document.querySelectorAll('tbody.grp')],empty=document.querySelector('#empty');"
              "q.addEventListener('input',()=>{const s=q.value.toLowerCase().trim();"
              "rows.forEach(r=>r.hidden=!r.dataset.search.includes(s));"
              "groups.forEach(g=>g.hidden=![...g.querySelectorAll('tr[data-search]')].some(r=>!r.hidden));"
              "empty.hidden=rows.some(r=>!r.hidden)});</script>")
    return (head("Text Classification Benchmark and Guide",
                 f"A benchmark of {len(APPS)} text classifiers on AG News and Banking77, and a guide to running, "
                 f"training and fine-tuning them.")
            + topbar("models")
            + f'<main><div class="pagehead"><div class="wrap"><p class="crumb"><span>Model guide</span><span>/</span>'
              f'<span>{len(APPS)} entries</span></p><h1>Text Classification Benchmark and Guide</h1>'
              f'<p class="lede">A benchmark and a hands-on guide to text classification. {len(APPS)} classifiers, from '
              f'TF-IDF baselines and networks trained from scratch to fine-tuned encoders, zero-shot models and LLMs, '
              f'are run on the same AG News and Banking77 test sets. Each model page explains what the model is, what '
              f'it needs, how to run it and call it from Python, and how to read its results; the trained entries walk '
              f'through training from scratch and fine-tuning step by step. Start with '
              f'<a href="#start">Get started</a>, or go straight to the <a href="#catalog">model table</a>.</p></div></div>'
            + f'<div class="band"><div class="wrap meta groups">{band}</div></div>'
            + f'<div class="wrap doc" id="intro"><aside class="toc"><p class="toc-title">Introduction</p>'
              f'<ol>{intro_toc}<li><a href="#catalog"><span>{table_num}</span>Model table</a></li></ol></aside>'
              f'<article>{intro}</article></div>'
            + f'<div class="wrap catalog-wrap" id="catalog"><div class="catalog-head"><div>'
              f'<p class="secnum">{table_num}</p>'
              f'<h2>Model table</h2></div><label class="search"><span>Filter</span><input id="search" type="search" '
              f'placeholder="Name, model, MLX, Ollama, supervised"></label></div>{table}'
              f'<p id="empty" hidden>No entry matches the filter.</p></div></main>'
            + footer() + script)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    expected = {"index.html"}
    scratch_names = {p["name"] for p in SCRATCH} | {p["name"] for p in FINETUNE}
    for i, app in enumerate(APPS):
        if app.name in scratch_names:  # written below, with the from-scratch layout
            continue
        name = f"{app.name}.html"
        (OUT / name).write_text(model_page(app, i), encoding="utf-8")
        expected.add(name)
    for i, p in enumerate(SCRATCH):
        name = f"{p['name']}.html"
        (OUT / name).write_text(scratch_page(p, i), encoding="utf-8")
        expected.add(name)
    for i, p in enumerate(FINETUNE):
        name = f"{p['name']}.html"
        (OUT / name).write_text(finetune_page(p, i), encoding="utf-8")
        expected.add(name)
    (OUT / "index.html").write_text(index_page(), encoding="utf-8")
    for old in OUT.glob("*.html"):
        if old.name not in expected:
            old.unlink()
    done = sum(1 for p in SCRATCH if is_implemented(p))
    print(f"Generated {len(APPS) - done - len(FINETUNE)} model pages, {len(SCRATCH)} from-scratch pages ({done} "
          f"implemented, {len(SCRATCH) - done} planned), {len(FINETUNE)} fine-tuned pages and index.html in {OUT} "
          f"({len(SAMPLES)} recorded samples)")


if __name__ == "__main__":
    main()
