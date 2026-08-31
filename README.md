# Prism AI Agent Search 🔮

An autonomous, multi-agent AI research search platform designed to automate web research, topic synthesis, and factual verification.

---

## 🌟 Features

- **Multi-Engine Search Fallback:** Switches across Serper API, DuckDuckGo, and Wikipedia to guarantee search results.
- **Domain Trust Filtering:** Restricts search queries to high-authority domains (`.gov`, `.edu`, `who.int`, `nature.com`).
- **Zero-Shot Aspect Classification:** Automatically organizes retrieved content into Pros/Advantages and Cons/Drawbacks using Hugging Face pipelines.
- **Cross-Source Fact Verification:** Evaluates multi-domain source alignment to compute a Factual Accuracy Score.
- **Semantic Clustering:** Groups findings by topic using sentence embeddings (`all-MiniLM-L6-v2`) and hierarchical clustering.

---

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shreyasanjays2007-sketch/Prism-AI-Agent-Search.git](https://github.com/shreyasanjays2007-sketch/Prism-AI-Agent-Search.git)
   cd Prism-AI-Agent-Search
