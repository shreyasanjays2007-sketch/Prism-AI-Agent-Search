import os
import re
from typing import Dict, List
from urllib.parse import parse_qs, urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(page_title="Prism AI", page_icon="🔮", layout="wide")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Option B: Explicit Domain Whitelist ---
HIGH_AUTHORITY_DOMAINS = [
    "gov", "gov.in", "gov.uk", "europa.eu", "who.int", "un.org", "cdc.gov", "nih.gov",
    "bbc.com", "reuters.com", "apnews.com", "bloomberg.com", "ft.com", 
    "thehindu.com", "nytimes.com", "wsj.com", "theguardian.com",
    "edu", "nature.com", "sciencedirect.com", "arxiv.org", "ncbi.nlm.nih.gov", "wikipedia.org"
]

def is_trusted_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    if not domain:
        return False
    return any(domain.endswith(trusted) or f".{trusted}" in domain or domain == trusted for trusted in HIGH_AUTHORITY_DOMAINS)


@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    aspect_classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
    )
    return embedder, aspect_classifier

embedder, aspect_classifier = load_models()

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"([.,!?;:])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\s+'s", "'s", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def decode_ddg_url(raw_url: str) -> str:
    if "duckduckgo.com/l/?" in raw_url or "duckduckgo.com/r/?" in raw_url:
        parsed = urlparse(raw_url)
        query_params = parse_qs(parsed.query)
        if "uddg" in query_params:
            return query_params["uddg"][0]
    return raw_url

def filter_relevant_results(
    query: str, items: List[Dict[str, str]], min_relevance: float = 0.25
) -> List[Dict[str, str]]:
    if not items:
        return []

    query_emb = embedder.encode([query])
    item_texts = [item["content"] for item in items]
    item_embs = embedder.encode(item_texts, show_progress_bar=False)

    similarities = cosine_similarity(query_emb, item_embs)[0]

    relevant_items = []
    for idx, sim in enumerate(similarities):
        if sim >= min_relevance:
            relevant_items.append(items[idx])

    return relevant_items

def agent1_search_and_scrape(
    query: str, api_key: str = "", restrict_to_whitelist: bool = True
) -> List[Dict[str, str]]:
    collected_data = []
    clean_key = api_key.encode("ascii", "ignore").decode("ascii").strip()

    if clean_key:
        try:
            url = "https://google.serper.dev/search"
            payload = {"q": query, "num": 15}
            headers = {
                "X-API-KEY": clean_key,
                "Content-Type": "application/json; charset=utf-8",
            }

            response = requests.post(url, headers=headers, json=payload, timeout=5)
            if response.status_code == 200:
                results = response.json()
                for item in results.get("organic", []):
                    snippet = item.get("snippet", "")
                    target_link = item.get("link", "")
                    
                    if restrict_to_whitelist and not is_trusted_domain(target_link):
                        continue

                    if len(snippet.split()) > 5 and target_link:
                        collected_data.append({
                            "url": target_link,
                            "content": clean_text(snippet),
                        })
            else:
                st.warning(f"API Key error ({response.status_code}). Falling back to web scraping.")
        except Exception as e:
            st.warning(f"API request failed: {e}. Falling back to web scraping.")

    if not collected_data:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
        clean_topic = re.sub(
            r"^(is|what is|how is|are|does|can|should|pros and cons of|impact of)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        ).rstrip("?").strip()

        try:
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&titles={clean_topic}"
            resp = requests.get(wiki_url, headers=headers, timeout=5).json()
            pages = resp.get("query", {}).get("pages", {})
            wiki_link = f"https://en.wikipedia.org/wiki/{clean_topic.replace(' ', '_')}"
            
            if not restrict_to_whitelist or is_trusted_domain(wiki_link):
                for p_id, p_data in pages.items():
                    if p_id != "-1":
                        extract = p_data.get("extract", "")
                        for para in extract.split("\n"):
                            if len(para.split()) > 10:
                                collected_data.append({
                                    "url": wiki_link,
                                    "content": clean_text(para),
                                })
        except Exception:
            pass

        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            res = requests.get(ddg_url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            
            results = soup.find_all("div", class_="result")
            for result in results:
                snippet_elem = result.find("a", class_="result__snippet")
                url_elem = result.find("a", class_="result__url")
                
                if snippet_elem:
                    text = snippet_elem.get_text(strip=True)
                    raw_href = url_elem.get("href", "") if url_elem else snippet_elem.get("href", "")
                    target_url = decode_ddg_url(raw_href) if raw_href else ddg_url

                    if not target_url.startswith("http"):
                        target_url = "https://" + target_url.lstrip("/")

                    if restrict_to_whitelist and not is_trusted_domain(target_url):
                        continue

                    if len(text.split()) > 6:
                        collected_data.append({
                            "url": target_url,
                            "content": clean_text(text),
                        })
        except Exception:
            pass

    return filter_relevant_results(query, collected_data)

def remove_duplicates(
    items: List[Dict[str, str]], sim_threshold: float = 0.80
) -> List[Dict[str, str]]:
    unique_exact = []
    seen = set()
    for item in items:
        norm = item["content"].lower()
        if norm not in seen:
            seen.add(norm)
            unique_exact.append(item)

    if not unique_exact:
        return []

    texts = [i["content"] for i in unique_exact]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    keep_indices = []
    dropped_indices = set()
    for i in range(len(unique_exact)):
        if i in dropped_indices:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(unique_exact)):
            if sim_matrix[i][j] >= sim_threshold:
                dropped_indices.add(j)

    return [unique_exact[idx] for idx in keep_indices]

def cluster_data(
    items: List[Dict[str, str]], distance_threshold: float = 0.55
) -> pd.DataFrame:
    texts = [i["content"] for i in items]
    embeddings = embedder.encode(texts, show_progress_bar=False)

    if len(items) == 1:
        df = pd.DataFrame(items)
        df["group_id"] = 0
        return df

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(embeddings)

    df = pd.DataFrame(items)
    df["group_id"] = labels
    return df.sort_values("group_id").reset_index(drop=True)

def extract_aspects_agents(
    items: List[Dict[str, str]], confidence_threshold: float = 0.35
) -> Dict[str, List[Dict[str, str]]]:
    labels = [
        "advantage, benefit, growth, positive impact",
        "drawback, risk, issue, challenge, negative impact",
        "neutral statement or factual description",
    ]
    pros, cons = [], []

    for item in items[:15]:
        text = item["content"]
        res = aspect_classifier(text, candidate_labels=labels)
        top_label, top_score = res["labels"][0], res["scores"][0]

        if top_score >= confidence_threshold:
            if "advantage" in top_label:
                pros.append(item)
            elif "drawback" in top_label:
                cons.append(item)

    if not pros and not cons:
        for item in items[:2]:
            pros.append(item)

    return {"pros": pros, "cons": cons}

# --- REVISED AGENT 4: Strict Cross-Source Fact Verification Scoring ---
def agent4_generate_summary(
    items: List[Dict[str, str]], query: str, aspects_dict: Dict[str, List[Dict[str, str]]], restricted: bool
) -> str:
    texts = [i["content"] for i in items]
    urls = [i.get("url", "") for i in items]
    num_items = len(texts)

    if num_items > 1:
        embs = embedder.encode(texts, show_progress_bar=False)
        sim_matrix = cosine_similarity(embs)

        # Count how many claims are verified across DIFFERENT domains (Cosine Sim >= 0.45)
        corroborated_claims = 0
        for i in range(num_items):
            domain_i = urlparse(urls[i]).netloc
            has_independent_support = False
            for j in range(num_items):
                if i != j:
                    domain_j = urlparse(urls[j]).netloc
                    # Check if another domain states a semantically similar claim
                    if domain_i != domain_j and sim_matrix[i][j] >= 0.45:
                        has_independent_support = True
                        break
            if has_independent_support:
                corroborated_claims += 1

        correctness_score = (corroborated_claims / num_items) * 100

        if correctness_score >= 60.0:
            fact_check = (
                f"✅ **Factual Accuracy Score: HIGH ({correctness_score:.1f}%)**\n\n"
                f"**{corroborated_claims} out of {num_items} claims** gathered were independently "
                f"corroborated across distinct whitelisted domains, demonstrating high factual consensus."
            )
        elif correctness_score >= 30.0:
            fact_check = (
                f"⚠️ **Factual Accuracy Score: MODERATE ({correctness_score:.1f}%)**\n\n"
                f"**{corroborated_claims} out of {num_items} claims** were corroborated across multiple "
                f"sources. Some claims originate from single sources and require further manual verification."
            )
        else:
            fact_check = (
                f"❓ **Factual Accuracy Score: LOW / UNVERIFIED ({correctness_score:.1f}%)**\n\n"
                f"Most collected claims come from uncorroborated, single sources with low consensus. "
                f"Treat individual findings as unverified claims."
            )
    else:
        correctness_score = 0.0
        fact_check = "ℹ️ **Factual Accuracy Score:** Insufficient data (single snippet found) to calculate cross-source verification."

    key_findings = [i["content"] for i in items[:4]]
    mode_label = "Whitelisted High-Authority Sources (.gov, .edu, BBC, Reuters)" if restricted else "All Web Sources"
    
    summary = f"### 📌 Research Overview\n\nVerified via **{mode_label}** for **'{query}'**:\n\n"
    for idx, point in enumerate(key_findings, 1):
        summary += f"{idx}. {point}\n\n"

    summary += f"---\n\n### 🛡️ Cross-Source Factual Verification\n\n{fact_check}\n\n---\n\n### ⚖️ Reliability Assessment\n\n"

    if correctness_score >= 60.0:
        summary += "The gathered facts demonstrate strong empirical consensus across independent, authoritative domains."
    elif correctness_score >= 30.0:
        summary += "Findings present moderate factual reliability, with mixed consensus across reporting outlets."
    else:
        summary += "Findings display low factual overlap across sources; further primary research is recommended."

    return summary

# --- Sidebar Configuration ---
st.sidebar.title("🔑 Configuration")
serper_api_key = st.sidebar.text_input(
    "Serper Search API Key (Optional):",
    type="password",
)

restrict_to_whitelist = st.sidebar.toggle(
    "🛡️ Restrict to Trusted Sources",
    value=True,
    help="Filters scraped URLs against an explicit whitelist of government, academic, and top news domains.",
)

# --- Main UI ---
st.title("🔮 Prism AI")
st.caption("Multi-agent system with Cross-Source Factual Corroboration Scoring.")

query = st.text_input(
    "Enter a topic or topic query:", placeholder="e.g., Global Economic Forecast or Public Health Guidelines"
)

if st.button("Run Research Pipeline", type="primary"):
    if not query.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.status("Gathering and Processing Data...", expanded=True) as status:
            st.write("1️⃣ **Agent 1:** Querying and validating domains against whitelist...")

            raw_data = agent1_search_and_scrape(
                query, api_key=serper_api_key, restrict_to_whitelist=restrict_to_whitelist
            )

            if not raw_data:
                status.update(
                    label="No results matched the whitelisted domains for this topic. Try disabling the trust toggle in the sidebar.",
                    state="error",
                )
            else:
                st.write("2️⃣ **System:** Eliminating semantic duplicates...")
                clean_data = remove_duplicates(raw_data)

                st.write("3️⃣ **System:** Clustering findings into semantic groups...")
                grouped_df = cluster_data(clean_data)

                st.write("4️⃣ **Agent 2 & Agent 3:** Extracting Pros and Cons...")
                aspects_dict = extract_aspects_agents(clean_data)

                st.write("5️⃣ **Agent 4:** Calculating Cross-Source Factual Accuracy...")
                summary_text = agent4_generate_summary(clean_data, query, aspects_dict, restrict_to_whitelist)

                st.session_state["aspects_dict"] = aspects_dict
                st.session_state["grouped_df"] = grouped_df
                st.session_state["summary_text"] = summary_text
                st.session_state["has_data"] = True

                status.update(
                    label="Multi-Agent Analysis Complete!", state="complete", expanded=False
                )

# Persistent UI Render Block
if st.session_state.get("has_data", False):
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "✅ Pros / Advantages",
        "❌ Cons / Drawbacks",
        "📂 Restructured Groups",
        "📝 Summary & Conclusion",
    ])

    aspects = st.session_state["aspects_dict"]

    with tab1:
        st.subheader("✅ Pros / Advantages *(Agent 2)*")
        if aspects["pros"]:
            for idx, item in enumerate(aspects["pros"], 1):
                with st.container(border=True):
                    st.markdown(f"**{idx}.** {item['content']}")
                    st.markdown(f"🔗 **Source Article:** [Read Full Source ({urlparse(item.get('url', '')).netloc})]({item.get('url', '#')})")
        else:
            st.info("Agent 2 found no explicit advantages in the whitelisted sources.")

    with tab2:
        st.subheader("❌ Cons / Drawbacks *(Agent 3)*")
        if aspects["cons"]:
            for idx, item in enumerate(aspects["cons"], 1):
                with st.container(border=True):
                    st.markdown(f"**{idx}.** {item['content']}")
                    st.markdown(f"🔗 **Source Article:** [Read Full Source ({urlparse(item.get('url', '')).netloc})]({item.get('url', '#')})")
        else:
            st.info("Agent 3 found no explicit drawbacks in the whitelisted sources.")

    with tab3:
        st.subheader("Clustered Findings *(Data via Agent 1)*")
        for group_id, group in st.session_state["grouped_df"].groupby("group_id"):
            with st.expander(
                f"Group {group_id + 1} ({len(group)} items)", expanded=True
            ):
                for _, row in group.iterrows():
                    st.markdown(f"• {row['content']}")
                    st.markdown(f"🔗 [Read Full Source ({urlparse(row['url']).netloc})]({row['url']})")

    with tab4:
        st.subheader("Practical Conclusion & Fact-Check *(Agent 4)*")
        st.markdown(st.session_state["summary_text"])