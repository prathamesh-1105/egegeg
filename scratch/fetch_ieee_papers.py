import urllib.request
import json
import re

queries = [
    "https://api.openalex.org/works?filter=title.search:PhysioNet%20EEG&per-page=25&sort=cited_by_count:desc",
    "https://api.openalex.org/works?filter=title.search:motor%20imagery%20EEG%20classification&per-page=25&sort=cited_by_count:desc"
]

papers = []

for q in queries:
    req = urllib.request.Request(q, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            for item in data.get('results', []):
                title = item.get('title', '')
                year = item.get('publication_year')
                cites = item.get('cited_by_count', 0)
                source = item.get('primary_location', {}) or {}
                source_info = source.get('source') or {}
                venue = source_info.get('display_name', '')
                doi = item.get('doi', '')
                
                # Reconstruct abstract
                abstract = ""
                inv = item.get('abstract_inverted_index')
                if inv:
                    wpos = []
                    for word, pos_list in inv.items():
                        for p in pos_list:
                            wpos.append((p, word))
                    wpos.sort()
                    abstract = " ".join([w for _, w in wpos[:200]])

                papers.append({
                    "title": title,
                    "year": year,
                    "citations": cites,
                    "venue": venue,
                    "doi": doi,
                    "abstract": abstract
                })
    except Exception as e:
        print(f"Error fetching {q}: {e}")

# Deduplicate by title
unique_papers = {}
for p in papers:
    if p["title"] not in unique_papers:
        unique_papers[p["title"]] = p

sorted_papers = sorted(unique_papers.values(), key=lambda x: x["citations"], reverse=True)

print(f"Retrieved {len(sorted_papers)} unique papers across searches.\n")

for i, p in enumerate(sorted_papers[:15], 1):
    print(f"--- Paper #{i} ---")
    print(f"Title: {p['title']}")
    print(f"Year: {p['year']} | Citations: {p['citations']} | Venue: {p['venue']}")
    print(f"DOI: {p['doi']}")
    # Search abstract for percentage numbers
    acc_matches = re.findall(r'(\d{2}(?:\.\d{1,2})?\s?%)', p['abstract'])
    if acc_matches:
        print(f"Reported Accuracies found in Abstract snippet: {acc_matches}")
    print(f"Abstract Snippet: {p['abstract'][:300]}...\n")
