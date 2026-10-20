import json
import urllib.request
import urllib.parse

def search_openalex(query, count=10):
    url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per_page={count}&filter=has_doi:true"
    req = urllib.request.Request(url, headers={'User-Agent': 'EEG-Research-Bot/1.0 (mailto:researcher@example.com)'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = []
            for item in data.get('results', []):
                title = item.get('display_name')
                year = item.get('publication_year')
                doi = item.get('doi')
                citations = item.get('cited_by_count')
                venue = item.get('primary_location', {}).get('source', {}).get('display_name')
                abstract_inverted = item.get('abstract_inverted_index')
                abstract = ""
                if abstract_inverted:
                    word_pos = []
                    for k, v in abstract_inverted.items():
                        for pos in v:
                            word_pos.append((pos, k))
                    word_pos.sort()
                    abstract = " ".join([w[1] for w in word_pos[:100]]) + "..."
                
                results.append({
                    "title": title,
                    "year": year,
                    "doi": doi,
                    "venue": venue,
                    "citations": citations,
                    "abstract": abstract
                })
            return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    q1 = search_openalex("PhysioNet EEG motor imagery classification IEEE", 10)
    q2 = search_openalex("PhysioNet motor imagery cross-subject accuracy IEEE", 10)
    
    with open("scratch/ieee_openalex_results.json", "w") as f:
        json.dump({"query1": q1, "query2": q2}, f, indent=2)
    print("Done fetching IEEE literature via OpenAlex.")
