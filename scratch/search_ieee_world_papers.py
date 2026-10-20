import urllib.request
import json

def search_openalex_papers():
    # Query OpenAlex for PhysioNet Motor Imagery papers in IEEE / top journals
    url = "https://api.openalex.org/works?search=PhysioNet+motor+imagery+EEG+classification&per-page=20&sort=cited_by_count:desc"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = []
            for item in data.get('results', []):
                title = item.get('title')
                year = item.get('publication_year')
                citations = item.get('cited_by_count')
                venue = item.get('primary_location', {}).get('source', {}).get('display_name')
                doi = item.get('doi')
                
                abstract = ""
                inverted = item.get('abstract_inverted_index')
                if inverted:
                    word_pos = []
                    for word, positions in inverted.items():
                        for pos in positions:
                            word_pos.append((pos, word))
                    word_pos.sort()
                    abstract = " ".join([w for _, w in word_pos[:150]])

                results.append({
                    "title": title,
                    "year": year,
                    "citations": citations,
                    "venue": venue,
                    "doi": doi,
                    "abstract": abstract
                })
            
            with open("scratch/world_ieee_eeg_papers.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Successfully retrieved {len(results)} top academic papers.")
    except Exception as e:
        print(f"Error searching OpenAlex: {e}")

if __name__ == "__main__":
    search_openalex_papers()
