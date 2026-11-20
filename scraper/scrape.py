"""Refresh scraper. Run where the target sites are reachable:
   python scraper/scrape.py  -> writes data/scraped.json (merged by the app at startup)."""
import json, re, sys, time, httpx
from pathlib import Path
from bs4 import BeautifulSoup
from langsmith import traceable

URLS = {
 "DarGlobal": ["https://darglobal.co.uk/investor","https://darglobal.co.uk/trump-tower-jeddah",
   "https://darglobal.co.uk/partners/the-trump-organization",
   "https://darglobal.co.uk/press/darglobal-partners-with-wasalt-to-advertise-trump-tower-jeddah",
   "https://darglobal.co.uk/press/darglobal-breaks-ground-on-trump-tower-jeddah"],
 "Wasalt": ["https://wasalt.sa/en","https://wasalt.sa/en/corporate","https://wasalt.sa/en/wb-daily-rentals","https://wasalt.com/en/about"],
}


@traceable(
    name="Fetch source page",
    run_type="tool",
    process_inputs=lambda inputs: {"url": inputs["url"]},
    process_outputs=lambda html: {"html_characters": len(html)},
)
def fetch_page(client, url):
    response = client.get(url)
    response.raise_for_status()
    return response.text


@traceable(
    name="Extract page content",
    run_type="chain",
    process_inputs=lambda inputs: {"company": inputs["company"], "url": inputs["url"], "html_characters": len(inputs["html"])},
    process_outputs=lambda result: {"title": result["title"], "text_characters": len(result["text"])},
)
def extract_page(company, url, html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    title = (soup.title.string or url).strip() if soup.title else url
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return {"company": company, "url": url, "title": title, "text": text}


@traceable(
    name="Chunk extracted page",
    run_type="chain",
    process_inputs=lambda inputs: {"title": inputs["title"], "text_characters": len(inputs["text"])},
    process_outputs=lambda result: {"chunk_count": len(result)},
)
def chunk_page(company, title, url, text, n=900):
    words, out, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        if sum(len(x)+1 for x in cur) > n: out.append(" ".join(cur)); cur=[]
    if cur:
        out.append(" ".join(cur))
    return [
        dict(company=company, title=f"{title} ({i + 1})", url=url, text=chunk)
        for i, chunk in enumerate(out)
    ]


@traceable(
    name="Write scraped documents",
    run_type="tool",
    process_inputs=lambda inputs: {"document_count": len(inputs["documents"]), "output_path": str(inputs["output_path"])},
    process_outputs=lambda count: {"document_count": count},
)
def write_documents(documents, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(documents, output_file, ensure_ascii=False, indent=1)
    return len(documents)


@traceable(
    name="Crawl delay",
    run_type="tool",
    process_inputs=lambda inputs: {"seconds": inputs["seconds"]},
    process_outputs=lambda result: result,
)
def crawl_delay(seconds):
    time.sleep(seconds)


@traceable(
    name="Scrape and ingest public sites",
    run_type="chain",
    process_outputs=lambda result: result,
)
def main():
    res=[]
    pages_scraped = 0
    with httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0 (research-bot)"}) as c:
        for co, urls in URLS.items():
            for u in urls:
                try:
                    html = fetch_page(c, u)
                except Exception as e:
                    print("skip",u,e,file=sys.stderr); continue
                page = extract_page(co, u, html)
                res.extend(chunk_page(page["company"], page["title"], page["url"], page["text"]))
                pages_scraped += 1
                crawl_delay(1)
    count = write_documents(res, Path("data/scraped.json"))
    print("chunks:",count)
    return {"document_count": count, "source_pages": pages_scraped}


if __name__=="__main__": main()
