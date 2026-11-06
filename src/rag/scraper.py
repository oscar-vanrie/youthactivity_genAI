import json
import os
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

# Constanten voor de scraper
START_URL = "https://chiro.be/spelletjes-en-liedjes/zoek-een-spel?search%5Bsearch_string%5D=&search%5Bcategory_id%5D=&search%5Btheme_id%5D=&search%5Barea_id%5D=&search%5Bintensity_id%5D=&search%5Bduration_from%5D=0&search%5Bduration_to%5D=1384&search%5Bgroup_count_from%5D=&search%5Bgroup_count_to%5D=&search%5Bage_from%5D=&search%5Bage_to%5D=&search%5Bplayer_count%5D=&op=Zoeken"
LINK_PATTERN = "/game/"

# Uitgebreide headers om te lijken op een echte browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7",
}


def scrape_site():
  game_links = set()
  page = 0

  print("Start met verzamelen van alle spel-URL's via paginering...")

  # 1. Paginering-lus om alle overzichtspagina's door te lopen
  while True:
    # Voeg het paginanummer toe aan de zoek-URL
    current_search_url = (
        f"{START_URL}&page={page}" if page > 0 else START_URL
    )
    print(f"Ophalen overzichtspagina {page}...")

    try:
      time.sleep(1.5)  # Beleefdheidspauze tussen overzichtspagina's
      response = requests.get(current_search_url, headers=HEADERS, timeout=10)
      response.raise_for_status()
    except Exception as e:
      print(f"Fout bij ophalen overzichtspagina {page}: {e}")
      break

    soup = BeautifulSoup(response.text, "html.parser")

    # Zoek alle spel links op de huidige pagina
    found_on_page = 0
    for a in soup.find_all("a", href=True):
      href = a["href"]
      if LINK_PATTERN in href:
        full_url = urljoin(START_URL, href)
        if full_url not in game_links:
          game_links.add(full_url)
          found_on_page += 1

    # Als er geen nieuwe spellen meer op deze pagina staan, zijn we klaar met zoeken
    if found_on_page == 0:
      print("Geen nieuwe spellen meer gevonden. Einde van de zoekresultaten.")
      break

    page += 1

  print(f"\nTotaal aantal unieke spel-URL's gevonden: {len(game_links)}")
  saved_games = []

  # 2. Doorloop alle gevonden detailpagina's
  for index, url in enumerate(game_links, start=1):
    print(
        f"[{index}/{len(game_links)}] Bezig met ophalen detailpagina: {url}"
    )
    try:
      time.sleep(2)  # Wacht minstens 2 seconden tussen requests
      res = requests.get(url, headers=HEADERS, timeout=10)
      res.raise_for_status()

      game_soup = BeautifulSoup(res.text, "html.parser")

      # Titel ophalen
      title_tag = game_soup.find("h1", class_="page-title")
      title = title_tag.get_text(strip=True) if title_tag else "Geen titel"

      # Linkerkant: Beschrijving en Materiaal
      description = ""
      material = []

      left_div = game_soup.find("div", class_="left")
      if left_div:
        paragraphs = left_div.find_all("p")
        desc_texts = [p.get_text(separator=" ", strip=True) for p in paragraphs]
        description = " ".join(desc_texts)

        # Materiaal ophalen uit de lijst onder de h3 met 'materiaal'
        mat_header = left_div.find(
            lambda tag: tag.name == "h3" and "materiaal" in tag.get_text().lower()
        )
        if mat_header:
          next_ul = mat_header.find_next_sibling("ul")
          if next_ul:
            material = [
                li.get_text(strip=True) for li in next_ul.find_all("li")
            ]

      # Rechterkant: Metadata (Duur, Groepen, Intensiteit, Leeftijd, Spelers, etc.)
      metadata = {
          "duur": None,
          "groepen": None,
          "intensiteit": None,
          "leeftijd": None,
          "spelers": None,
          "soort": [],
          "terrein": [],
          "themas": [],
      }

      right_div = game_soup.find("div", class_="right")
      if right_div:
        right_text = right_div.get_text(separator="\n")
        lines = [line.strip() for line in right_text.split("\n") if line.strip()]

        for i, line in enumerate(lines):
          lower_line = line.lower()
          if "duur:" in lower_line and i + 1 < len(lines):
            metadata["duur"] = lines[i + 1]
          elif "groepen:" in lower_line and i + 1 < len(lines):
            metadata["groepen"] = lines[i + 1]
          elif "intensiteit:" in lower_line and i + 1 < len(lines):
            metadata["intensiteit"] = lines[i + 1]
          elif "leeftijd:" in lower_line and i + 1 < len(lines):
            metadata["leeftijd"] = lines[i + 1]
          elif "spelers:" in lower_line and i + 1 < len(lines):
            metadata["spelers"] = lines[i + 1]

        # Categorieën, terreinen en thema's via de links in de rechterkolom
        for a_tag in right_div.find_all("a", href=True):
          href = a_tag["href"]
          val = a_tag.get_text(strip=True)
          if "category_id" in href:
            metadata["soort"].append(val)
          elif "area_id" in href:
            metadata["terrein"].append(val)
          elif "theme_id" in href:
            metadata["themas"].append(val)

      # Schoon overbodige witruimte op in de tekst
      description = " ".join(description.split())

      # Structuur toevoegen aan de lijst
      game_data = {
          "title": title,
          "source_url": url,
          "text": description,
          "material": material,
          "metadata": metadata,
      }
      saved_games.append(game_data)

    except Exception as e:
      print(f"Fout bij verwerken van pagina {url}: {e}")

  # 3. Sla alles op in data/raw/games_raw.json
  output_dir = os.path.join("data", "raw")
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, "games_raw.json")

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(saved_games, f, ensure_ascii=False, indent=4)

  print(f"\nKlaar! Totaal aantal opgeslagen spellen: {len(saved_games)}")
  print(f"Resultaten opgeslagen in: {output_path}")


if __name__ == "__main__":
  scrape_site()