"""Re-download the 24 raw genome sequences from NCBI.

    python download_raw_genomes.py raw

The raw FASTA files are about 90 MB in total and are not kept in this repository, because they are
public reference sequences that can be fetched by accession. `dataset.csv` already contains the
9,600 extracted fragments, so you only need these if you want to rebuild the dataset from scratch
or repeat the 16-mer sharing comparison between E. coli O157:H7 and K-12.

Accessions come from genomes.csv next to this script. Files are written as raw_<accession>.fasta,
the naming the analysis scripts expect. Existing files are skipped, so the script can be re-run
after an interruption.

NCBI asks automated clients to identify themselves and to stay under three requests per second
without an API key; this script sleeps between requests and sends a generic client name.
"""
import sys, os, time, urllib.request, urllib.error
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else 'raw'
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
URL = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
       '?db=nuccore&id={acc}&rettype=fasta&retmode=text')

genomes = pd.read_csv(os.path.join(HERE, 'genomes.csv'))
print(f'{len(genomes)} accessions, writing to {OUT}/', flush=True)

for n, row in enumerate(genomes.itertuples(), 1):
    acc = row.genome_id
    path = os.path.join(OUT, f'raw_{acc}.fasta')
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f'[{n}/{len(genomes)}] {acc} already present, skipping', flush=True)
        continue
    req = urllib.request.Request(URL.format(acc=acc), headers={'User-Agent': 'genome-fetch-script'})
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            if not body.startswith(b'>'):
                raise ValueError('response was not FASTA')
            with open(path, 'wb') as f:
                f.write(body)
            print(f'[{n}/{len(genomes)}] {acc} {len(body) / 1e6:.1f} MB  {row.organism}', flush=True)
            break
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            if attempt == 3:
                print(f'[{n}/{len(genomes)}] {acc} FAILED after 3 attempts: {e}', flush=True)
            else:
                time.sleep(5 * attempt)
    time.sleep(0.5)                      # stay well under NCBI's rate limit

print('done', flush=True)
