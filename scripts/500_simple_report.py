'''
python3 500_simple_report.py -i ../0_reg/4digit/report_e
'''

import argparse
import os
from pathlib import Path
from bs4 import BeautifulSoup


def main():
    parser = argparse.ArgumentParser(description="Simple report parser for PIN analysis")
    parser.add_argument("--input-dir", "-i", type=str, required=True,
                        help="Input directory containing index.html report file")
    args = parser.parse_args()

    input_dir = args.input_dir
    
    # Get parent and grandparent folder names from input directory path
    # Example: ../0_reg/4digit/report_e_huang -> /0_reg/4digit/
    input_path = Path(input_dir).resolve()
    parent_name = input_path.parent.name        # e.g., "4digit"
    grandparent_name = input_path.parent.parent.name  # e.g., "0_reg"
    
    # Format as /grandparent/parent/
    folder_path = f"/{grandparent_name}/{parent_name}/"

    # Read the HTML file
    html_path = os.path.join(input_dir, 'index.html')
    
    if not os.path.exists(html_path):
        print(f"Error: {html_path} not found. Please check the input directory.")
        return

    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    # Find the table rows (skip the header)
    rows = soup.find_all('tr')[1:]

    ranks = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 3:
            rank_text = cols[2].get_text(strip=True)
            # Only keep numeric ranks
            try:
                rank = int(rank_text)
                ranks.append(rank)
            except ValueError:
                continue

    # Sort the ranks
    ranks.sort()

    # Append results to ./simple_report.txt
    output_path = '../simple_report.txt'
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'{folder_path}\n')
        f.write(f'{str(ranks)}\n\n')
    
    print(f"Appended to {output_path}")
    print(f'{folder_path}')
    print(ranks)


if __name__ == '__main__':
    main()
