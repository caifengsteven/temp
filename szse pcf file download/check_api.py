"""Test different pagination parameter names."""
import requests

url = 'https://www.szse.cn/api/report/ShowReport'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.szse.cn/www/disclosure/fund/currency/',
}

# Try different parameter names for page
param_names = ['pageno', 'PAGENO', 'pageNo', 'page', 'PAGE', 'pagenum', 'PAGENUM']

for param_name in param_names:
    print(f"\nTrying param '{param_name}'...")

    first_record_p1 = None
    first_record_p2 = None

    for page in [1, 2]:
        params = {
            'SHOWTYPE': 'JSON',
            'CATALOGID': 'sgshqd',
            'txtStart': '2026-01-08',
            'txtEnd': '2026-01-09',
            param_name: page,
            'pagesize': 20,
        }

        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code != 200 or len(r.text) < 100:
            print(f'  Page {page}: Error {r.status_code}')
            continue

        data = r.json()
        records = data[0]['data']

        # Get first record's jjdm for comparison
        first_jjdm = records[0].get('jjdm', '')[:50] if records else 'EMPTY'

        if page == 1:
            first_record_p1 = first_jjdm
        else:
            first_record_p2 = first_jjdm

    if first_record_p1 and first_record_p2:
        if first_record_p1 == first_record_p2:
            print(f'  Result: SAME data on both pages')
        else:
            print(f'  Result: DIFFERENT data - THIS WORKS!')
            print(f'  Page 1 first: {first_record_p1[:30]}...')
            print(f'  Page 2 first: {first_record_p2[:30]}...')

