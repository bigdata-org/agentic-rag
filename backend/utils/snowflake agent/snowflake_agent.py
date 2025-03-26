import pandas as pd
from dotenv import load_dotenv
import os 
from snowflake.connector.pandas_tools import write_pandas
from datetime import datetime
from dateutil.relativedelta import relativedelta
from firecrawl import FirecrawlApp
from snowflake_connector import conn
import re


load_dotenv()
app = FirecrawlApp(api_key=os.getenv('FIRE_CRAWL'))
cursor = conn.cursor()

def get_year_and_qtr(date_str):
    if date_str == 'Current':
        today = datetime.today() - relativedelta(months=1)
    else:
        today = pd.to_datetime(date_str, errors='coerce') - relativedelta(months=1)

    year = today.year
    qtr = (today.month - 1) // 3 + 1
    return year, qtr



def etl_snowflake():
    response = app.scrape_url(url='https://finance.yahoo.com/quote/NVDA/key-statistics/', params={
        'formats': [ 'markdown' ],
        'includeTags': [ 'table' ]
    })

    # Assuming `response['markdown']` contains your Markdown table
    data = response['markdown']

    # Regex pattern to extract the Markdown table
    pattern = r"(\|.*?Current.*?\|\n\|.*?\|\n(?:\|.*?\|\n)+)"

    # Search for the table in the text
    match = re.search(pattern, data, re.DOTALL)

    if match:
        table_text = match.group(1)  
        rows = [row.strip() for row in table_text.split("\n") if row.strip()]
        data = [row.strip('|').split('|') for row in rows]
        df = pd.DataFrame(data[1:], columns=data[0])  # Skip first two rows
        df.columns = [col.strip() for col in df.columns]
    
    df.rename(columns={'': 'column'}, inplace=True)
    df_cleaned = df.iloc[1:].reset_index(drop=True)
    df_cleaned.set_index('column', inplace=True)

    df_transposed = df_cleaned.transpose()
    df_transposed[' Market Cap '] = (df_transposed[' Market Cap '].astype(str).str.replace('T', '', regex=False).astype(float).round(2))
    df_transposed[' Enterprise Value '] = (df_transposed[' Enterprise Value '].astype(str).str.replace('T', '', regex=False).astype(float).round(2))
    # df_transposed_reset = df_transposed.reset_index()
    df_transposed=df_transposed.reset_index().rename(columns={'index': 'DATE_PERIOD'})
    df_transposed.columns=["DATE_PERIOD","MARKET_CAP","ENTERPRISE_VALUE","TRAILING_PE","FORWARD_PE","PEG_RATIO","SALES_PRICE","BOOK_PRICE","REVENUE","EBITDA"]

    df_transposed[['YEAR', 'QTR']] = df_transposed['DATE_PERIOD'].apply(lambda x: pd.Series(get_year_and_qtr(x)))
    cursor = conn.cursor()
    cursor.execute('TRUNCATE TABLE VALUATION_METRICS')
    success, num_chunks, num_rows, _ = write_pandas(conn, df_transposed, "VALUATION_METRICS")
    
    return success

