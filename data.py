
import requests
import pandas as pd
from bs4 import BeautifulSoup


def GetStockData():

    finance_root = "https://finance.naver.com"

    url = 'https://finance.naver.com/sise/sise_market_sum.nhn'

    html = requests.get(url).text

    soup = BeautifulSoup(html, "html.parser")

    stocks = soup.select(".type_2 > tbody > tr > td > a")

    codes = [stock.attrs['href'].split("=")[1]
             for stock in stocks][::2]  # stock_code
    names = [stock.text for stock in stocks if stock.text != ""]  # stock_name

    datas = []

    for code, name in zip(codes, names):
        print(name)
        df = pd.DataFrame()

        for i in range(1, 21):
            sise_day = f'https://finance.naver.com/item/sise_day.nhn?code={code}&page={i}'
            data = pd.read_html(sise_day, header=0)
            data = data[0][['날짜', '종가', '시가', '고가', '저가', '거래량']]
            data = data.drop([0, 6, 7, 8, 14])

            if df.empty:
                df = data
            else:
                df = pd.concat([df, data])
        df.reset_index(drop=True, inplace=True)

        df.to_csv(f'data/{name}.csv', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    GetStockData()
