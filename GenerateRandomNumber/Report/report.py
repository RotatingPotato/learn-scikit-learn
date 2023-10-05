import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def randomNum(n):
    fNum = np.random.random() # 生成第一個隨機數
    # 生成剩下的隨機數
    num = [fNum]
    for i in range(1, n):
        nextNum = num[i - 1] + np.random.uniform(-0.1, 0.1)
    for i in range(len(num)): # 將數據標準化
        num[i] = num[i] / np.std(num)
    num = [min(max(number, 0), 1) for number in num] # 將數據限制在 0 到 1 之間
    return num

def main():
    # 設定起始年份
    startYear = 1923
    # 設定上下浮動範圍
    revRange = 0.5
    proRange = 0.2
    assRange = 0.1
    negAssRange = 0.05
    # 生成 100 筆模擬報表資料
    data = []
    for i in range(100):
        year = startYear + i                    # 年份
        revAll = randomNum(1)[0] * 10000000     # 生成總收入
        revAll = revAll * (1 + revRange * np.random.rand())
        profit = randomNum(1)[0] * 1000000      # 生成淨利潤
        profit = profit * (1 + proRange * np.random.rand())
        assAll = randomNum(1)[0] * 100000000    # 生成總資產
        assAll = assAll * (1 + assRange * np.random.rand())
        negAssAll = randomNum(1)[0] * 100000000 # 生成負總資產
        negAssAll = negAssAll * (1 + negAssRange * np.random.rand())
        # 將資料加入列表
        data.append([year, revAll, profit, assAll, negAssAll])
    # 將列表轉換為 Pandas 資料框
    df = pd.DataFrame(data, columns=["年份", "總收入", "淨利潤", "總資產", "負總資產"])
    # 輸出結果
    plt.plot(df["年份"], df["總收入"])
    plt.title("Report")
    plt.show()
    print(df)
    df.to_csv("report.csv")

if __name__ == "__main__":
    main()
