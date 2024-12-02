import pandas as pd

def process_waste_data(file_path):
    # 读取数据
    df = pd.read_csv(file_path)

    # 删除Waste_Type中存在Glass、Metal、Paper、Plastic的行
    df = df[~df['Waste_Type'].isin(['Glass', 'Metal', 'Paper', 'Plastic'])]

    # 按照Location分组并保存为不同的CSV文件
    for location, group in df.groupby('Location(city)'):
        # 删除Location这一列
        group = group.drop(columns=['Location(city)'])
        # 保存为CSV文件
        group.to_csv(f"{location}_waste_data.csv", index=False)

# 调用函数
process_waste_data(r'E:\项目\垃圾分类\code\test.txt')