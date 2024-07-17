import pandas as pd
import numpy as np
import openpyxl
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import poisson, chisquare, probplot
import seaborn as sns
#将frenet-down.xlsx文件中的五个子表 lane1, lane2, lane3, lane4, lane5 合并成一个数据表。
def Combined_frenet_down(stats_time,end_time):
    # 读取 Excel 文件
    file_path = 'dataset/frenet-down.xlsx'
    xls = pd.ExcelFile(file_path, data_only=True)
    # 读取所有子表
    data_frames = []
    total_filtered_count = 0
    total_initial_count = 0
    for sheet_name in xls.sheet_names:
        # 读取每个子表
        df = pd.read_excel(xls, sheetname=sheet_name)
        # 打印每个子表过滤前的数据数量
        initial_count = len(df)
        total_initial_count += initial_count
        print(f"{sheet_name}过滤前的数据数量: {initial_count}")
        # 过滤时间戳在120-240之间的数据
        df_filtered = df[(df['time(s)'] >= stats_time) & (df['time(s)'] <= end_time)]
        data_frames.append(df_filtered)
        # 打印每个子表过滤后的数据数量
        filtered_count = len(df_filtered)
        total_filtered_count += filtered_count
        print(f"{sheet_name}过滤后的数据数量: {filtered_count}")
    # 合并所有子表
    combined_df = pd.concat(data_frames, ignore_index=True)
    # 保存合并后的数据到新的 Excel 文件
    save_path= 'dataset/'+str(stats_time)+'-'+str(end_time)+'s_combined_frenet_down.csv'
    combined_df.to_csv(save_path, index=False)
    # 打印汇总后的数据数量
    print(f"所有子表汇总后的数据数量: {total_filtered_count}")
    print("子表合并完成，结果已保存到"+save_path)

#删除time(s)有重复值的车辆数据
#也就是每个时刻，车辆只能被记录一次，清理违规数据（收集过程精度问题）
def Cleaned_frenet(stats_time,end_time):
    read_path = 'dataset/'+str(stats_time)+'-'+str(end_time)+'s_combined_frenet_down.csv'
    data = pd.read_csv(read_path)
    # 检查每个vehicleID是否有重复的time(s)
    duplicates = data[data.duplicated(subset=['vehicleID', 'time(s)'], keep=False)]
    duplicate_times = duplicates.groupby('vehicleID')['time(s)'].apply(list).reset_index()
    print("Vehicle IDs with duplicate time(s) and their duplicate times:")
    print(duplicate_times)
    # 获取有重复time(s)的vehicleID列表
    duplicate_vehicle_ids = duplicates['vehicleID'].unique()
    # 打印有重复time(s)的vehicleID
    print("Vehicle IDs with duplicate time(s):", duplicate_vehicle_ids)
    # 删除有重复time(s)的vehicleID的所有行
    cleaned_data = data[~data['vehicleID'].isin(duplicate_vehicle_ids)]
    # 保存清理后的数据到新的CSV文件
    save_path = 'dataset/'+str(stats_time)+'-'+str(end_time)+'s_cleaned_frenet_down.csv'
    cleaned_data.to_csv(save_path, index=False)
    print(f"Number of vehicleIDs with duplicate time(s): {len(duplicate_vehicle_ids)}")

#根据120-240s_cleaned_frenet_down.csv数据
# 计算出每个车辆在经过150m监测点时，通信范围分别为20m、40m、60m、80m、100m，通信范围内车辆数量
def Calculate_communication_counts(stats_time,end_time):
    read_path = 'dataset/' + str(stats_time) + '-' + str(end_time) + 's_cleaned_frenet_down.csv'
    data = pd.read_csv(read_path)

    # 定义通信范围
    ranges = [20, 40, 60, 80, 100]

    # 初始化结果 DataFrame
    result_columns = ['vehicleID', 'time(s)', 'longitudinalDistance(m)'] + [f'count_{r}m' for r in ranges]
    result_df = pd.DataFrame(columns=result_columns)

    # 过滤出每辆车在 longitudinalDistance(m) >= 100m 且 <= 150m 的瞬间
    vehicles_over_150m = data[(data['longitudinalDistance(m)'] >= 100) & (data['longitudinalDistance(m)'] <= 150)]
    # 只记录每个 vehicleID 达到条件的第一个时间点
    first_occurrences = vehicles_over_150m.groupby('vehicleID').first().reset_index()

    # 计算每个时间点不同通信范围内的车辆数量
    for index, row in tqdm(first_occurrences.iterrows(), desc='进度', unit='单位'):
        vehicle_id = row['vehicleID']
        time = row['time(s)']
        focal_position = row['longitudinalDistance(m)']

        counts = []
        for r in ranges:
            count = data[(data['time(s)'] == time) &
                         (data['vehicleID'] != vehicle_id)].apply(
                lambda x: abs(x['longitudinalDistance(m)'] - focal_position) <= r, axis=1).sum()
            counts.append(count)

        result_row = [vehicle_id, time, row['longitudinalDistance(m)']] + counts
        result_df.loc[len(result_df)] = result_row

    # 保存结果到新的CSV文件
    save_path = 'dataset/' + str(stats_time) + '-' + str(end_time) + 's_vehicle_communication_counts.csv'
    result_df.to_csv(save_path, index=False)

#对通信范围内车辆数量进行建模,统计模型来拟合数据
def Model_communication_counts(stats_time,end_time):
    # 加载数据
    read_path = 'dataset/' + str(stats_time) + '-' + str(end_time) + 's_vehicle_communication_counts.csv'
    data = pd.read_csv(read_path)

    # 定义通信范围
    ranges = [20, 40, 60, 80, 100]
    # 初始化结果 DataFrame
    result_columns = [f'count_{r}m' for r in ranges]

    df = data[result_columns]

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 绘制箱线图
    sns.boxplot(data=df)

    # 设置图形标题和标签
    plt.xlabel('Communication range(m)')
    plt.ylabel('Number of vehicles')

    # 保存图像到与表格相同的路径
    save_path = 'dataset/' + str(stats_time) + '-' + str(end_time) + 's_vehicle_communication_histograms.png'
    plt.savefig(save_path, dpi=1000)

    # 显示图像
    plt.show()
#当车辆id首次经过longitudinalDistance(m)>=100m时，
# 计算车辆范围(100m)与其他车辆与当前车辆之间持续时长。
# 当车辆第一次经过100m时，范围内接触持续时长在大于[0,1/30s]的车辆个数，(1/30,0.1s]的车辆个数，(0.1,1s]的车辆个数，(1,10s]的车辆个数，大于10s长的车辆个数。
def Calculate_communication_duration(stats_time,end_time):
    # 加载数据
    read_path = 'dataset/' + str(stats_time) + '-' + str(end_time) + 's_cleaned_frenet_down.csv'
    data = pd.read_csv(read_path)
    # 定义接触距离阈值
    contact_distance_threshold = 100
    # 定义时间区间
    time_intervals = [0, 1 / 30, 0.1, 1, 10]

    # 初始化统计字典
    duration_counts = {interval: 0 for interval in time_intervals}
    # 初始化结果DataFrame
    result_columns = ['vehicleID', 'time(s)', 'contact_0_1/30s', 'contact_1/30_0.1s', 'contact_0.1_1s', 'contact_1_10s',
                      'contact_10s']
    result_df = pd.DataFrame(columns=result_columns)

    # 过滤出每辆车在 longitudinalDistance(m) >= 100m 的瞬间
    vehicles_over_100m = data[(data['longitudinalDistance(m)'] >= 100) & (data['longitudinalDistance(m)'] <= 101)]
    # 只记录每个 vehicleID 达到条件的第一个时间点
    first_occurrences = vehicles_over_100m.groupby('vehicleID').first().reset_index()
    print("检测车辆数量为：",len(first_occurrences))
    for index, row in tqdm(first_occurrences.iterrows(),desc='进度',unit='单位'):
        vehicle_id = row['vehicleID']
        time = row['time(s)']
        current_distance = row['longitudinalDistance(m)']

        # 初始化接触时长统计
        duration_counts = {'contact_0_1/30s': 0, 'contact_1/30_0.1s': 0, 'contact_0.1_1s': 0, 'contact_1_10s': 0, 'contact_10s': 0}

        # 计算当前车辆与其他车辆的接触持续时间
        other_vehicles = data[(data['time(s)'] == time) & (data['vehicleID'] != vehicle_id)]
        for _, other_row in tqdm(other_vehicles.iterrows(),desc='子进度',unit='单位'):

            other_vehicle_id = other_row['vehicleID']
            other_distance = other_row['longitudinalDistance(m)']
            # 如果两车距离小于阈值，则计算为接触
            if abs(other_distance - current_distance) < contact_distance_threshold:
                contact_start_time = time
                contact_end_time = time
                next_time = time + 1 / 30

                while next_time <= end_time:  # 根据你的数据范围调整
                    next_distances = data[(abs(next_time - data['time(s)']) < 0.033) & (data['vehicleID'] == other_vehicle_id)]
                    next_current_distance = data[(abs(next_time - data['time(s)']) < 0.033) & (data['vehicleID'] == vehicle_id)]
                    if (len(next_distances) > 0) & (len(next_current_distance)>0):
                        next_distance = next_distances.iloc[0]['longitudinalDistance(m)']
                        next_current_distance = next_current_distance.iloc[0]['longitudinalDistance(m)']
                        if abs(next_distance - next_current_distance) < contact_distance_threshold:
                            contact_end_time = next_time
                            next_time += 1/30
                        else:
                            break
                    else:
                        break

                # 计算接触持续时间
                contact_duration = contact_end_time - contact_start_time

                # 根据持续时间区间统计
                if 0 < contact_duration <= 1 / 30:
                    duration_counts['contact_0_1/30s'] += 1
                elif 1 / 30 < contact_duration <= 0.1:
                    duration_counts['contact_1/30_0.1s'] += 1
                elif 0.1 < contact_duration <= 1:
                    duration_counts['contact_0.1_1s'] += 1
                elif 1 < contact_duration <= 10:
                    duration_counts['contact_1_10s'] += 1
                elif contact_duration > 10:
                    duration_counts['contact_10s'] += 1

            # 添加结果到结果DataFrame
        result_df = result_df.append({
            'vehicleID': vehicle_id,
            'time(s)': time,
            'contact_0_1/30s': duration_counts['contact_0_1/30s'],
            'contact_1/30_0.1s': duration_counts['contact_1/30_0.1s'],
            'contact_0.1_1s': duration_counts['contact_0.1_1s'],
            'contact_1_10s': duration_counts['contact_1_10s'],
            'contact_10s': duration_counts['contact_10s']
        }, ignore_index=True)

    # 保存结果到 CSV 文件
    save_path = 'dataset/' + str(stats_time) + '-' + str(end_time) + 's_communication_duration_counts.csv'
    result_df.to_csv(save_path, index=False)
    print("结果已保存到"+save_path)

def Model_communication_duration(stats_time,end_time):

    # 加载数据
    read_path = 'dataset/' + str(stats_time) + '-' + str(end_time) + 's_communication_duration_counts.csv'
    data = pd.read_csv(read_path)

    # 描述数据
    print(data.describe())

    # 计算各时间段的平均接触数量
    average_contacts = data[
        ['contact_0_1/30s', 'contact_1/30_0.1s', 'contact_0.1_1s', 'contact_1_10s', 'contact_10s']].mean()
    print("各时间段的平均接触数量",average_contacts)
    # 计算各时间段的总接触数量
    total_contacts = data[
        ['contact_0_1/30s', 'contact_1/30_0.1s', 'contact_0.1_1s', 'contact_1_10s', 'contact_10s']].sum()
    print("各时间段的总接触数量", total_contacts)
    # 可视化接触时间分布
    fig, ax = plt.subplots(figsize=(16, 9))

    # 设置字体和颜色

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1

    # 绘制堆叠条形图
    data[['contact_0_1/30s', 'contact_1/30_0.1s', 'contact_0.1_1s', 'contact_1_10s', 'contact_10s']].plot(kind='bar',stacked=True,ax=ax)
    # 设置图像标签和标题
    ax.set_xlabel('Vehicle ID', fontsize=12)
    ax.set_ylabel('Number of Vehicles Contacted', fontsize=12)
    ax.legend(['0-0.033s', '0.033-0.1s', '0.1-1s', '1-10s', '>10s'], loc='upper right', fontsize=10)
    # ax.set_title('Vehicle Contact Time Distribution', fontsize=14)

    # 设置横坐标标签，选择性显示部分标签
    num_labels = len(data)
    step = max(1, num_labels // 20)  # 显示大约10个标签
    ax.set_xticks(range(0, num_labels, step))
    ax.set_xticklabels(data['vehicleID'][::step].astype(int), rotation=45, ha='right', fontsize=10)

    # 调整布局以使图像更加紧凑
    plt.tight_layout()

    # 保存图像到与表格相同的路径
    save_path = 'dataset/' + str(stats_time) + '-' + str(end_time) + 's_communication_duration_counts.png'
    plt.savefig(save_path, dpi=800)

    # 显示图像
    plt.show()


if __name__ == '__main__':
    stats_time = 120
    end_time = 360
    #1.合并数据集，截取120-240s数据
    # Combined_frenet_down(stats_time,end_time)
    #2.过滤time(s)有重复值的车辆数据
    # Cleaned_frenet(stats_time,end_time)
    #3.1计算通信范围内，车辆数量，每0.03s
    # Calculate_communication_counts(stats_time,end_time)
    #3.1.1使用统计模型来拟合数据，如正态分布、指数分布等
    # Model_communication_counts(stats_time,end_time)

    #3.2车辆与其他车辆在不同通信范围内的车辆数量
    # Calculate_communication_duration(stats_time,end_time)
    #3.2.1使用统计模型来拟合数据，如正态分布、指数分布等
    # Model_communication_duration(stats_time,end_time)
