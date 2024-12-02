import gradio as gr
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from PIL import Image
import matplotlib

matplotlib.use('Agg')

# 设置matplotlib的全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 处理建议字典 Processing advice dictionary
suggestions = {
    0: "Recyclable Waste：Clean and sort recyclables like paper, plastic, glass, and metal. Use designated recycling bins and follow your community’s collection guidelines to ensure proper recycling.",
    1: "Hazardous Waste：Identify hazardous items like batteries and chemicals. Use designated collection points or scheduled events in your area to safely dispose of these materials, ensuring safety and environmental protection.",
    2: "Kitchen Waste：Start composting kitchen scraps. Use a compost bin or vermicomposting system to convert food waste into nutrient-rich soil, which can be used in home gardens or shared with community gardens.",
    3: "Other Waste：Use standard waste bins for non-recyclable, non-hazardous, and non-kitchen waste. Dispose of this waste according to local guidelines, ensuring bins are emptied regularly to maintain cleanliness and hygiene.",
    # 添加更多类别和建议  Add more categories and suggestions
}

# 处理类别字典  Processing category dictionary
Garbage_classification = {
    0: "recyclable",
    1: "hazardous",
    2: "kitchen",
    3: "other",
    # 添加更多类别和建议 Add more categories and suggestions
}

# Loading the YOLO model
model = YOLO('best.pt')


def classify_and_suggest(image):
    # 使用 YOLO 进行预测 Use YOLO for prediction
    predictions = model.predict(source=image, imgsz=320, device='cpu', save=False)  # 保存带框的图像 Save the image with boxes

    results = []
    category_counts = {i: 0 for i in Garbage_classification.keys()}  # 初始化类别计数 Initialize category counts
    for pred in predictions:
        boxes = pred.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0].item())  # 获取类别 Get the category
                suggestion = Garbage_classification.get(cls, "Unknown category")  # 获取处理建议 Get the processing suggestion
                # 将结果存储为元组格式，位置坐标取整 Store the result in tuple format, and round the position coordinates
                results.append((
                    len(results) + 1,  # 序号 Serial number
                    suggestion,  # 使用建议替代类别 Use suggestions instead of categories
                    [int(coord) for coord in box.xyxy[0].tolist()],  # 获取框的位置并取整   Get the position of the box and round it
                    round(box.conf[0].item(), 2)  # 获取置信度并保留两位小数    Get confidence and keep two decimal places
                ))

                category_counts[cls] += 1  # 更新类别计数 Update category count

                # 在图像上绘制检测框 Draw detection boxes on the image
                image = pred.plot(font_size=0.02)  # 使用 YOLO 自带的方式绘制框 Draw boxes in the way provided by YOLO

    # 转换颜色空间（如果需要） Convert color space (if needed)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将 RGB 转换为 BGR Convert RGB to BGR

    # 绘制柱状图 Create a bar chart
    bar_chart = create_bar_chart(category_counts)  # 创建柱状图的函数 Function to create a bar chart

    # 创建处理意见文本 Create processing advice text
    detected_categories = [Garbage_classification[cls] for cls in category_counts if category_counts[cls] > 0]
    suggestions_text = "The types of garbage in the picture are: " + ", ".join(detected_categories)  # 创建处理意见文本 Create processing advice text
    # 添加处理建议 Add processing advice
    for cls in category_counts:
        if category_counts[cls] > 0:
            suggestions_text += f"\n{suggestions[cls]}"  # 从第二行开始添加处理建议 Add processing advice from the second line

    # 返回带框的图像、结果列表、柱状图和处理意见 Return the image with boxes, result list, bar chart, and processing advice
    return image, results, bar_chart, suggestions_text  # 返回图像、结果、柱状图和处理意见 Return image, result, bar chart, and processing advice

def create_bar_chart(category_counts):
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    plt.figure()  # 创建新的图形 Create a new figure
    # 使用不同颜色绘制柱状图 Draw bar chart with different colors
    colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99']  # 定义颜色列表 Define color list
    plt.bar(categories, counts, color=colors[:len(categories)])  # 根据类别数量选择颜色 Choose color according to category count
    plt.xlabel('class')
    plt.ylabel('num')
    plt.title('Number of each category')
    plt.xticks(categories, [Garbage_classification[i] for i in categories])
    plt.tight_layout()

    # 将图形保存到内存中并返回 Save the figure to memory and return
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # 关闭图形 Close the figure

    # 将字节数据转换为 NumPy 数组 Convert byte data to NumPy array
    img_array = np.frombuffer(buf.getvalue(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 解码为图像 Decode to image
    return img  # 返回图像数据 Return image data

def analyze_waste_data(file):
    """
    分析废弃物数据并生成可视化图表 analysis waste data and generate visualization charts

    参数 Parameters:
        file: 上传的CSV文件对象 uploaded CSV file object

    返回 return:
        list: a list containing 5 PIL image objects:
            1. 每月各类废弃物堆叠柱状图 monthly total tonnes by waste type
            2. 分类准确率月度趋势图 classification accuracy monthly trend
            3. 回收率月度趋势图 recycling rate monthly trend
            4. 废弃物类型总量分布饼图 total tonnes by waste type pie chart
            5. 各类废弃物的分类准确率和回收率对比图 comparison of classification accuracy and recycling rate by waste type
    """
    try:
        with plt.style.context('default'):
            plt.close('all')  # close all existing plots to avoid memory leaks

            # read the CSV file data
            df = pd.read_csv(file.name)
            images = []  # store all generated plots

            ###################
            # Plot1: monthly stacked bar chart
            ###################

            # group by year, month, waste type and calculate total
            monthly_data = df.groupby(['Year', 'Month', 'Waste_Type']).agg({
                'Monthly_Total_Tonnes': 'sum'
            }).reset_index()

            # convert data to pivot table format for stacked bar chart
            pivot_data = monthly_data.pivot_table(
                index='Month',
                columns='Waste_Type',
                values='Monthly_Total_Tonnes',
                fill_value=0
            )

            # create plot and set style
            fig1 = plt.figure(figsize=(10, 5))
            ax1 = fig1.add_subplot(111)
            #  define color scheme to ensure different waste types have different colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # plot stacked bar chart
            bars = pivot_data.plot(kind='bar', stacked=True, ax=ax1, color=colors)

            # add data labels to each bar
            for bar in bars.containers:
                ax1.bar_label(bar, label_type='center')

            # set plot title and axis labels
            ax1.set_title('Monthly Total Tonnes by Waste Type')
            ax1.set_ylabel('Total Tonnes')
            ax1.set_xlabel('Month')

            # set x-axis tick labels (01-12 months)
            month_labels = [f"{month:02d}" for month in range(1, 13)]
            ax1.set_xticks(range(len(month_labels)))
            ax1.set_xticklabels(month_labels, rotation=45)

            # add total amount line
            total_monthly_tonnes = pivot_data.sum(axis=1)
            ax1.plot(range(len(total_monthly_tonnes)), total_monthly_tonnes.values,
                     color='black', marker='o', linewidth=2, label='Total')

            # place legend outside the plot to the right
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # save plot
            plt.tight_layout()
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', bbox_inches='tight', dpi=100)
            buf1.seek(0)
            plt.close(fig1)
            images.append(Image.open(buf1))

            ###################
            #Plot2: classification accuracy line chart
            ###################

            # calculate average classification accuracy per month
            monthly_accuracy = df.groupby(['Year', 'Month']).agg({
                'Classification_Accuracy': 'mean'
            }).reset_index()
            # convert percentage to decimal
            monthly_accuracy['Classification_Accuracy'] /= 100

            # create plot
            fig2 = plt.figure(figsize=(10, 5))
            ax2 = fig2.add_subplot(111)
            # plot line chart
            ax2.plot(range(len(monthly_accuracy)), monthly_accuracy['Classification_Accuracy'],
                     marker='o', color='lightgreen', linewidth=2)
            ax2.set_title('Average Classification Accuracy')
            ax2.set_ylabel('Accuracy')

            # add data labels
            for i, value in enumerate(monthly_accuracy['Classification_Accuracy']):
                ax2.annotate(f'{value:.2%}',
                             xy=(i, value),
                             xytext=(0, 5),
                             textcoords='offset points',
                             ha='center')

            # set x-axis labels
            ax2.set_xticks(range(len(monthly_accuracy)))
            ax2.set_xticklabels(monthly_accuracy['Month'], rotation=45)

            # save plot
            plt.tight_layout()
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', bbox_inches='tight', dpi=100)
            buf2.seek(0)
            plt.close(fig2)
            images.append(Image.open(buf2))

            ###################
            # Plot3：recycling rate line chart
            ###################

            # calculate average recycling rate per month
            monthly_recycling = df.groupby(['Year', 'Month']).agg({
                'Recycling_Rate': 'mean'
            }).reset_index()
            # convert percentage to decimal
            monthly_recycling['Recycling_Rate'] /= 100

            # create plot
            fig3 = plt.figure(figsize=(10, 5))
            ax3 = fig3.add_subplot(111)
            # plot line chart
            ax3.plot(range(len(monthly_recycling)), monthly_recycling['Recycling_Rate'],
                     marker='o', color='salmon', linewidth=2)
            ax3.set_title('Average Recycling Rate')
            ax3.set_ylabel('Recycling Rate')

            # add data labels
            for i, value in enumerate(monthly_recycling['Recycling_Rate']):
                ax3.annotate(f'{value:.2%}',
                             xy=(i, value),
                             xytext=(0, 5),
                             textcoords='offset points',
                             ha='center')

            # set x-axis labels
            ax3.set_xticks(range(len(monthly_recycling)))
            ax3.set_xticklabels(monthly_recycling['Month'], rotation=45)

            # save plot
            plt.tight_layout()
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format='png', bbox_inches='tight', dpi=100)
            buf3.seek(0)
            plt.close(fig3)
            images.append(Image.open(buf3))

            ###################
            # plot4: pie chart of total tonnes by waste type
            ###################

            # calculate total tonnes by waste type
            waste_type_total = df.groupby('Waste_Type').agg({
                'Monthly_Total_Tonnes': 'sum'
            }).reset_index()

            # calculate yearly average accuracy and recycling rate by waste type
            yearly_accuracy = df.groupby('Waste_Type').agg({
                'Classification_Accuracy': 'mean',
                'Recycling_Rate': 'mean'
            }).reset_index()

            # create pie chart
            fig4 = plt.figure(figsize=(6, 6))
            ax4 = fig4.add_subplot(111)
            ax4.pie(waste_type_total['Monthly_Total_Tonnes'],
                    labels=waste_type_total['Waste_Type'],
                    autopct='%1.1f%%')
            ax4.set_title('Total Tonnes by Waste Type')

            # save plot
            plt.tight_layout()
            buf4 = io.BytesIO()
            fig4.savefig(buf4, format='png', bbox_inches='tight', dpi=100)
            buf4.seek(0)
            plt.close(fig4)
            images.append(Image.open(buf4))

            ###################
            # Plot5: comparison of classification accuracy and recycling rate by waste type
            ###################

            # create plot
            fig5 = plt.figure(figsize=(12, 6))
            ax5_1 = fig5.add_subplot(111)
            ax5_2 = ax5_1.twinx()  # create a twin Y-axis

            # Set bar chart parameters
            x = np.arange(len(yearly_accuracy['Waste_Type']))
            width = 0.35  # bar width

            # draw classification accuracy bar chart (left Y-axis)
            bars1 = ax5_1.bar(x - width / 2, yearly_accuracy['Classification_Accuracy'] / 100,
                              width, label='Classification Accuracy', color='lightblue')
            # draw recycling rate bar chart (right Y-axis)
            bars2 = ax5_2.bar(x + width / 2, yearly_accuracy['Recycling_Rate'] / 100,
                              width, label='Recycling Rate', color='salmon')

            # Set plot title and axis labels
            ax5_1.set_title('Average Classification Accuracy and Recycling Rate by Waste Type')
            ax5_1.set_ylabel('Classification Accuracy')
            ax5_2.set_ylabel('Recycling Rate')

            # Set x-axis ticks and labels
            ax5_1.set_xticks(x)
            ax5_1.set_xticklabels(yearly_accuracy['Waste_Type'], rotation=45)

            # add a legend
            ax5_1.legend(loc='upper left')
            ax5_2.legend(loc='upper right')

            # define a function to add data labels
            def autolabel(bars, ax):
               # Add data labels to bar chart
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2%}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')

            # add data labels to both bar charts
            autolabel(bars1, ax5_1)
            autolabel(bars2, ax5_2)

            # set Y-axis limits to leave space for data labels
            ax5_1.set_ylim(0, max(yearly_accuracy['Classification_Accuracy'] / 100) * 1.2)
            ax5_2.set_ylim(0, max(yearly_accuracy['Recycling_Rate'] / 100) * 1.2)

            # save plot
            plt.tight_layout()
            buf5 = io.BytesIO()
            fig5.savefig(buf5, format='png', bbox_inches='tight', dpi=100)
            buf5.seek(0)
            plt.close(fig5)
            images.append(Image.open(buf5))

            return images

    except Exception as e:
        print(f"Error in analyze_waste_data: {str(e)}")
        return None
    finally:
        # save and close all plots to avoid memory leaks
        plt.close('all')


def analyze_single_category(file, category):
    """
    分析单个废弃物类别的详细数据并生成可视化图表 analysis detailed data for a single waste category and generate visualization charts

    参数:
        file: 上传的CSV文件对象 uploaded CSV file object
        category: waste category to analyze 要分析的废弃物类别 ('recyclable', 'hazardous', 'kitchen', 'other')

    返回:
        list: 包含4个PIL图像对象的列表，分别是(a list containing 4 PIL image objects):
            1. 月度总量柱状图 monthly total tonnes bar chart
            2. 月度分布饼图 monthly distribution pie chart
            3. 分类准确率趋势图 classification accuracy trend chart
            4. 回收率趋势图 recycling rate trend chart
    """
    # return empty results if no file
    if not file:
        return [None] * 4

    # read the CSV file data
    df = pd.read_csv(file.name)

    # initialize dictionary to store all category images
    category_images = {}

    # waste category mapping(short name to full name)
    category_mapping = {
        "recyclable": "Recyclable Waste",
        "hazardous": "Hazardous Waste",
        "kitchen": "Kitchen Waste",
        "other": "Other Waste"
    }

    #  all possible categories
    all_categories = ["recyclable", "hazardous", "kitchen", "other"]

    # create a dataframe with all months (1-12)
    all_months = pd.DataFrame({'Month': range(1, 13)})

    # generate charts for each category
    for waste_type in all_categories:
        images = []  #store all generated plots for the current category
        actual_waste_type = category_mapping[waste_type]  # get the actual waste type name
        category_data = df[df['Waste_Type'] == actual_waste_type]  #filter data for the selected category

        ###################
        # plot1: monthly total tonnes bar chart
        ###################

        # calculate total monthly tonnes
        monthly_tonnes = category_data.groupby('Month')['Monthly_Total_Tonnes'].sum().reset_index()
        # fill missing months with 0
        monthly_tonnes = pd.merge(all_months, monthly_tonnes, on='Month', how='left').fillna(0)

        # create bar chart
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.bar(monthly_tonnes['Month'], monthly_tonnes['Monthly_Total_Tonnes'], color='orange')
        ax1.set_title(f'Monthly Total Tonnes - {waste_type}')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Total Tonnes')
        ax1.set_xticks(range(1, 13))

        # save plot
        plt.tight_layout()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        plt.close(fig1)
        images.append(Image.open(buf1))

        ###################
         # Plot2: monthly distribution pie chart
        ###################

        #  create pie chart
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        pie_data = monthly_tonnes['Monthly_Total_Tonnes']

        # only plot pie chart if there is data
        if pie_data.sum() > 0:
            ax2.pie(pie_data, labels=[f'Month {i}' for i in range(1, 13)], autopct='%1.1f%%')
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')

        ax2.set_title(f'Monthly Distribution - {waste_type}')

        #save plot
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        plt.close(fig2)
        images.append(Image.open(buf2))

        ###################
        # Plot3: classification accuracy trend chart
        ###################

        # calculate average classification accuracy per month
        monthly_accuracy = category_data.groupby('Month')['Classification_Accuracy'].mean().reset_index()
        # ensure all months are displayed, fill missing months with 0
        monthly_accuracy = pd.merge(all_months, monthly_accuracy, on='Month', how='left').fillna(0)

        # create line chart
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(monthly_accuracy['Month'], monthly_accuracy['Classification_Accuracy'] / 100,
                 marker='o', color='lightgreen', linewidth=2)
        ax3.set_title(f'Classification Accuracy - {waste_type}')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(range(1, 13))

        # add data labels (only add labels for non-zero values)
        for i, value in enumerate(monthly_accuracy['Classification_Accuracy']):
            if value > 0:
                ax3.annotate(f'{value / 100:.2%}',
                             xy=(monthly_accuracy['Month'].iloc[i], value / 100),
                             xytext=(0, 5),
                             textcoords='offset points',
                             ha='center')

        # save plot
        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png')
        buf3.seek(0)
        plt.close(fig3)
        images.append(Image.open(buf3))

        ###################
         #plot4: recycling rate trend chart
        ###################

        # calculate average recycling rate per month
        monthly_recycling = category_data.groupby('Month')['Recycling_Rate'].mean().reset_index()
        # ensure all months are displayed, fill missing months with 0
        monthly_recycling = pd.merge(all_months, monthly_recycling, on='Month', how='left').fillna(0)

        # create line chart
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(monthly_recycling['Month'], monthly_recycling['Recycling_Rate'] / 100,
                 marker='o', color='salmon', linewidth=2)
        ax4.set_title(f'Recycling Rate - {waste_type}')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Recycling Rate')
        ax4.set_xticks(range(1, 13))

        # add data labels (only add labels for non-zero values)
        for i, value in enumerate(monthly_recycling['Recycling_Rate']):
            if value > 0:
                ax4.annotate(f'{value / 100:.2%}',
                             xy=(monthly_recycling['Month'].iloc[i], value / 100),
                             xytext=(0, 5),
                             textcoords='offset points',
                             ha='center')

        # save plot
        plt.tight_layout()
        buf4 = io.BytesIO()
        plt.savefig(buf4, format='png')
        buf4.seek(0)
        plt.close(fig4)
        images.append(Image.open(buf4))

        # store all images for the current category in the dictionary
        category_images[waste_type] = images

    # return images for the selected category
    return category_images[category]


def create_garbage_interface():
    """创建垃圾识别界面"""
    with gr.Column() as container:
        gr.Markdown("##  Intelligent household waste detection and classification system")
        gr.Markdown(
            " Upload garbage pictures, the system will automatically classify and provide processing suggestions.")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label=" Upload trash image ")  # Display frame1
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Test Result")  # Display frame2

        with gr.Row():
            with gr.Column():
                bar_chart_output = gr.Image(label="Category Bar Chart")  # category bar chart display box
            with gr.Column():
                results_output = gr.Dataframe(headers=["Serial Number", "category", "position", "confidence"],
                                              label="Detection result table")  # Display frame3
            with gr.Column():
                detect_button = gr.Button("Object Detection")  # button2
                save_chart_button = gr.Button("Save distribution map")  # new button: save distribution map
                save_data_button = gr.Button("Save the data table")  # new button: save data table

        # add processing advice text window
        suggestions_output = gr.Textbox(label="Handling opinions", interactive=False)  # create a blank text window

        # Button click
        detect_button.click(classify_and_suggest, inputs=image_input,
                            outputs=[image_output, results_output, bar_chart_output,
                                     suggestions_output])  # Update output

        # add save distribution map function
        def save_bar_chart(bar_chart):
            cv2.imwrite('bar_chart.png', bar_chart)  # Save the histogram as a PNG file
            return "The distribution map has been saved successfully！"  # Return success message

        save_chart_button.click(save_bar_chart, inputs=bar_chart_output, outputs=None)  # 绑定保存分布图按钮

        # add save data table function
        def save_results_as_csv(results):
            import pandas as pd
            df = pd.DataFrame(results, columns=["Serial number", "Category", "Position", "Confidence"])
            df.to_csv('results.csv', index=False)  # Save as CSV file
            return "The data table has been saved successfully！"  # Return success message

        save_data_button.click(save_results_as_csv, inputs=results_output, outputs=None)  # Bind save data table button

        # 显示成功消息的弹窗 Display pop-up window for success message
        success_message_output = gr.Markdown("")  # 创建一个Markdown组件用于显示消息 create a Markdown component to display messages

        save_chart_button.click(save_bar_chart, inputs=bar_chart_output,
                                outputs=success_message_output)  # 绑定保存分布图按钮 Bind the save distribution map button
        save_data_button.click(save_results_as_csv, inputs=results_output,
                               outputs=success_message_output)  # 绑定保存数据表按钮 Bind the save data table button

        return container


def create_analysis_interface():
    """创建数据分析界面"""
    with gr.Column() as container:
        with gr.Tabs() as tabs:
            with gr.Tab("Overall Analysis"):
                gr.Markdown("## Waste Classification Analysis Visualization")
                gr.Markdown(
                    "Upload include Year、Month、Monthly_Total_Tonnes、Classification_Accuracy、Recycling_Rate and Waste_Type's CSV file for analysis")

                with gr.Row():
                    file_input = gr.File(label="Upload CSV file")
                    analyze_button = gr.Button("Analysis")

                with gr.Row():
                    output1 = gr.Image(type="pil", label="Each month's total tonnage")

                with gr.Row():
                    output2 = gr.Image(type="pil", label="Classification accuracy")
                    output3 = gr.Image(type="pil", label=" Recycle Rate")

                with gr.Row():
                    pie_output = gr.Image(type="pil", label="Each year's Monthly_Total_Tonnes pie chart")
                    bar_output = gr.Image(type="pil", label="Classification accuracy and recycling rate bar chart")

                analyze_button.click(fn=analyze_waste_data, inputs=file_input,
                                     outputs=[output1, output2, output3, pie_output, bar_output])

            with gr.Tab("Single Category Analysis"):
                with gr.Row():
                    category_select = gr.Radio(
                        choices=["recyclable", "hazardous", "kitchen", "other"],
                        label=" Choose Waste Category",
                        value="recyclable",  # set default value
                        type="value"
                    )

                with gr.Row():
                    with gr.Column():
                        output_single1 = gr.Image(type="pil", label=" each month's total tonnage")
                    with gr.Column():
                        output_single2 = gr.Image(type="pil", label=" each month's distribution")

                with gr.Row():
                    with gr.Column():
                        output_single3 = gr.Image(type="pil", label=" Classification accuracy")
                    with gr.Column():
                        output_single4 = gr.Image(type="pil", label=" recyle rate")

                # update single category analysis when category is changed
                category_select.change(
                    fn=analyze_single_category,
                    inputs=[file_input, category_select],
                    outputs=[output_single1, output_single2, output_single3, output_single4]
                )

                # update single category analysis when file is uploaded
                file_input.change(
                    fn=analyze_single_category,
                    inputs=[file_input, category_select],
                    outputs=[output_single1, output_single2, output_single3, output_single4]
                )

        return container


# 创建主界面
with gr.Blocks(
        css=".gradio-container { background-image: url('https://images.unsplash.com/photo-1577563908411-5077b6dc7624?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&dl=volodymyr-hryshchenko-V5vqWC9gyEU-unsplash.jpg&w=2400'); background-size: cover; }") as iface:
    gr.Markdown("# Waste Classification and Analysis System")

    with gr.Row():
        garbage_btn = gr.Button("Waste Identification")
        analysis_btn = gr.Button("Data analysis")

    garbage_interface = create_garbage_interface()
    analysis_interface = create_analysis_interface()

    # 默认显示垃圾识别界面
    analysis_interface.visible = False


    def switch_interface(show_garbage):
        return {
            garbage_interface: gr.update(visible=show_garbage),
            analysis_interface: gr.update(visible=not show_garbage)
        }


    garbage_btn.click(
        fn=lambda: switch_interface(True),
        inputs=[],
        outputs=[garbage_interface, analysis_interface]
    )

    analysis_btn.click(
        fn=lambda: switch_interface(False),
        inputs=[],
        outputs=[garbage_interface, analysis_interface]
    )

if __name__ == "__main__":
    iface.launch()