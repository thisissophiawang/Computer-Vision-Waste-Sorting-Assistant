import gradio as gr
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Processing advice dictionary
suggestions = {
    0: "Recyclable Waste：Clean and sort recyclables like paper, plastic, glass, and metal. Use designated recycling bins and follow your community’s collection guidelines to ensure proper recycling.",
    1: "Hazardous Waste：Identify hazardous items like batteries and chemicals. Use designated collection points or scheduled events in your area to safely dispose of these materials, ensuring safety and environmental protection.",
    2: "Kitchen Waste：Start composting kitchen scraps. Use a compost bin or vermicomposting system to convert food waste into nutrient-rich soil, which can be used in home gardens or shared with community gardens.",
    3: "Other Waste：Use standard waste bins for non-recyclable, non-hazardous, and non-kitchen waste. Dispose of this waste according to local guidelines, ensuring bins are emptied regularly to maintain cleanliness and hygiene.",
    # Add more categories and suggestions
}

# Processing category dictionary
Garbage_classification = {
    0: "recyclable",
    1: "hazardous",
    2: "kitchen",
    3: "other",
    # Add more categories and suggestions
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


# Gradio interface
with gr.Blocks(
        css=".gradio-container { background-image: url('https://images.unsplash.com/photo-1577563908411-5077b6dc7624?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&dl=volodymyr-hryshchenko-V5vqWC9gyEU-unsplash.jpg&w=2400'); background-size: cover; }") as iface:
    gr.Markdown("##  Intelligent household waste detection and classification system")
    gr.Markdown(" Upload garbage pictures, the system will automatically classify and provide processing suggestions.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label=" Upload trash image ")  # Display frame1
        with gr.Column():
            image_output = gr.Image(type="numpy", label="Test Result")  # Display frame2

    with gr.Row():
        with gr.Column():
            bar_chart_output = gr.Image(label="Category Bar Chart")  # category bar chart display box
        with gr.Column():
            results_output = gr.Dataframe(headers=["Serial Number", "category", "position", "confidence"], label="Detection result table")  # Display frame3
        with gr.Column():
            detect_button = gr.Button("Object Detection")  #button2
            save_chart_button = gr.Button("Save distribution map")  #new button: save distribution map
            save_data_button = gr.Button("Save the data table")  # new button: save data table

    #add processing advice text window
    suggestions_output = gr.Textbox(label="Handling opinions", interactive=False)  #create a blank text window

    # Button click
    detect_button.click(classify_and_suggest, inputs=image_input,
                        outputs=[image_output, results_output, bar_chart_output, suggestions_output])  #Update output

    #add save distribution map function
    def save_bar_chart(bar_chart):
        cv2.imwrite('bar_chart.png', bar_chart)  # Save the histogram as a PNG file
        return "The distribution map has been saved successfully！"  #Return success message

    save_chart_button.click(save_bar_chart, inputs=bar_chart_output, outputs=None)  # 绑定保存分布图按钮

    # add save data table function
    def save_results_as_csv(results):
        import pandas as pd
        df = pd.DataFrame(results, columns=["Serial number", "Category", "Position", "Confidence"])
        df.to_csv('results.csv', index=False)  #Save as CSV file
        return "The data table has been saved successfully！"  #Return success message

    save_data_button.click(save_results_as_csv, inputs=results_output, outputs=None)  #Bind save data table button

    # 显示成功消息的弹窗 Display pop-up window for success message
    success_message_output = gr.Markdown("")  # 创建一个Markdown组件用于显示消息 create a Markdown component to display messages

    save_chart_button.click(save_bar_chart, inputs=bar_chart_output, outputs=success_message_output)  # 绑定保存分布图按钮 Bind the save distribution map button
    save_data_button.click(save_results_as_csv, inputs=results_output, outputs=success_message_output)  # 绑定保存数据表按钮 Bind the save data table button

if __name__ == "__main__":
    iface.launch()