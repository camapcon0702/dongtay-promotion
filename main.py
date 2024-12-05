import requests
import pandas as pd

# Giả sử API trả về dữ liệu JSON

# base_url = 'http://127.0.0.1:8080/'

# url_get_promotion_data = base_url + 'get_promotion_data'

# response = requests.get(url_get_promotion_data)

# if response.status_code == 200:
#     data = response.json()  # Chuyển JSON thành Python dictionary/list
#     df = pd.DataFrame(data)  # Chuyển thành DataFrame
#     print(df)
# else:
#     print(f"Lỗi khi gọi API: {response.status_code}")
    
# df['Published_date'] = pd.to_datetime(df['Published_date']).dt.date

# df

from panel import Column, Row
import holoviews as hv  # Import holoviews đúng cách
from holoviews import opts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hvplot.pandas
import panel as pn


# url_video_count = base_url + 'get_video_count'

# response_get_video_count = requests.get(url_video_count)
# if response.status_code == 200:
#     data = response_get_video_count.json()  # Chuyển JSON thành Python dictionary/list
#     video_count = pd.DataFrame(data)  # Chuyển thành DataFrame
#     video_count
# else:
#     print(f"Lỗi khi gọi API: {response.status_code}")

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Base URL của API
base_url = 'http://127.0.0.1:8080/'
url_get_promotion_data = base_url + 'get_promotion_data'
url_video_count = base_url + 'get_video_count'
url_keyword_count = base_url + 'get_keyword'
url_clustering = base_url + 'get_clustering'
url_classification = base_url + 'get_classification'

# Hàm để gọi API
def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Lỗi khi gọi API {url}: {response.status_code}")
        return None

# Chạy các yêu cầu API song song
with ThreadPoolExecutor() as executor:
    future_promotion_data = executor.submit(fetch_data, url_get_promotion_data)
    future_video_count = executor.submit(fetch_data, url_video_count)
    future_keyword_count = executor.submit(fetch_data, url_keyword_count)
    future_clustering = executor.submit(fetch_data, url_clustering)
    future_classification = executor.submit(fetch_data, url_classification)

# Lấy dữ liệu từ các yêu cầu
promotion_data = future_promotion_data.result()
video_count_data = future_video_count.result()
keyword_count_data = future_keyword_count.result()
clustering_data = future_clustering.result()
classification_data = future_classification.result()

# Xử lý dữ liệu sau khi lấy được
if promotion_data:
    df = pd.DataFrame(promotion_data)
    df['Published_date'] = pd.to_datetime(df['Published_date']).dt.date
    print(df)

if video_count_data:
    video_count = pd.DataFrame(video_count_data)
    print(video_count)

if keyword_count_data:
    keyword_count = pd.DataFrame(keyword_count_data)
    print(keyword_count)

if clustering_data:
    clustering = pd.DataFrame(clustering_data)
    print(clustering)
    
if classification_data:
    classification = pd.DataFrame(classification_data)
    print(classification)

# Kích hoạt holoviews
hv.extension('bokeh')

# Giả sử bạn đã có DataFrame df và video_count_per_day
# Tạo các biểu đồ sử dụng hvplot hoặc holoviews (hv.Curve)

# Biểu đồ Views
views_plot = df.hvplot(x='Published_date', y='Views', title='Views', color='blue', height=300, width=1400)

# Biểu đồ likes
likes_plot = df.hvplot(x='Published_date', y='Likes', title='Likes', color='orange', height=300, width=1400)

# Biểu đồ comments
comments_plot = df.hvplot(x='Published_date', y='Comments', title='Comments', color='red', height=300, width=1400)

# Biểu đồ video count per day
video_count_plot = video_count.hvplot(x='Keyword', y='Video Count', title='Video Count', color='green', height=300, width=1400)

# Đặt các biểu đồ vào layout
dashboard = pn.Column(views_plot, likes_plot, comments_plot)
# dashboard = pn.Column(views_plot)

keywords = video_count['Keyword']

print(keywords)

df['Year'] = pd.to_datetime(df['Published_date']).dt.year

data_per_year = {2016: [],2017: [],2018: [],2019: [],2020: [],2021: [],2022: [], 2023: [], 2024: []}

# Calculate the number of videos containing each keyword for each year
for year in data_per_year.keys():
    for keyword in keywords:
        # Count occurrences of the keyword in titles for the given year
        count = df[(df['Year'] == year) & (df['Title'].str.contains(keyword, case=False, na=False))].shape[0]
        data_per_year[year].append(count)

# Prepare data for radar chart
data = pd.DataFrame(data_per_year, index=keywords)

# We will now use hvplot to generate a radar chart-like plot
radar_chart = data.hvplot.line(
    title="Số Video theo Chương trình qua Các Năm", 
    xlabel="Tên chương trình", 
    ylabel="Số Lượng Video", 
    line_width=2, 
    width=1400, 
    height=600, 
    grid=True,
    color=['blue', 'green', 'red'],  # Different colors for each year
    legend='top',
    rot=45
)

# Display the chart
print(radar_chart)


word_freq = dict(zip(keyword_count['Keyword'], keyword_count['Video Count']))


from wordcloud import WordCloud

# Tạo Word Cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis"
).generate_from_frequencies(word_freq)

# Vẽ biểu đồ Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

# Lưu hình ảnh Word Cloud vào bộ nhớ
import io
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)

# Tạo dashboard Panel
wordcloud_pane = pn.pane.PNG(buf, width=800, height=400)
wc_dashboard = pn.Column(
    pn.pane.Markdown("# Word Cloud for Keywords"),
    wordcloud_pane
)



#Clustering

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# clustering['Like_to_View_Ratio'] = clustering['Likes'] / clustering['Views']

# data = data.dropna(subset=['Like_to_View_Ratio'])

# # Chuẩn hóa đặc trưng tỷ lệ Like/Views
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(data[['Like_to_View_Ratio']])

# # Áp dụng KMeans phân cụm
# kmeans = KMeans(n_clusters=3, random_state=42)
# data['Cluster'] = kmeans.fit_predict(scaled_features)

print(clustering)

# Vẽ biểu đồ phân cụm
scatter_plot = clustering.hvplot.scatter(
    x='Like_to_View_Ratio',
    y='Likes',
    by='Cluster',
    hover_cols=['Title'],  # Thêm cột tùy chọn để hiển thị thông tin
    title='KMeans Clustering of Videos based on Like-to-View Ratio',
    xlabel='Like/Views Ratio',
    ylabel='Likes',
    width=1400,
    height=600,
)

# Tạo ứng dụng Panel
# pn.extension()
dashboard_cluster = pn.Column(
    "# KMeans Clustering of Videos",
    pn.pane.Markdown("## Biểu đồ thể hiện mức độ tương tác của người xem"),
    scatter_plot,
)


classification_plot = classification.hvplot.scatter(
    x='Views', 
    y='Likes', 
    by='Target',  # Chia theo 'Target'
    legend='top', 
    title='Video Classification by Views and Likes',
    xlabel='Views', 
    ylabel='Likes', 
    height=600, 
    width=1400, 
    size=10, 
    alpha=0.8,
    colorbar=True,  # Thêm thanh màu nếu cần
)

# 3. Xây dựng giao diện Panel
cf_dashboard = pn.Column(
    "# Video Classification Dashboard",
    classification_plot,
)


top_10_videos = df.nlargest(10, 'Views').sort_values(by='Views', ascending=True)
print(top_10_videos)

top_10_plot = top_10_videos.hvplot.barh(
    x='Title', 
    y='Views', 
    title='Top 10 Video Có Lượt Xem Cao Nhất',
    xlabel='Video', 
    ylabel='Lượt Xem', 
    width=1400, 
    height=600, 
    color='orange',
    rot=0  # Không xoay tên trục x
)

# 3. Tạo giao diện Panel
top_10_dashboard = pn.Column(
    "# Dashboard: Top 10 Video Có Lượt Xem Cao Nhất",
    top_10_plot,
)

# Hiển thị dashboard
# dashboard.servable()
dashboard
sidebar = [
    pn.pane.Markdown("# YouTube Video Data Dashboard"),
    pn.pane.Markdown("### A dashboard displaying the statistics of video views, likes, comments, and video counts over time."),
    pn.pane.Markdown("## Settings"),
    # Slider for year or any other settings can go here, for example:
    # pn.widgets.IntSlider(name='Year', start=2015, end=2024, step=1, value=2024),
]

# Main content layout
main = [
    pn.Row(pn.Column(views_plot, margin=(0, 25))),
    pn.Row(pn.Column(likes_plot, margin=(0, 25))),
    pn.Row(pn.Column(comments_plot, margin=(0, 25))),
    pn.Row(pn.Column(video_count_plot, margin=(0, 25))),
    pn.Row(pn.Column(top_10_dashboard, margin=(0, 25))),
    pn.Row(pn.Column(radar_chart, margin=(0, 25))),
    pn.Row(pn.Column(wc_dashboard, margin=(0, 25))),
    pn.Row(pn.Column(dashboard_cluster, margin=(0, 25))),
    pn.Row(pn.Column(cf_dashboard, margin=(0, 25))),
]

# Create template with sidebar and main content
template = pn.template.FastListTemplate(
    title='YouTube Video Dashboard',
    sidebar=sidebar,
    main=main,
    accent_base_color="#88d8b0",
    header_background="#88d8b0"
)

# Show the template
template.servable()