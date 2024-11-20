import requests

#サーバIPアドレス
server_url = 'http://127.0.0.1:60003/server_ImageRecognize'

# 送信データ
post_data = {
    'image': '/home/porche1223/3-3___WebAPI画像処理システム/ImageRecognizeByWebAPI/WebAPI_Dataset/children_playing_football.jpeg'
}
# POSTリクエストの送信
response = requests.post(server_url, json=post_data)

print("Status Code:", response.status_code)
print("Response Data:", response.json())