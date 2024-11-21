import requests

#サーバIPアドレス
server_url = 'http://127.0.0.1:60003/server_ImageRecognize'

# 送信データ
post_data = {
    'image': '/home/porche1223/3-3___WebAPI画像処理システム/ImageRecognizeByWebAPI/WebAPI_Dataset/person_chair.jpg'
}
# POSTリクエストの送信
response = requests.post(server_url, json=post_data).json()

print('検出結果： ', response['検出結果'])
print('予測結果： ', response['分析結果'])