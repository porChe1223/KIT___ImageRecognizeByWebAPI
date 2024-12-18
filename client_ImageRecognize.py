import os
import base64
import requests

# サーバIPアドレス
server_url = 'http://127.0.0.1:60003/server_ImageRecognize'

# ベースディレクトリ
base_dir = '/home/porche1223/KIT___WebAPI画像認識システム/KIT___ImageRecognize_WebAPI/WebAPI_Dataset/'


while True:
     # モード選択
    mode = input('モードを選択してください (R: 画像認識, F: ファインチューニング, X: 終了)： ')
    if mode == 'X':
        break

    if mode == 'R':
        # 画像パスを指定
        image = input('画像パスを入力してください(X: 終了)： ')
        if image == 'X':
            break

        image_path = os.path.join(base_dir, image)

        with open(image_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # 送信データ
        post_data = {
            'mode': 'R',
            'image': image_base64
        }

    elif mode == 'F':
        # ファインチューニングモード
        dir_path = input('ディレクトリパスを入力してください(X: 終了)： ')
        if dir_path == 'X':
            break
        dir_path = os.path.join(base_dir, dir_path)

        images_base64 = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    images_base64.append(image_base64)

        # 送信データ
        post_data = {
            'mode': 'F',
            'images': images_base64
        }

    else:
        print('無効な選択です。もう一度選択してください。')
        continue


    # POSTリクエストの送信
    response = requests.post(server_url, json=post_data)

    if mode == 'R':
        print('検出結果： ', response.json['検出結果'])
        print('予測結果： ', response.json['分析結果'])
    
    elif mode == 'F':
        if response.status_code == 200:
            with open('YourModel.pt', 'wb') as f:
                f.write(response.content)
            print('ファインチューニングされたモデルがダウンロードされました: YourModel.pt')
        else:
            print('エラー: ', response.json().get('error', '不明なエラー'))

    print('\n')

    # 画像パス例
    # kicking_ball.jpeg
    # holding_ball.jpg
    # throwing_ball.jpg
    # person_chair.jpg
    # person_table_chair.jpg
    # standing_near_car.jpg
    # sitting_near_car.jpg
    # riding_car.jpg