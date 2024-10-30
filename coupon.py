import requests
from PIL import Image
import io
import keras
from keras import ops, layers
import numpy as np 
import tensorflow as tf
from captcha import decode_batch_predictions
import sys
import warnings
import os 
warnings.filterwarnings('ignore')


def encode_single_sample(img_path, label):
    char_to_num = layers.StringLookup(vocabulary=list([chr(ord('0') + i) for i in range(10)]), mask_token=None)
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.adjust_contrast(img, 256)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.where(img < 1, 1., 0.)

    # 4. Resize to the desired size
    img = ops.image.resize(img, [40, 120])

    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = ops.transpose(img, axes=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


def preprocess(data, captchaId):
    img = Image.open(io.BytesIO(data))
    w,h= img.size
    px = img.load()
    visited = [[False for j in range(w)] for i in range(h)]
    dy=[1,-1,0,0]
    dx=[0,0,1,-1]
    def bfs(px, i, j, w, h):
        sz = 0
        q = [(i, j)]
        visited[i][j] = True
        checked = []
        while len(q) > 0:
            y,x = q.pop()
            checked.append((y, x))
            for d in range(4):
                ny = y + dy[d]
                nx = x + dx[d]
                if ny < 0 or nx < 0 or ny >= h or nx >= w: continue # out of range
                if px[nx, ny] == 0: continue # transparent
                if visited[ny][nx]: continue # already visited
                visited[ny][nx] = True
                q.append((ny, nx))
                sz += 1
        
        threshold = 36
        if sz <= threshold:
            return True, checked
        return False, None

    for i in range(h):
        for j in range(w):
            if visited[i][j]: continue
            if px[j, i] > 1: 
                need_replace, checked = bfs(px, i, j, w, h)
                if need_replace:
                    for (y, x) in checked:
                        img.putpixel((x, y), 0)
    img.save(f'{captchaId}.png')
    return img 


if __name__ == '__main__':
    
    sess = requests.session()

    generate = sess.post('https://mail.advrpg.com/api/v1/captcha/generate').json()
    captchaId = generate['data']['captchaId']

    captcha = sess.get(f'https://mail.advrpg.com/api/v1/captcha/image/{captchaId}').content

    captcha = preprocess(captcha, captchaId)

    pred = keras.models.load_model('./pred3.keras')

    inference_dataset = tf.data.Dataset.from_tensor_slices((np.array([f'./{captchaId}.png']), np.array(['1111'])))
    inference_dataset = (
        inference_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(1)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    pred = pred.predict(inference_dataset, verbose=0)
    user_id = os.environ['USER_ID']
    pred = decode_batch_predictions(pred)[0]
    data = {
        'captcha': pred, 
        'captchaId': captchaId, 
        'giftCode':  sys.argv[1], 
        'userId': user_id
    }

    print('req data:', data)
    resp = sess.post('https://mail.advrpg.com/api/v1/giftcode/claim', json=data).json()

    print('resp data:', resp)
    match resp['code']:
        case 0:
            print('Congratulations! Your rewards have been sent to your in-game Mailbox. Go and check it out!')
        case 20001: 
            print('Redeem failed; information incorrect')
        case 20002: 
            print('Redeem failed; incorrect Verification Code')
        case 20003: 
            print('Oh no, we suspect your ID is incorrect. Please check again.')
        case 20401: 
            print('Oh no, we suspect your Rewards Code is incorrect. Please check again.')
        case 20402: 
            print('Oh no, your Rewards Code has already been used!')
        case 20403: 
            print('Oh no, your Rewards Code has expired...')
        case 20404: 
            print('Redeem code is not activated, please try it later.')
        case 20409: 
            print('This redemption code has already been redeemed and can no longer be redeemed.')
        case 20407: 
            print('Codes of similar items can only be claimed once.')
        case 20410: 
            print('You are temporarily unable to redeem this gift code.')
        case 30001: 
            print('Server is busy, please try again.')
        case _:
            print(f'Server is busy, please try again. {resp["code"]}')


