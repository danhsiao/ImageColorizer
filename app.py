from flask import Flask,request,render_template,send_from_directory, url_for
import os
import numpy as np
import cv2

app = Flask(__name__)
os.makedirs('uploads',exist_ok=True)
os.makedirs('static',exist_ok=True)

@app.route('/', methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            colorized_image = colorize_image(filepath)
            output_filepath = os.path.join('static', filename)
            cv2.imwrite(output_filepath, colorized_image)
            return render_template('result.html', image_url=url_for('static', filename=filename))
    return render_template('upload.html')
def colorize_image(image_path):
    prototxt_path = 'models/colorization_deploy_v2.prototxt'
    model_path = 'models/colorization_release_v2.caffemodel'
    kernel_path = 'models/pts_in_hull.npy'

    net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2,313,1,1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313],2.606,dtype="float32")]

    bw_image = cv2.imread(image_path)
    normalized = bw_image.astype("float32")/255.0
    lab = cv2.cvtColor(normalized,cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab,(224,224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :,:,:].transpose((1,2,0))

    ab = cv2.resize(ab,(bw_image.shape[1],bw_image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:,:,np.newaxis],ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")
    return colorized

if __name__ == '__main__':
    app.run(debug=True,port=8000)
