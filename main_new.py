from flask import Flask, request, render_template, send_from_directory, jsonify,send_file
import sqlite3
from PIL import Image
from flask.wrappers import Request, Response
from Preprocessing import convert_to_image_tensor, invert_image
import torch
from Model import SiameseConvNet, distance_metric ,ContrastiveLoss
from io import BytesIO
from Denoise import Denoise
import random
import json
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from skimage import img_as_ubyte, io
from skimage import measure, morphology
import os

app = Flask(__name__, static_folder='./frontend/build/static',
            template_folder='./frontend/build')


def load_model():
    device = torch.device('cpu')
    model = SiameseConvNet().eval()
    model.load_state_dict(torch.load(
        'Models/model_large_epoch_20', map_location=device))
    return model


def connect_to_db():
    conn = sqlite3.connect('user_signatures1.db')
    return conn


def get_file_from_db(customer_id):
    cursor = connect_to_db().cursor()
    select_fname = """SELECT sign1 from signatures5 where customer_id = ?"""
    cursor.execute(select_fname, (customer_id,))
    item = cursor.fetchone()
    cursor.connection.commit()
    return item


def extract_signature(source_image):
    """Extract signature from an input image.
    Parameters
    ----------
    source_image : numpy ndarray
        The pinut image.
    Returns
    -------
    numpy ndarray
        An image with the extracted signatures5.
    """
    # read the input image
    npimg=cv2.fastNlMeansDenoisingColored(source_image, None, 10, 10, 7, 15)
    # npimg = source_image.astype(np.uint8)
    # npimg = np.frombuffer(source_image,dtype=np.uint8)
    
    # source_image.convertTo(img_chn,CV_16UC3, 256);
    print(npimg)
    img = npimg
    img=cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    img = cv2.resize(img, (900, 900))
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    plt.imsave('gasd.png',img)
    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    #image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

    '''
    # plot the connected components (for debugging)
    ax.imshow(image_label_overlay)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    '''

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area/counter)
    print("the_biggest_component: " + str(the_biggest_component))
    print("average: " + str(average))

    # experimental-based ratio calculation, modify it for your cases
    # a4_constant is used as a threshold value to remove connected pixels
    # are smaller than a4_constant for A4 size scanned documents
    a4_constant = (((average/84.0)*250.0)+100)*1.5
    print("a4_constant: " + str(a4_constant))

    # remove the connected pixels are smaller than a4_constant
    b = morphology.remove_small_objects(blobs_labels, a4_constant)

    #b = b.astype(np.unit8)
    # save the the pre-version which is the image is labelled with colors
    # as considering connected components
    plt.imsave('pre_version.png', b)

    # read the pre-version
    img = cv2.imread('pre_version.png', 0)
    # ensure binary
    img = cv2.threshold(img, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # save the the result
    cv2.imwrite("output.png", img)
    return img


def main():
    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS signatures5 (customer_id TEXT PRIMARY KEY,sign1 BLOB)"""
    cursor = connect_to_db().cursor()
    cursor.execute(CREATE_TABLE)
    cursor.connection.commit()
    # cid = """SELECT id FROM history ORDER BY id DESC LIMIT 1"""
    # cursor.execute(select_fname, (customer_id,))
    # item = cursor.fetchone()
    # cursor.connection.commit()
    # return item
    
    # DELETE_DATA = """DELETE FROM signatures5"""
    # cursor1 = connect_to_db().cursor()
    # cursor1.execute(DELETE_DATA)
    # cursor1.connection.commit()
    # For heroku, remove this line. We'll use gunicorn to run the app
    app.run()  # app.run(debug=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file1 = request.files['uploadedImage1']
    # file2 = request.files['uploadedImage2']
    # file3 = request.files['uploadedImage3']
    customer_id = request.form['customerID']
    print(customer_id)
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        query = """DELETE FROM signatures5 where customer_id=?"""
        cursor.execute(query, (customer_id,))
        cursor = conn.cursor()
        query = """INSERT INTO signatures5 VALUES(?,?)"""
        cursor.execute(query, (customer_id, file1.read(),
                               ))
        conn.commit()
        return jsonify({"error": False})
    except Exception as e:
        print(e)
        return jsonify({"error": True})


@app.route('/verify', methods=['POST'])
def verify():
    try:
        CREATE_TABLE = """CREATE TABLE IF NOT EXISTS signcout9 (id INTEGER PRIMARY KEY)"""
        cursor = connect_to_db().cursor()
        cursor.execute(CREATE_TABLE)
        cursor.connection.commit()
        cid = """SELECT id FROM signcout9 ORDER BY id DESC """
        item=cursor.execute(cid)
        mainid=1
        print("q1")
        if item :
            print("q2")
            item1 = cursor.fetchone()

            # cursor.connection.commit()
            if item1==None :
                mainid=1
                conn = connect_to_db()
                cursor = conn.cursor()
                query = """INSERT INTO signcout9 VALUES(?)"""
                cursor.execute(query, (1, 
                                    ))
                conn.commit()
                cursor.close()
            else :
                print("adadad")
                print(item1[0])
                mainid=int(item1[0])+1 
                print("added",mainid)
                cursor.close()
                conn = connect_to_db()
                cursor = conn.cursor()
                query = """INSERT INTO signcout9 VALUES(?)"""
                cursor.execute(query, (mainid, 
                                    ))
                conn.commit()
                cursor.close()
        # test = request.json
        # print(test)
        # return test['title']
        # customer_id = request.form['customerID']
        customer_id = mainid
        # file1 = request.files['uploadedImage1']
        # file2 = request.files['uploadedImage2']
        # file3 = request.files['uploadedImage3']
        # customer_id = request.form['customerID']
        print(request.form)
        # print("sad")
        # s1=request.files['newSignature']
        # print(type(s1))
        # npimg = np.frombuffer(s1, np.uint8)
        # result=cv2.fastNlMeansDenoisingColored(s1, None, 10, 10, 7, 15)
        # print(type(result))
        file1 = request.files['newSignature']
        file2 = request.files['questioned']
        input_image = Image.open(request.files['newSignature'])
        print("Asdsa")
        questioned = request.files['questioned']
        input_image1 = Image.open(request.files['questioned'])
        # conn = connect_to_db()
        # cursor = conn.cursor()
        print("1")
        # query = """INSERT INTO signatures5 VALUES(?,?)"""
        # cursor.execute(query, (customer_id, questioned.read()
        #                        ))
        # conn.commit()
        print(customer_id)
        # path=os.path.join
        print(os.path)
        # path = r'D:\signature\signature69'
        # directory = r'D:\signature\signature69'
        # img = cv2.imread(path) 
        # os.chdir(directory) 
        # print("Before saving")   
        # print(os.listdir(directory))   
        # filename = 'cat.jpg'
        # cv2.imwrite(filename, img) 
        # print("After saving")  
        # print(os.listdir(directory))
        input_image.save( 'aaa'+str(customer_id)+'.png')
        input_image1.save( 'bbb'+str(customer_id)+'.png')
        
        # plt.imsave('abc'+str(customer_id)+'.png', input_image)
        # plt.imsave('xyz'+str(customer_id)+'.png', input_image1)
        Denoise.test('aaa'+str(customer_id)+'.png','bbb'+str(customer_id)+'.png')
        input_image= Image.open('aaa'+str(customer_id)+'.png')
        input_image1= Image.open('bbb'+str(customer_id)+'.png')
        # img1 = extract_signature(request.files['newSignature'])
        # image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        # lower = np.array([90, 38, 0])
        # upper = np.array([145, 255, 255])
        # mask = cv2.inRange(image, lower, upper)
        # print(input_image)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # close = cv2.morphologyEx(
        #     opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # cnts = cv2.findContours(close, cv2.RETR_EXTERNAL,
        #                         cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # boxes = []
        # for c in cnts:
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     boxes.append([x, y, x+w, y+h])

        # boxes = np.asarray(boxes)
        # left = np.min(boxes[:, 0])
        # top = np.min(boxes[:, 1])
        # right = np.max(boxes[:, 2])
        # bottom = np.max(boxes[:, 3])

        # result[close == 0] = (255, 255, 255)
        # ROI = result[top:bottom, left:right].copy()
        # cv2.rectangle(result, (left, top), (right, bottom), (36, 255, 12), 2)

        # cv2.imshow('result', result)
        # cv2.imshow('ROI', ROI)
        # cv2.imshow('close', close)
        # cv2.imwrite('result.png', result)
        # cv2.imwrite('ROI.png', ROI)
        print(type(input_image))
        input_image_tensor = convert_to_image_tensor(
            invert_image(input_image)).view(1, 1, 220, 155)
        customer_sample_images = get_file_from_db(customer_id)
        # print(customer_sample_images)
        plt.imsave('don.png', input_image_tensor)
        # if not customer_sample_images:
        #     return jsonify({'error': "uyugy"})
        # anchor_images = [Image.open(BytesIO(x))
        #                  for x in customer_sample_images]
        anchor_image_tensors = convert_to_image_tensor(invert_image(input_image1)).view(-1, 1, 220, 155)
                                # for x in anchor_images
        model = load_model()

        mindist = math.inf
        # for anci in anchor_image_tensors:
        f_A, f_X = model.forward(anchor_image_tensors, input_image_tensor)
        f1=model.forward_once(anchor_image_tensors).detach().numpy()
        f2=model.forward_once(input_image_tensor).detach().numpy()
            # plt.imshow(f1)
            # cv2.imshow('result.png', f1)
        plt.imsave('file1.png', f1)
        plt.imsave('file2.png', f1)
            # pq=ContrastiveLoss().forward(f1,f2,0.145139)
            # print("model1",f1)
            # print("model2",f2)
        out_arr = np.subtract(f1, f2) 
        plt.imsave('file3.png', out_arr)
        print ("Output array: ", out_arr)
            # print("bla",pq)
        dist = float(distance_metric(f_A, f_X).detach().numpy())
        mindist = min(mindist, dist)
        # return send_file(
        # 'aaa'+str(customer_id)+'.png',
        # as_attachment=True,
        # attachment_filename='aaa'+str(customer_id)+'.png',
        # mimetype='image/jpeg',
        # # abcd=jsonify({"match": True, "error": False, "threshold": "%.6f" % (0.145139), "distance": "%.6f" % (mindist)})
        # )
        # os.remove(  'aaa'+str(customer_id)+'.png')
        # os.remove(  'bbb'+str(customer_id)+'.png')
        matcheddis=mindist*100
        percen=100-matcheddis
        print(percen)
        if dist <= 0.145139:  # Threshold obtained using Test.py
            return jsonify({"match": True, "error": False, "threshold": "%.6f" % (0.145139), "distance": "%.6f" % (mindist),"difference":str(out_arr),'fileid':mainid,'percentage':percen})
        return jsonify({"match": False, "error": False, "threshold": 0.145139, "distance": round(mindist, 6),"difference":str(out_arr),'fileid':mainid,'percentage':percen})
    except Exception as e:
        print(e)
        return jsonify({"error": True, "adad": str(e)})
@app.route("/imagedenoise1", methods=['POST'])
def imagedenoise1():
      customer_id=request.form['id']
      return send_file(
        'aaa'+str(customer_id)+'.png',
        as_attachment=True,
        attachment_filename='aaa'+str(customer_id)+'.png',
        mimetype='image/jpeg',
        # abcd=jsonify({"match": True, "error": False, "threshold": "%.6f" % (0.145139), "distance": "%.6f" % (mindist)})
        )
@app.route("/imagedenoise2", methods=['POST'])
def imagedenoise2():
      customer_id=request.form['id']
      return send_file(
        'bbb'+str(customer_id)+'.png',
        as_attachment=True,
        attachment_filename='aaa'+str(customer_id)+'.png',
        mimetype='image/jpeg',
        # abcd=jsonify({"match": True, "error": False, "threshold": "%.6f" % (0.145139), "distance": "%.6f" % (mindist)})
        )
@app.route("/deletefile", methods=['POST'])
def deletef():
    customer_id=request.form['id']
    os.remove(  'aaa'+str(customer_id)+'.png')
    os.remove(  'bbb'+str(customer_id)+'.png')
    return "success"
@app.route("/manifest.json")
def manifest():
    return send_from_directory('./frontend/build', 'manifest.json')


@app.route("/favicon.ico")
def favicon():
    return send_from_directory('./frontend/build', 'favicon.ico')


if __name__ == '__main__':
    main()
