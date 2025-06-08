import cprs as RT_CPR # real time car plate recognition system
import os
from flask import Flask, request, redirect, url_for, make_response
from werkzeug.utils import secure_filename
from flask import send_from_directory
import database as db
UPLOAD_FOLDER = './upload/'
ALLOWED_EXTENSIONS = set(['mp4'])
records = db.Record("records.json")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(save_path)
            print(save_path)
            ret, plate_number = RT_CPR.start_predict(save_path)
            if ret :
                if db.search_plate(records, plate_number):
                    return make_response("Plate database search success", 200)
                    #return redirect(url_for('uploaded_file',filename=filename))
                else:
                    # Plate recognized but not found in database (404 Not Found)
                    return make_response("Plate database search failed", 400)
            else:
                # Plate recognition failure (400 Bad Request or custom code)
                response = make_response("Plate recognition predict failed", 400)
                return response
            
    return '''
    <!doctype html> 車輛門禁系統
    <title>Upload new inference file</title>
    <h1>Upload new inference File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
#對上傳檔案進行訪問
@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)   
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)