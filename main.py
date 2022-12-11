from flask import Flask, render_template, redirect, Response, url_for
import cv2

from flask_bootstrap import Bootstrap
from flask_ckeditor import CKEditor

from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

from flask_login import   LoginManager,  logout_user
from forms import  RegisterForm, Login
from keras.models import load_model
import numpy as np
from tensorflow.keras.utils import img_to_array
camera = cv2.VideoCapture(0)
#face_classifier = cv2.CascadeClassifier(r'C:\Users\Lenovo\ml digi\facialrecog\haarcascade_frontalface_default.xml')
#classifier = load_model(r'C:\Users\Lenovo\ml digi\facialrecog\final_model (1).h5', compile=True)
app = Flask(__name__)


a=[]


def func():
    while True:
        success,frame=camera.read()

        #read the camera frame
        if not success:
            break
        else:
            face_classifier = cv2.CascadeClassifier(r'C:\Users\Lenovo\ml digi\facialrecog\haarcascade_frontalface_default.xml')
            classifier = load_model(r'C:\Users\Lenovo\ml digi\facialrecog\final_model (1).h5', compile=False)

            emotion_labels = ["anger","contempt","disgust","fear","Happy","neutrality","sadness","surprise"]
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(144,144),interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)
                    prediction = classifier.predict(roi)#[0]
                    label=emotion_labels[prediction.argmax()]
                    a.append(label)

                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            ret,buffer=cv2.imencode(".jpg",frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'
ckeditor = CKEditor(app)
Bootstrap(app)

login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


##CONNECT TO DB

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///new-books-collection.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


##CREATE TABLE
class User(db.Model):
    __tablename__ = "User"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100))

db.create_all()

# CREATE RECORD



@app.route('/')
def get_all_posts():
    return render_template("index.html")





# Register new users into the User database
@app.route('/register', methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hash_and_salted_password = generate_password_hash(form.Password.data)
        new_user = User(email=form.Email.data, name=form.Name.data, password=hash_and_salted_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html", form=form)



@app.route("/vedio")
def vedio():
    return Response(func(),mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/login_successful', methods=["GET", "POST"])
def login_successful():
    return render_template("loh.hyml.html", )


@app.route('/login', methods=["GET", "POST"])
def login():
    form = Login()
    if form.validate_on_submit():
        email = form.Email.data
        password = form.Password.data
        user = User.query.filter_by(email=email).first()

        # Email doesn't exist
        if not user or not check_password_hash(user.password, password):
            return redirect(url_for('login'))
        else:

            return redirect(url_for('login_successful'))
    return render_template("login.html", form=form)


@app.route('/logout')
def logout():
    logout_user()
    return render_template("Lout.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route('/song')
def song():
    camera.release()
    if a[-1] in [ "Happy", "neutrality", "surprise"]:
        return render_template("song.html")
    elif a[-1] in ["anger", "sadness"]:
        return render_template("sadsong.html")
    elif a[-1] in ["contempt", "disgust", "fear"]:
        return render_template("fearsong.html")



if __name__ == "__main__":
    app.run(port=5500, debug=True)

