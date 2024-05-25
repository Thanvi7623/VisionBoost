from flask import Flask, render_template, request, redirect, flash
from Enhancement import SeeInDark as SID

app = Flask(__name__)
app.secret_key = 'your_secret_key'
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('Aboutus.html')
@app.route('/processImage', methods=['POST'])
def processImage():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            # Ensure that the image is in ARW format
            if img.filename.endswith('.ARW'):
                # Perform image enhancement on the ARW file directly

                # Serve the enhanced image as a response
                process = SID.enchanceImage(img)
                if process:
                    return render_template("processImage.html")
                else:
                    flash("Error: Image not processed", 'error')
                    return redirect(request.referrer)  # Redirect back to the previous page with the error message

    # Return an error message if something goes wrong
    flash("Error404: Image not processed", 'error')
    return redirect(request.referrer)  # Redirect back to the previous page with the error message

if __name__ == '__main__':
    app.run(debug=True)
