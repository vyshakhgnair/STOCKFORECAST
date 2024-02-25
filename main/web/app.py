from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import yfinance as yf
import requests
from Full_model import Full_model
import threading
import os


app = Flask(__name__, static_url_path='/static')
app.secret_key = "your_secret_key_here"

model_trained = False
trained_stock = None
pred_price = None
rmse_score = None
accu_score = None


def check_login(username, password):
    conn = sqlite3.connect('web\StockForecast.db')
    cur = conn.cursor()

    # Retrieve the hashed password for the given username
    cur.execute("SELECT PASSWORD FROM USERS WHERE USERNAME = ?", (username,))
    result = cur.fetchone()
    print(result)
    if result is not None:
        hashed_password = result[0]

        # Hash the password entered by the user and compare it with the stored hash
        if check_password_hash(hashed_password, password):
            print("Login successful!")
            return True
        else:
            print("Incorrect password. Please try again.")
            return False
    else:
        print("Username not found. Please check your username and try again.")
        return False

    conn.close()


# define the root route
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # check if user exists in database
        user = request.form.get('username')
        password = request.form.get('password')
        # perform necessary authentication and validation checks here
        if check_login(user, password):
            flash('Login successful')
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect username or password')
        return redirect(url_for('login'))

    # if method is GET, render the login page
    return render_template('login.html')


# define the dashboard route
@app.route('/dashboard')
def dashboard():
    # render the dashboard page
    return render_template('dashboard.html')


@app.errorhandler(requests.exceptions.HTTPError)
def handle_http_error(query):
    return render_template('error.html', error=query)


# define the register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    # connect to the database
    conn = sqlite3.connect('main\web\StockForecast.db')
    cur = conn.cursor()
    if request.method == 'POST':
        # perform necessary validation checks here
        name = request.form.get('name')
        user = request.form.get('username')
        password = request.form.get('password')
        cpassword = request.form.get('confirm_password')
        # perform necessary validation checks here
        if password != cpassword:
            flash('Passwords do not match')
            return redirect(url_for('register'))
        else:
            hashed_password = generate_password_hash(password)
            try:
                cur.execute("INSERT INTO USERS (Name, USERNAME, PASSWORD) VALUES (?, ?, ?)",
                            (name, user, hashed_password))
                conn.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                # The username is already taken
                return redirect(url_for('register'))

    # if method is GET, render the registration page
    return render_template('login.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['search']
    try:
        ticker = yf.Ticker(query).info["symbol"]
        stock = yf.Ticker(ticker)
        print(stock)
        return redirect(url_for('loader', stock=ticker))
    except ValueError:
        # handle the HTTP error using the error handler
        return handle_http_error(query)
    except requests.exceptions.HTTPError:
        # handle the HTTP error using the error handler
        return handle_http_error(query)


@app.route('/load/<stock>')
def loader(stock):
    global model_trained, trained_stock

    # Set the global variables
    model_trained = False
    trained_stock = stock

    # Train the model in a separate thread
    threading.Thread(target=train_model, args=(stock,)).start()

    return redirect(url_for('loading'))


@app.route('/result')
def result():
    global model_trained, trained_stock, pred_price, rmse_score

    if model_trained:
        # Reset the global variables
        model_trained = False
        stock = trained_stock
        trained_stock = None

        # Define the list of image files to display
        image_files = [
            '16_Day_Price_Prediction.png',
            'Chart_with_predictions.png',
            'loss.png',
            'tickerpic.png'
        ]

        # Create the list of image URLs to pass to the template
        image_urls = [url_for('static', filename=file) for file in image_files]

        return render_template('result.html', stock=stock, image_urls=image_urls, pred_price=pred_price,
                               rmse_score=rmse_score)
    else:
        return redirect(url_for('loading'))


def train_model(stock):
    # Perform the model training here
    global pred_price, rmse_score, accu_score

    pred_price, rmse_score = Full_model(stock)

    # Update the training status
    global model_trained
    model_trained = True


@app.route('/loading')
def loading():
    return render_template('loading.html')


@app.route('/status')
def status():
    global model_trained

    return jsonify({'model_trained': model_trained})


if __name__ == '__main__':
    app.run(debug=True)
