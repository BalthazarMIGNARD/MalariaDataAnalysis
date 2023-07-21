from flask import Flask, request, render_template
import pandas as pd
import base64
from displaycorr1 import displayCorr1
from corr1 import corrIndex2  # Import the function for generating the scatter plot

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename.endswith('.xlsx'):
        # df = pd.read_excel(file)
        
        # Generate the scatter plot using the corrIndex2 function
        scatter_plot_data = corrIndex2(file)
        
        return render_template('index.html', scatter_plot=scatter_plot_data)
    else:
        return 'Invalid file format. Please upload an XLSX file.'

if __name__ == '__main__':
    app.run()
