from flask import Flask, render_template, request
import search_engine
import time
from search_engine import SearchEngine, InvertedIndex, DocumentProcessing, Document, NaiveBayesClassifier, ClassifierDataFrame, KNN
import sys

app = Flask(__name__)

results = {}
list_values = []
result_dict = {}
html_file = ""

@app.route("/")
def root():
    return render_template('home.html')

@app.route("/allResults", methods=['GET', 'POST'])
def allResults():
    global documents
    global results
    global list_values
    global result_dict
    global html_file
    if request.method == 'POST':
        user_input = request.form
        query = user_input["query"]
        type = user_input["search_type"]
        if type == 'bs':
            # starting_time = time.time()
            results, documents = search_engine.run("--bs", query)
        elif type == 'vsm':
            # starting_time = time.time()
            results, documents = search_engine.run("--vsm", query)
            # print("Time taken: {}".format(time.time() - starting_time))
        list_docid = []
        list_docid = results["all"]
        list_values = []
        if list_docid is None:
            list_values.append("No document found.")
        else:
            for l in list_docid:
                list_values.append(documents[l])
        html_file = "allResults.html"
    return render_template(html_file, result=list_values)

@app.route("/resultContent", methods=['GET', 'POST'])
def resultContent():
    if request.method == 'POST':
        user_input = dict(request.form)
        #print(user_input, file=sys.stderr)
        value = user_input["doc"]
        return render_template('resultContent.html', result=value)

@app.route("/businessResults")
def businessResults():
    global results
    global documents
    list_docid = []
    list_docid = results.get("business")
    list_values = []
    if list_docid is None:
        list_values.append("No document found.")
    else:
        for l in list_docid:
            list_values.append(documents[l])
    return render_template('businessResults.html', result=list_values)

@app.route("/entertainmentResults")
def entertainmentResults():
    global results
    global documents
    list_docid = []
    list_docid = results.get("entertainment")
    list_values = []
    if list_docid is None:
        list_values.append("No document found.")
    else:
        for l in list_docid:
            list_values.append(documents[l])
    return render_template('entertainmentResults.html', result=list_values)

@app.route("/politicsResults")
def politicsResults():
    global results
    global documents
    list_docid = []
    list_docid = results.get("politics")
    list_values = []
    if list_docid is None:
        list_values.append("No document found.")
    else:
        for l in list_docid:
            list_values.append(documents[l])
    return render_template('politicsResults.html', result=list_values)

@app.route("/sportResults")
def sportResults():
    global results
    global documents
    list_docid = []
    list_docid = results.get("sport")
    list_values = []
    if list_docid is None:
        list_values.append("No document found.")
    else:
        for l in list_docid:
            list_values.append(documents[l])
    return render_template('sportResults.html', result=list_values)

@app.route("/technologyResults")
def technologyResults():
    global results
    global documents
    list_docid = []
    list_docid = results.get("tech")
    list_values = []
    if list_docid is None:
        list_values.append("No document found.")
    else:
        for l in list_docid:
            list_values.append(documents[l])
    return render_template('technologyResults.html', result=list_values)

if __name__ == '__main__':
    app.run(debug=True)
