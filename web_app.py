from flask import Flask, render_template, request
import search_engine
import time
from search_engine import SearchEngine, InvertedIndex, DocumentProcessing, Document, NaiveBayesClassifier, ClassifierDataFrame

app = Flask(__name__)

@app.route("/")
def root():
    return render_template('home.html')

@app.route("/results", methods=['GET', 'POST'])
def results():
    global documents
    if request.method == 'POST':
    #do something
        user_input = request.form
        query = user_input["query"]
        type = user_input["search_type"]
        if type == 'bs':
            starting_time = time.time()
            results, documents = search_engine.run("--bs", query)
            print("Time taken: {}".format(time.time() - starting_time))
            result_dict = {}
            for result in results:
                result_dict[result.id] = documents[result.id]
        elif type == 'vsm':
            starting_time = time.time()
            results, documents = search_engine.run("--vsm", query)
            print("Time taken: {}".format(time.time() - starting_time))
            result_dict = {}
            for doc in results:
                result_dict[doc] = documents[doc]
        # no .pickle yet
        elif type == 'nb':
            starting_time = time.time()
            results, documents = search_engine.run("--nb", query)
            print("Time taken: {}".format(time.time() - starting_time))
            result_dict = {}
            for result in results:
                result_dict[result.id] = documents[result.id]
    return render_template('results.html', result=result_dict)

if __name__ == '__main__':
    app.run()
