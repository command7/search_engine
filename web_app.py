from flask import Flask, render_template, request
import search_engine
from search_engine import SearchEngine, InvertedIndex, DocumentProcessing, Document

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
        print(query)
        #results = search("--bs", query)
        results, documents = search_engine.run("--bs", query)
        with open("query_result.txt", "w+") as handle:
            result_dict = {}
            for result in results:
                print(result.id)
                print(documents[result.id])
                result_dict[result.id] = documents[result.id]
    return render_template('results.html', result=result_dict)

if __name__ == '__main__':
    app.run()
