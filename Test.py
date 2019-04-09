import search_engine
from search_engine import SearchEngine, InvertedIndex, DocumentProcessing, Document
import pickle


# search_engine = SearchEngine.load_engine(
#             filename="pickled_objects/Boolean_Search_Engine.pickle")
# query = "Angel Serenity"
# results = search_engine.boolean_and_query(query)
# with open("query_result.txt", "w+") as handle:
#     for result in results:
#         handle.write("Document Number: {}\n".format(result.id))
#         print("Document Number: {}".format(result.id))
#         handle.write(search_engine.documents[result.id] + "\n\n")
#         print(search_engine.documents[result.id][:100] + "\n\n")
#     handle.write("Documents IDs : \n {}".format([doc.id for doc in results]))
#     print("Documents IDs : \n {} \n".format([doc.id for doc in results]))
#     print("Total Number of Documents found: {}\n".format(len(results)))
#     handle.write("Total Number of Documents found: {}\n".format
#                  (len(results)))
# print(search_engine.run("--nb", "documents/entertainment/001.txt"))
