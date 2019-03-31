# Information Retrieval & Classification

## Description

An Information Retrieval system that is capable of accommodating
*   Boolean search queries
*   Positional search queries
*   Free text queries
*   Naive Bayes Classification

## Dependencies Required

* Pandas
* Numpy
* Nltk
* Pickle

## How To Run

Currently this is a command line application and the following commands can 
be used to utilize the features of the system.

### Boolean Search

> search_engine.py --bs *query

### Positional Search

> search_engine.py --ps *query

### Free Text Search

> search_engine.py --vsm *query

### Naive Bayes Classification

> search_engine.py --nb document_path

Where document_path = Location of document to classify

### Test Queries to check functionality

AND Queries
* Jackie Ballard Parliament
* Research and Development

Positional Queries

* Idowu made his breakthrough
* Third generation mobile

Free text Queries
* Cyber computers government
* Actress actor theatre

Test Classification Documents

* test_1.txt
* test_2.txt 



## Examples

> search_engine.py --bs Anderson country

The above command will search for documents containing both "Anderson" and 
"country".

> search_engine.py --ps New York

The above command will search for documents that contain "New" and "York" in
 the same order.
 
 > search_engine.py --vsm Today in politics
 
 The above command will search and provides top 10 documents that are 
 similar to the query.