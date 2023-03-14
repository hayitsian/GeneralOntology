README file for the General Index

- There are no rights reserved on this public domain data.
- This is an alpha release dated October 4, 2021.
- The General Index was created by Public.Resource.Org, Inc., a 501(c)(3) nonprofit.
- The URL for the General Index is https://archive.org/details/GeneralIndex
- The data files total 4.7 tbytes, but will expand to 37.9 tbytes when unzipped.

- The corpus of 107,233,728 articles has been split into 16 slices, numbered from 0 to f.
- The files in this distribution were created using the Postgres pgdump command. 
- The collection is not complete and text extraction was not always successful. 

- The metadata and sample files are here (https://archive.org/download/GeneralIndex/data).
- The ngrams and keywords files are each on their own item. 
- Ngrams are on identifiers with the naming scheme GeneralIndex.ngrams.n where n=0..f
- Keywords are on identifiers with the scheme GeneralIndex.keywords.n where n=0..f
- So, ngram slice 0 is at https://archive.org/download/GeneralIndex.ngrams.0

You can see all the items here:
https://archive.org/search.php?query=%22general%20index%22%20AND%20collection%3Amulticasting

1. The ngrams Table
- The _ngrams table is the core of the General Index.
- SpaCy is used to extract ngrams, from unigrams to 5-grams, into the doc_ngrams_n tables.
- There are 355,279,820,087 rows in total.
- Each row represents how many instances of an n-gram are in an article.
- The files unzip to 2.1-2.3 tbytes each, for a total of 36 tbytes. 
- There are 3 sample files generated using head and fgrep.

2. The keywords table
- The _keywords table extracts the meaningful terms in a document.
- YAKE is used to extract document keywords. 
- There are 19,740,906,314 rows. 
- The files unzip to 95-102 gbytes each, for a total of 1.6 tbytes. 
- Sample files are available. 

3. The metadata table
- The _info table attempts to map an md5 unique identifier to metadata.
- In some cases, we are unable to extract appropriate metadata. 
- In some cases, the data may be wrong. 
- The files unzip to 70 gbytes total.
- A sample file is available. 

- *NEW* An updated combined metadata file that unzips to 70 gbytes is available.
- The slice metadata files have also been updated with enhanced metadata. 

An easy way to begin is to start working with a single slice.
Loading the keywords and metadata for one slice is a way to work with the data. 
While we provide Postgres load files, feel free to parse these into other formats.
We hope to add other information, such as td/idf in the future. 

==========
The Tables
==========

doc_ngrams_n – 16 slices: 0-f
  dkey [text]: document key (md5 hash of document)
  ngram [text]: proper case version of ngrams (unigrams, bigrams, trigrams, 4grams, 5grams)
  ngram_lc [text]: lower case version of ngrams – best for search
  ngram_tokens [int]: number of tokens (words) in the ngram (e.g., unigrams: 1, bigrams: 2)
  term_freq [numeric]: number of occurences of the ngram in the document
  doc_count [int]: always 1 (used for other analytic purposes)
  insert_date [date]: date record inserted into table, initial load has a null insert_date

doc_keywords_n – 16 slices: 0-f
  dkey [text]: document key (md5 hash of document)
  keywords [text]: proper case version of keywords captured by YAKE process, from 1 to 5grams
  keywords_lc [text]: lower case version of keywords
  keywords_tokens [int]: number of tokens (words) in the keywords phrase (e.g., unigrams: 1, bigrams: 2)
  keyword_score [numeric]: YAKE score of how meaninful the word is in the document, the smaller value, the more meaningful
  doc_count [int]: always 1 (used for other analytic purposes)
  insert_date [date]: date record inserted into table, initial load has a null insert_date

doc_info_n – 16 slices: 0-f
  dkey [text]: document key (md5 hash of document)
  meta_doi [text]: DOI for doc from doc_meta source
  doc_doi [text]: DOI for doc from original text
  doi [text]: DOI for doc from doc_meta if available, else from original text
  doc_pub_date [date]: publish date for document from original text
  meta_pub_date [date]: publish date for document from original text
  pub_date [date]: publish date for document from doc_meta if available, else from original text
  doc_authors [text]: list of authors from original text
  meta_authors [text]: list of authors from doc_meta
  authors [text]: list of authors from doc_meta if available, else from original text
  doc_title [text]: document title from original text
  meta_title [text]: document title from doc_meta
  title [text]: document title from doc_meta if available, else from original text

/sign/ Carl Malamud (carl@media.org)  :seal:
Last revised: Mon Oct 22 12:17:08 PDT 2021



