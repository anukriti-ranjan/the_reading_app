import pandas as pd
import random
import unicodedata
import re
from pathlib import Path
import streamlit as st
import cohere
import hnswlib
import os
import numpy as np

API_KEY = st.secrets["cohere_API_key"]
co = cohere.Client(API_KEY)

def clean_text(s):
    # Turn a Unicode string to plain ASCII
    def unicodeToAscii(s1):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s1)
            if unicodedata.category(c) != 'Mn')
    s = s.replace("&nbsp;","")
    s =" ".join(unicodeToAscii(s.strip()).split())
    return s.lower()

embedding_size = 1024    #Size of embeddings
top_k_hits = 30         #Output k hits
#generated from cohere embed 
# embeddings of title + abstract of JAIR papers
corpus_embeddings = np.load('jair_paper_embedings.npy')


#ref: https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_quora_hnswlib.py
#Defining our hnswlib index
index_path = "./hnswlib.index"
#We use Inner Product (dot-product) as Index. 
index = hnswlib.Index(space = 'cosine', dim = embedding_size)

if os.path.exists(index_path):
    index.load_index(index_path)
# Controlling the recall by setting ef:
index.set_ef(50)  # ef should always be > top_k_hits

df=pd.read_csv("jair_papers.csv")
corpus_sentences = df['title'].values.tolist()


with st.form('search_form'):
    inp_query = st.text_input('Search your query',"")
    submitted = st.form_submit_button('Submit')
    if submitted:
        question_embedding = co.embed(texts=[clean_text(inp_query)],
                            model="small", 
                            truncate="LEFT").embeddings[0]

        #We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = index.knn_query(question_embedding, k=top_k_hits)

        # We extract corpus ids and scores for the first query
        hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        search_results=[]
        for hit in hits[0:top_k_hits]:
            search_results.append(corpus_sentences[hit['corpus_id']])

        
        while True:
            title = st.radio(
                "What's the title you want to open",
                set(search_results))
            abstract=df.abstract.values[df.title.values.tolist().index(title)]
            st.write(abstract)


# the sidebar gives 3 options to readers
# 1)to search for meaning internally
# 2)to generate keywords
# 3)to test your understanding via questions
with st.sidebar:
        option = st.selectbox(
                    'Choose',
                    ('Seek Clarification', 'Generate Keywords', 'Generate Questions'))
        if option=="Seek Clarification":
            with st.form('clarification_form'):
                phrase=st.text_input('Phrase',"")
                context=st.text_input('Context',"")
                submitted = st.form_submit_button('Submit')
                if submitted:
                    if len(phrase)==0:
                        st.write("Enter phrase")
                    elif len(context)==0:
                        st.write("Enter context")
                    else:

            
                        response = co.generate(
                                        model='large',
                                        prompt=f'Given a phrase and context, this program will generate explaination of the phrase in the given context. \n﻿\nPhrase: Augmented reality\nContext: Artificial Intelligence\nExplanation: Augmented reality (AR) is an interactive experience that combines the real world and computer-generated content.The content can span multiple sensory modalities, including visual, auditory, haptic, somatosensory and olfactory.\n--\nPhrase: {phrase}\nContext: {context}\nExplanation:',
                                        max_tokens=200,
                                        temperature=0.5,
                                        k=0,
                                        p=1,
                                        frequency_penalty=0.2,
                                        presence_penalty=0.2,
                                        stop_sequences=["--"],
                                        return_likelihoods='NONE')
                        st.write(response.generations[0].text[0:-2])

        elif option=='Generate Keywords':

            response = co.generate(
                            model='large',
                            prompt=f'Given a paragraph, this program will generate important keywords.\n\nParagraph: We study the computational complexity of abstract argumentation semantics based on weak admissibility, a recently introduced concept to deal with arguments of self-defeating nature. Our results reveal that semantics based on weak admissibility are of much higher complexity (under typical assumptions) compared to all argumentation semantics which have been analysed in terms of complexity so far. In fact, we show PSPACE-completeness of all non-trivial standard decision problems for weak-admissible based semantics. We then investigate potential tractable fragments and show that restricting the frameworks under consideration to certain graph-classes significantly reduces the complexity. We also show that weak-admissibility based extensions can be computed by dividing the given graph into its strongly connected components (SCCs). This technique ensures that the bottleneck when computing extensions is the size of the largest SCC instead of the size of the graph itself and therefore contributes to the search for fixed-parameter tractable implementations for reasoning with weak admissibility.\nKeywords: computational complexity, abstract argumentation semantics, PSPACE-completeness, weak-admissible based semantics\n--\nParagraph: A declarative action model is a compact representation of the state transitions of dynamic systems that generalizes over world objects. The specification of declarative action models is often a complex hand-crafted task. In this paper we formulate declarative action models via state constraints, and present the learning of such models as a combinatorial search. The comprehensive framework presented here allows us to connect the learning of declarative action models to well-known problem solving tasks. In addition, our framework allows us to characterize the existing work in the literature according to four dimensions: (1) the target action models, in terms of the state transitions they define; (2) the available learning examples; (3) the functions used to guide the learning process, and to evaluate the quality of the learned action models; (4) the learning algorithm. Last, the paper lists relevant successful applications of the learning of declarative actions models and discusses some open challenges with the aim of encouraging future research work.\nKeywords: declarative action model, state constraints, combinatorial search\n--\nParagraph: {abstract}\nKeywords:',
                            max_tokens=15,
                            temperature=0.5,
                            k=0,
                            p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop_sequences=["--"],
                            return_likelihoods='NONE')
            st.write(response.generations[0].text)
        else: 

                response = co.generate(
                            model='large',
                            prompt=f'Given a paragraph, this program will generate questions from the paragraph.\n\nParagraph: A declarative action model is a compact representation of the state transitions of dynamic systems that generalizes over world objects. The specification of declarative action models is often a complex hand-crafted task. In this paper we formulate declarative action models via state constraints, and present the learning of such models as a combinatorial search. The comprehensive framework presented here allows us to connect the learning of declarative action models to well-known problem solving tasks. In addition, our framework allows us to characterize the existing work in the literature according to four dimensions: (1) the target action models, in terms of the state transitions they define; (2) the available learning examples; (3) the functions used to guide the learning process, and to evaluate the quality of the learned action models; (4) the learning algorithm. Last, the paper lists relevant successful applications of the learning of declarative actions models and discusses some open challenges with the aim of encouraging future research work.\nQuestions: What is a declarative action model? How does the framework mentioned help to characterize the existing work? What are some of the open challenges with respect to the successful applications of the learning of declarative actions models?\n--\nParagraph: {abstract}\nQuestions:',
                            max_tokens=50,
                            temperature=0.5,
                            k=0,
                            p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop_sequences=["--"],
                            return_likelihoods='NONE')
                st.write(response.generations[0].text)














