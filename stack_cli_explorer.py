#!/usr/bin/env python
# coding: utf8

import logger_init
import requests
from bs4 import BeautifulSoup
import simplejson as json
import os
import os.path
from pathlib import Path
from datetime import datetime
import stack_rc_classifier_baseline

# init
s = requests.Session()
base_url = 'https://api.stackexchange.com/2.2/'
# site = 'stackoverflow'
site = 'serverfault'

annotation_filename = './outputs/annotated_qa_items_dict.json'
logger = logger_init.init_logger("stackcli_explorer")


def get_number_input(user_input, default=5):
    try:
        val = int(user_input)
        return int(user_input)
    except ValueError:
        logger.info("No.. input is not a number. Using default ({})".format(default))
        return default


def find_questions(query_terms, page_nb=1, page_size=30):
    logger.info("finding questions with seqrch terms [{}]".format(query_terms))
    search_api = 'search/advanced'

    search_params = 'page={}&pagesize={}&order=desc&sort=relevance&accepted=True&site={}&filter=!bDxS)Rni*ko3Nm'.format(page_nb, page_size, site)
    search_query_params = ''
    if len(query_terms) > 0:
        search_query_params = '&q={}&title={}'.format(query_terms, query_terms)
    query_url = base_url+search_api+'?'+search_params+search_query_params

    logger.info(query_url)

    search_response = s.get(query_url)
    search_response.raise_for_status()

    responses = search_response.json()

    logger.info("Remaining query quota: {}".format(responses["quota_remaining"]))

    return responses


def get_annotation(message):
    while(True):
        typed_key = input(message)

        if typed_key == 'a' or typed_key == 'A':
            return True
        elif typed_key == 'r' or typed_key == 'R':
            return False
        else:
            logger.info("invalid input: {}".format(typed_key))
            logger.info("Invalid input, only A/R accepted.")


def filter_questions(questions, N=5):
    print(chr(27) + "[2J\033[0;0H====================================================")
    logger.info("Found {} possible results, keeping top {}".format(len(questions), N))
    accepted_questions = dict()
    answer_to_question = dict()
    for item in questions:
        if N <= 0:
            break

        # skeip questions already annotated
        if str(item["question_id"]) in qa_items_dict:
            logger.info("Question #{} already processed, skipping it!".format(item["question_id"]))
            logger.info("question #{} skipped".format(item["question_id"]))
        # keep the ones that have a validated anwser
        elif "accepted_answer_id" in item:
            print('\n\n====================================================')
            logger.info("Q#{} - {}".format(item["question_id"], BeautifulSoup(item["title"], 'html.parser').get_text()))
            print('====================================================')

            if get_annotation("(A) accept | (R) reject: "):
                logger.info("question #{} accepted".format(item["question_id"]))
                accepted_questions[item["question_id"]] = item
                answer_to_question[item["accepted_answer_id"]] = item["question_id"]
                N = N-1
            else:
                logger.info("question #{} rejected".format(item["question_id"]))

    return accepted_questions, answer_to_question


def get_answers_body(answers_ids):
    answers_api = 'answers/'
    answers_params = 'order=desc&sort=activity&site={}&filter=!-.AG)rCVNMau'.format(site)
    answers_ids_params = ''
    for id in answers_ids:
        answers_ids_params += str(id)+";"
    answers_ids_params = answers_ids_params[:-1]

    answer_api_url = base_url+answers_api+answers_ids_params+'?'+answers_params
    logger.info(answer_api_url)

    answers_response = s.get(answer_api_url)
    answers_response.raise_for_status()

    responses = answers_response.json()

    logger.info("Remaining query quota: {}".format(responses["quota_remaining"]))

    return responses["items"]


def train_classifier(model_output_dir = "./outputs/models/"):
    if os.path.isdir(model_output_dir):
        logger.info("Previous model to be loaded: {}".format(model_output_dir))
    else:
        logger.info("No preexisting model yet to load.")
    
    return stack_rc_classifier_baseline.train(annotation_filename, model=None, output_dir=model_output_dir, n_iter=20, n_texts=2000, init_tok2vec=None)


def save_annotation(qa_items):
    logger.info("Saving annotation in {}".format(annotation_filename))
    with open(annotation_filename, 'w') as f:
        json.dump(qa_items, f)


print(chr(27) + "[2J\033[0;0H====================================================")
qa_items_dict = dict()
if os.path.isfile(annotation_filename):
    with open(annotation_filename) as f:
        qa_items_dict.update(json.load(f))
        logger.info("{} loaded with {} qa item already annotated".format(annotation_filename, len(qa_items_dict)))
        logger.info("{} qa item already annotated".format(len(qa_items_dict)))

        nlp_model = train_classifier("./outputs/models/")

# number of expected questions to annotate
logger.info("\n====================================================")
N = get_number_input(input("How many question to explore? (default is 5): "))

# get keywords inputs for search
print('\n\n====================================================')
search_query = input("Enter your search query based on server logs (press enter for random questions): ")
# query top 30 results
search_responses = find_questions(search_query)

questions_found = search_responses["items"]

# present questions's title for possible rejection
accepted_questions_map, answer_to_question_map = filter_questions(questions_found, N)

logger.info("{} questions accepted, looking deeper now...".format(len(accepted_questions_map)))

# get answer body
answers = get_answers_body(answer_to_question_map.keys())

# local performance records for prediction against annotation
tp = 0.0  # True positives
fp = 1e-8  # False positives
fn = 1e-8  # False negatives
tn = 0.0  # True negatives

# present question body and accepted answer body then ask if it can be considered as valid answser and possible root cause
for answer in answers:
    question = accepted_questions_map[answer["question_id"]]

    if question is None:
        logger.info("Missing question {} in to match answer {} retrieved data... Skipping".format(answer["question_id"], answer["answer_id"]))
    else:
        qa_item = dict()
        qa_item["question"] = question
        qa_item["answer"] = answer
        qa_item["validated_answer"] = None
        qa_item["validated_root_cause"] = None

        print(chr(27) + "[2J\033[0;0H====================================================")
        logger.info("Q#{} - {}".format(question["question_id"], BeautifulSoup(question["title"], 'html.parser').get_text()))
        print('----------------------------------------------------')
        logger.info("Question body\n"+BeautifulSoup(question["body"], 'html.parser').get_text())
        print('----------------------------------------------------')
        logger.info("Answer body\n"+BeautifulSoup(answer["body"], 'html.parser').get_text())

        # predict RC class from answer text only        
        doc = nlp_model(BeautifulSoup(answer["body"], 'html.parser').get_text())
        # boolean prediction just on local score (could use thresholding but need to learn a good one...)
        predicted_root_cause = doc.cats["POSITIVE"] > doc.cats["NEGATIVE"]
        
        print('====================================================')

        # get annotation
        qa_item["validated_answer"] = get_annotation("Good answer: (A) accept | (R) reject: ")
        qa_item["validated_root_cause"] = get_annotation("Root cause in answer: (A) accept | (R) reject: ")

        print('====================================================')

        # present prediction (wrongly assuming that score translates to percent)
        logger.info(doc.cats)
        logger.info("{:05.2f}% chance of answer containing root cause.".format(100*doc.cats["POSITIVE"]))

        # check against prediction
        msg = "... one more sample to learn."

        if predicted_root_cause and qa_item["validated_root_cause"]:
            msg = "Cool the model was right!"
            tp += 1.0
        elif predicted_root_cause and not qa_item["validated_root_cause"]:
            fp += 1.0
        elif not predicted_root_cause and not qa_item["validated_root_cause"]:
            msg = "Cool the model was right!"
            tn += 1
        elif not predicted_root_cause and qa_item["validated_root_cause"]:
            fn += 1

        input("\n{} Press Enter to continue...".format(msg))

        qa_items_dict[question["question_id"]] = qa_item

save_annotation(qa_items_dict)

# local performance for prediction against annotation
precision = tp / (tp + fp)
recall = tp / (tp + fn)
if (precision + recall) == 0:
    f_score = 0.0
else:
    f_score = 2 * (precision * recall) / (precision + recall)

accuracy = (tp + tn)/(tp + tn + fp + fn)
perf = {"rc_precision": precision, "rc_recall": recall, "rc_f1score": f_score, "accuracy": accuracy}
logger.info(perf)
logger.info(perf)
