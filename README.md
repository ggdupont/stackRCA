# StackRCA - notes

Find solution to server issues identified in logs using stackoverflow as a knowledge base. Identify root causes in these.

## TL;DR - Just launch it

```shell
docker-compose run stackrca
```

## Approach

### Startig point

* no pre-existing annotated data
* no clear definitiion of "root cause"
* no knowledge of collection language specificities vs "normal" english

### "Plan of attack"

Fancy model might be as good (or bad) as simple one since we have very few data. The core will be to build up a first clean dataset of annotated Q/A.

* Collect data
* Assess structure of the data and metadata that could be queried to speed the search
* Given a “symptom” find relevant answered questions
* Among these, detect if a root cause is presented
* Try to assess the “confidence” in the proposed answer
* Setup annotation tools and eventualy active learning

## Collect data

### Analysis of stackoverflow API

General API:
https://api.stackexchange.com/docs 

Need to register on stack Apps
https://stackapps.com/apps/oauth/view/17982

API is cleaned and well documented. Allows to retrieve data in json but with multiple throttle to limit bandwidth/CPU usage on their end.

/!\ search api is limited and recommendation is to go through search engine with domain restriction

Can search on tags and title but not on question content
https://api.stackexchange.com/2.2/search?order=desc&sort=activity&intitle=error&site=stackoverflow

Response is an array of items and some meta to navigate the results. It gives quota information status:

```json
   "has_more": true,
   "quota_max": 300,
   "quota_remaining": 286
```

Sample item (ie answer):

```json
{
           "tags": [
               "c#",
               "visual-studio",
               "visual-studio-2012",
               "configuration",
               "edit-and-continue"
           ],
           "owner": {
               "reputation": 8844,
               "user_id": 900570,
               "user_type": "registered",
               "accept_rate": 100,
               "profile_image": "https://www.gravatar.com/avatar/dcd5fb635665e9ea3634f7cd413e9ad4?s=128&d=identicon&r=PG",
               "display_name": "Avi Turner",
               "link": "https://stackoverflow.com/users/900570/avi-turner"
           },
           "is_answered": true,
           "view_count": 144203,
           "protected_date": 1496999677,
           "accepted_answer_id": 20692783,
           "answer_count": 26,
           "score": 74,
           "last_activity_date": 1590700893,
           "creation_date": 1386668460,
           "last_edit_date": 1495535470,
           "question_id": 20490857,
           "content_license": "CC BY-SA 3.0",
           "link": "https://stackoverflow.com/questions/20490857/visual-studio-getting-error-metadata-file-xyz-could-not-be-found-after-edi",
           "title": "Visual studio - getting error &quot;Metadata file &#39;XYZ&#39; could not be found&quot; after edit continue"
       }
```

It provides the to `question_id` query the question api. Plus it says if there is answers `"is_answered": true` and if there is an accepted answer `"accepted_answer_id": 20692783`
This enables to jump to the validated answer.

### Analysis of stackoverflow data dump

Can be downloaded on [internet archive website](https:// archive.org/download/stackexchange) on a quarterly basis and subsite by subsite.
Data is exported in XML.

Files list:

* Badges.xml
* Comments.xml
* PostHistory.xml  
* PostLinks.xml
* Posts.xml
* Tags.xml
* Users.xml
* Votes.xml

Post sample:

```xml
<row
Id="1"
PostTypeId="1"
AcceptedAnswerId="509"
CreationDate="2009-04-30T06:49:01.807"
Score="19"
ViewCount="5620"
Body="&lt;p&gt;Our nightly full (and periodic differential) backups are becoming quite large, due mostly to the amount of indexes on our tables; roughly half the backup size is comprised of indexes.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;We're using the &lt;strong&gt;Simple&lt;/strong&gt; recovery model for our backups.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;Is there any way, through using &lt;code&gt;FileGroups&lt;/code&gt; or some other file-partitioning method, to &lt;strong&gt;exclude&lt;/strong&gt; indexes from the backups?&lt;/p&gt;&#xA;&#xA;&lt;p&gt;It would be nice if this could be extended to full-text catalogs, as well.&lt;/p&gt;&#xA;"
OwnerUserId="3"
LastEditorUserId="919"
LastEditDate="2009-05-04T02:11:16.667"
LastActivityDate="2009-05-10T15:22:39.707"
Title="How to exclude indexes from backups in SQL Server 2008"
Tags="&lt;sql-server&gt;&lt;backup&gt;&lt;sql-server-2008&gt;&lt;indexes&gt;"
AnswerCount="4"
CommentCount="0"
FavoriteCount="3"/>
```

Most of the data is in the posts (whether it’s a question of an answer) and can be distinguished by the `PostTypeId` meta. It also directly provides the `AcceptedAnswerId` for questions. Seems a good source for large scale unsupervised training. Not clear for question/answer match and annotation (need extra pre-processing to reconstruct the links).

## Assess structure

The objective is to speed up the search process and have heuristics to eliminate posts that might not lead to a root cause.

**Hypothesis:** every case will start with some “server error” based on logs and thus basic keywords can be extracted from these.

First approach:

* Filter the error logs to construct a (good) search query
* Query the search api with the search from the server error on the title AND the body
* Filter out results to keep only the one with an accepted answer
* Keep the top N (static based on processing time available)
* Assess match between the server error and the retrieved questions
* Keep only the top K based on threshold of match
* Assess if for each question and its accepted answer selected contains root cause => binary classification problem

## Find relevant answered questions and assess root cause

Build a simple explorator using the REST api allowing to query for questions, checked the accepted answers and build a small annotated dataset.

=> stack_cli_explorer.py

### Tain basic root cause detector

Train a basic model for classification of root cause presence in answer. Using simplest example from spacy as a baseline until sufficient data is collected.

### Active learning style annotation approach

* launch annotator cli
* if any data exists already
    * load data
    * train model
* query for more questions
* for each accepted question
    * predict class with classifier
    * possibly: 
        * if confidence high => skip
        * if confidence low => ask user feedback for annotation
* as soon as more data annotated (ie 10% more): ask to retrain the model
* report performance and continue

### TODO

* [ ] assess balance of annotated data
* [ ] assess inter-annotator agreement on annotation
* [ ] test thresholding the classification score to enable rejection
* [ ] analyse vocabulary and distributional semantics on the offline corpus
* [ ] explore unsupervised pre-training for better model
