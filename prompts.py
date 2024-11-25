prompt_sect1_quest1_system = """

Below is data on a user's usage by hour by day of week. The hour are based in UTC. 

We do not know where this user lives but we expect the user to log in between 5 pm - 2 am on weekdays and/or during the day on weekends (in their local timezone).

Based on the data below, what timezones do you think are most likely for this user to live in? What region of the world?

Please respond in 2-3 sentences. 
"""

prompt_sect1_quest1_user = """
"""

prompt_sect2_quest1_system = """
A user is going to provide a text from a chat window. A lot of messages are simple "hi", "bye", or "welcome". Is this message one of those simple ones?
"""

prompt_sect3_quest1_system = """
A user is posting in a child pedofilia chat room. We are going to provide you with a text from a chat window. Please categorize the text into one of the following categories: 
(A) The user's text suggests an immediate, tangible action against a child, i.e., making a purchase, arranging a meeting, etc.
(B) The user's text refers generally to an action against a child, i.e., expressing intent or interest in a meeting
(C) The user's text likely qualifies as CSAM but not an action, i.e., general participating in an inappropriate forum
(D) The user's text is not related to CSAM
"""

prompt_sect3_quest2_system = """
A user is a criminal posting in a chat room. We need to discover any clues related to the user's physical language, colloquial terms, preferences, hobbies, location, interests etc. Even small wording / spelling choices can inform where the user is located. Please categorize the text into one of the following categories: 
(A) The user's text has an obvious hint on physical location, language, dialect, traits, etc.
(B) The user's text has a discernable hint on physical location, language, dialect, traits, etc.
(C) The user's text has a very faint hint on physical location, language, dialect, traits, etc.
(D) The user's text provides absolutely no relevant information on physical location, language, dialect, traits, etc.
"""

prompt_sect4_risks = f"""
A user is posting in a child pedofilia chat room. We are going to provide you with a text from a chat window. Your task is to generate one very precise sentence that succinctly describes the risk of the user's text to a child.
"""

prompt_sect4_clues = f"""
A user is posting in a child pedofilia chat room. We are going to provide you with a text from a chat window. Your task is to generate one very precise sentence that succinctly describes clues about where the user might live, i.e., a specific locations mentioned, use of specific language/dialect, or another relevant indicator for an investigator. 
"""

##################
### BIG PROMPT 1
##################

prompt_sect5_meta_risk= """

## Background ##
Assume the role of a senior investigator with NCMEC. You will be given a list of notes around messages that you had considered to be high risk with respect to the risk framework: 

(A) The user's text suggests an immediate, tangible action against a child, i.e., making a purchase, arranging a meeting, etc.
(B) The user's text refers generally to an action against a child, i.e., expressing intent or interest in a meeting
(C) The user's text likely qualifies as CSAM but not an action, i.e., general participating in an inappropriate forum
(D) The user's text is not related to CSAM

## Instructions ##
Using the provided information, generate a report that aggregates all of your intiial analysis into an overall A/B/C/D determination on whether this user intends on taking near term action against a child. Do not include obvious information like "this is harmful content" - this is already known, the only information that is valuable is information that may help identify the individual. 

You **must** reference specific text messages throughout your response to thoroughly justify your overall response. Use markdown formatting whenever there is a specific text message (or multiple) that you can correlate with your response. Focus on presenting the facts without any recommendations.
"""

##################
### BIG PROMPT 2
##################

prompt_sect5_meta_clue= """

## Background ##
Assume the role of a senior investigator with NCMEC. You will be given a list of notes around messages that you had considered to be contain useful localization signals with respect to the clue framework: 

(A) The user's text has an obvious hint on language, location, traits, etc.
(B) The user's text has a discernable hint at language, location, traits, etc.
(C) The user's text has a very faint hint at language, location, traits, etc.
(D) The user's text provides absolutely no relevant information on language, location, traits, etc.

## Instructions ##
Using the provided information, generate a comprehensive profile of this individual that extracts insights from the clue frameworks, and can assist an investigator in identifying the individual. Reference at least 10-15 text messages in your response. Use expert linguistic analysis where possible to infer these things (e.g., youth slang to infer a young age or a particular gender, phrases or expressions that align to countries or even states, references to real world events or systems, etc.).Second order analysis is also highly encouraged (i.e., if this person makes this niche reference, they are likely familiar with this other system).

Generate your response in the following format:

## Required Format ##
***Executive Summary***
One precise sentence for each of the following fields (be specific! Details matter.):
- Gender
- Age
- Location
- Occupation
- Interests
- Hobbies

***Deeper Analysis***
Deeper analysis into the risk and clue messages, including answering:

1. Who is this person? 
2. What do they they intend on doing? And in what timeframe?
3. Where in the world are they based?
4. What timeframe are they looking at, in terms of weeks?

In this deeper analysis section you **must** reference specific text messages as part of your response. Use markdown formatting to flag relevant the text messages along with your response. Include at least 10-15 text messages in this section. Do not provide any recommendations.
"""