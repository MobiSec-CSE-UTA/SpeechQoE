# SpeechQoE
[\[Paper\]](https://www.google.com) &ensp; [\[Dataset\]](https://www.google.com)

The SpeechQoE dataset is a collection of 7600 speech segments with user experience opinion annotated in the range [1, 5] from the worst to the best in online voice services. 

## Description
The dataset collect from 38 subjects. Two subjects form a pair to complete 200 calling sessions, each lasting 90 seconds. They sit in two separate rooms to finish a so-called Richard’s task which is like charades. Two subjects take turns describing a shape for the other to guess. A subjective assessment is conducted based on the calling experience the subject perceived after each session.

## Format
The data is organized into three levels, user_id -> assessment_score -> session_id. Especially, the assessment_score is the label for all the speech files under this folder.
 
 The speech files are represented by raw amplitude and all mute periods have been removed. 
