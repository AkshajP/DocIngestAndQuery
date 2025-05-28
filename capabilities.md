# Capabilites
## Admin 
 - general statistics on document processes
 - Upload documents onto the platform for processing
 - persistent processing that can resume processing from last known step
 - in case of server crashes, file locks are abandoned
    - server start itself will unlock older locks,
    - if any recent file has a lock, the document management table has the option to get it unlocked
    - in case retries are not working, a central lock release button is also made
 - ability to view processed, failed, pending files 
 - view OCR'ed chunks of the document processed
 - re process from any of the steps from the pipeline
 - view the document itself
 - view details of the document that is processed

## Query
 - (Auth is borrowed using case_id and user_id as a dependency)
 - choose documents to query through DASH like interface
 - query the selected documents
 - view a streaming response from the LLM 
 - view sources of the answer
    - view which chunks are internally processed (raptor summary nodes) and original document nodes
    - view the document citation in sidebar with an annotation overlay for all original document chunks
 - regenerate response of a given message
 - auto allocated title from first few words of prompt
 - change title of the chat 
 - ability to update documents selected
 - All text generated is rendered in markdown for easy reading 
 - ability to delete chat 
 - view all chats ordered by latest modified 



# Technical
##  Query 
    - 
 
