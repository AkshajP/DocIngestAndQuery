# Notes

Currently if the document is uploaded, then the case_path attribute needs to be manually set in registry.json 
to ensure the folder structure is seen else files will be under un-categorized.

## Future Enhancements

- [ ] Sidebar and DASH modal integration
- [ ] Regenerated messages should be multi-view
- [ ] Remove welcome message from backend
- [ ] Add copy and feedback icons and api to frontend
- [x] Check why sources are not showing up
- [x] ability to interact with doc processings - what failed
- [x] Filter expressions for text and table content from vector db
- [x] Emulate DASH using AG Grid 
    - [ ] Complexity - considering folder structure as an upload.-- ADD MANUALLY. 
    - [x] Deciding migration of `registry.json`
- [x] Stream response

---

- [x] ag grid interact
- [x] doc process tracking
- [ ] AI int for long context answer generation
    1. i have already implemented a heirarchical chunking and indexing strategy,

    2. a hybrid retrieval - (vector + keyword/sparse search like BM25) to catch both conceptual relevance and exact term matches. 
        2.1 can re rank on a translated query

    3. i have my own twist to query focused selection- each of my documents has a metadata, so i was thinking of having a smart document router which passes what to search in what document to bring back results.

    4. I was thinking of including a smart token filler that dynamically adjusts the top_k parameter based on available prompt space, optimizing for more sources when chat history is small and reducing as it grows.

embeddings generatED folder structure

## STUBBBB

## Advanced

- Retry mechanism and mid-process persistence
case user role documents

### errors:





Document Selection with AG Grid: Summary & Plan

I want:

Documents organized in a hierarchical folder structure (e.g., A/doc_abc.pdf)

Implemented using AG Grid with tree data nesting features

Rich metadata display beyond what's in registry.json

Selected document IDs passed to existing API functions

Steps Taken

Initial Setup

Created a data transformer utility to convert flat document lists into hierarchical tree structure

Implemented mock metadata generation (importance, size, page counts)

Organized documents into department-based folders (Legal, Finance, HR, etc.)

AG Grid Component Implementation

Built a custom AG Grid document selector component

Configured tree data structure with proper parent-child relationships

Added document selection capability with checkboxes

Implemented visual indicators (folder/file icons, status badges)

Error Resolution

Fixed module registration by using AllEnterpriseModule

Resolved theming conflict by setting theme="legacy"

Ensured component properly integrates with existing application

Plan for Continuing Development

Enhanced Filtering & Searching

Implement advanced filtering options for each column

Add global search functionality to quickly find documents

Allow filtering by document metadata (status, importance, etc.)

Improved User Experience

Add sorting for each column

Implement expand/collapse all functionality

Add context menus for additional actions

Visual Enhancements

Improve folder/file styling and visual hierarchy

Add tooltips for additional information on hover

Implement document previews on selection

Data Management

Improve document metadata extraction from actual files

Add better folder organization logic (perhaps date-based or by document type)

Implement document grouping preferences

Integration Refinement

Ensure seamless integration with chat creation workflow

Test with larger document sets for performance

Add proper error handling and loading states

With these improvements, the document selector will provide a powerful and intuitive interface for users to find and select documents for chat sessions.



