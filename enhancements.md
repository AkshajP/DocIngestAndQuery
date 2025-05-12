# Notes

## Future Enhancements

- [ ] Remove welcome message from backend
- [ ] Add copy and feedback icons and api to frontend
- [ ] Check why sources are not showing up
- [ ] Emulate DASH using AG Grid 
    - [ ] Complexity - considering folder structure as an upload.
    - [ ] Deciding migration of `registry.json`
- [ ] Sidebar and DASH modal integration
- [ ] Stream response
- [ ] ability to interact with doc processings - what failed

---

- [ ] ag grid interact
- [x] doc process tracking
- [ ] AI int for long context answer generation

embeddings generatED folder structure

## STUBBBB

## Advanced

- Retry mechanism and mid-process persistence
case user role documents

### errors:
on processing very small file I think: 
[2025-05-09 12:59:50,621] [   ERROR] upload.py:312 - Error processing document doc_1746775567_Index_Volume_7: Cannot use scipy.linalg.eigh for sparse A with k >= N. Use scipy.linalg.eigh(A.toarray()) or reduce k.
[2025-05-09 12:59:50,621] [   ERROR] upload.py:363 - Document doc_1746775567_Index_Volume_7 processing failed at processing: Cannot use scipy.linalg.eigh for sparse A with k >= N. Use scipy.linalg.eigh(A.toarray()) or reduce k.


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



