**Revised Phase 1 Plan: Document Processing Pipeline with Celery**


## Phase 1 – Step 1: Infrastructure & Celery Foundation

**Goal**: Stand up Redis + Celery; verify simple task execution; create tracking table.

1. **Modify/Add Files**

   * `docker-compose.yaml`

     * Add a Redis service on port 6379.
     * Define separate services for “web” (API) and “worker.”
   * `requirements.txt`

     * Add dependencies: `celery[redis]` and any other Celery extras (e.g., `redis`).
   * `core/celery_app.py` (new)

     * Create a Celery application instance.
     * Configure `broker_url` and `result_backend` to point at Redis via environment variables (e.g., `REDIS_URL`).
     * Use JSON serialization (`task_serializer="json"`, `result_serializer="json"`).
   * `worker.py` (new)

     * Entry point to launch a Celery worker (e.g., `celery -A core.celery_app worker --loglevel=info --concurrency=3 -Q document_processing`).
   * `core/config.py`

     * Add `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` entries (read from env).
   * `db/migrations/add_document_tasks_table.sql`

     * Create a `document_tasks` table with:

       * `id` (primary key)
       * `document_id` (FK to documents)
       * `current_stage` (varchar)
       * `celery_task_id` (varchar)
       * `task_status` (varchar: “PENDING”, “STARTED”, “SUCCESS”, “FAILURE”, “PAUSED”, “CANCELLED”)
       * `created_at`, `updated_at` (timestamps)
       * `can_pause` (bool), `can_resume` (bool), `can_cancel` (bool)
       * Any metadata fields (e.g., `started_at`, `completed_at`, `worker_id`)

2. **Implementation Details**

   * **Redis as Broker & Backend**:

     * Set `broker_url = redis://redis:6379/0` and `result_backend = redis://redis:6379/1` in `core/config.py`.
   * **Celery Startup**:

     * In `docker-compose.yaml`, define environment variables for both “web” and “worker” containers to point at Redis.
     * Expose 6379 internally; no external exposure.
   * **Unit Test**:

     * Write a minimal Celery task in `core/celery_app.py` (e.g., `@celery.task def ping(): return "pong"`).
     * Add a unit test that enqueues `ping.delay()`, waits for result, and asserts `"pong"`.

3. **Success Criteria**

   * Redis container starts without errors.
   * Running `celery -A core.celery_app worker` picks up the test task, and the unit test passes.
   * The `document_tasks` table exists in Postgres and can be queried.

---

## Phase 1 – Step 2: Task State Management

**Goal**: Extend state‐tracking to incorporate Celery task IDs and status flags.

1. **Modify/Add Files**

   * `services/document/processing_state_manager.py`

     * Refactor existing state manager to record and update:

       * `celery_task_id` for each stage
       * `task_status` in the `document_tasks` table
       * Boolean flags: `can_pause`, `can_resume`, `can_cancel`
   * `services/celery/task_state_manager.py` (new)

     * Encapsulate logic for:

       * Inserting a new row into `document_tasks` when a workflow begins.
       * Updating status transitions (`PENDING` → `STARTED` → `PAUSED` → `RESUMED` → `SUCCESS` → `FAILURE` → `CANCELLED`).
       * Fetching latest status for polling endpoints.
       * Handling TTL or cleanup policies for old tasks (optional).
   * `db/document_store/repository.py`

     * Alter ORM or SQL methods to read/write the new columns in `document_tasks`.
     * Add methods to:

       * Mark a task as paused/resumed/cancelled.
       * Query by `celery_task_id` or `document_id`.

2. **Implementation Details**

   * **Tracking Workflows**

     * On “upload” (Phase 1 Step 4), insert a new `document_tasks` row with `current_stage = "uploaded"`, `task_status = "PENDING"`.
     * As Celery tasks start, update `task_status = "STARTED"` and `can_pause = true`, etc.
   * **Pause/Resume/Cancel Flags**

     * `can_pause` = true when a task is eligible to pause; false once fully complete or already paused/cancelled.
     * `can_resume` = true when status is “PAUSED.”
     * `can_cancel` = true unless the task is already in terminal status.
   * **Unit Tests**

     * For each state‐manager function:

       1. Insert a dummy row; verify updating from “PENDING” to “STARTED” sets correct flags.
       2. Simulate a pause request: set `task_status = "PAUSED"`; assert that `can_resume` flips to true and `can_pause` flips to false.
       3. Simulate cancel: ensure status becomes “CANCELLED.”

3. **Success Criteria**

   * The state manager can insert and retrieve task records.
   * Status transitions behave as expected in unit tests.
   * No database schema errors when writing/reading `document_tasks`.

---

## Phase 1 – Step 3: Core Document Processing Tasks

**Goal**: Implement each processing stage as a separate Celery task (with basic pause/resume/cancel hooks).

1. **Modify/Add Files**

   * `services/celery/tasks/base_task.py` (new)

     * Define a `BaseDocumentTask` subclass of `celery.Task` that:

       * Provides `on_success`, `on_failure`, and `on_retry` to update `document_tasks.task_status`.
       * Checks a “pause flag” from `document_tasks` before executing each major sub‐operation. If flagged, raise a custom exception to let the task stop at the next safe point.
       * Listens for a “cancel flag” and terminates immediately, cleaning up any in‐memory buffers.
   * `services/celery/tasks/document_tasks.py` (new)

     * Five independent Celery tasks:

       1. `extract_document_task(document_id)`
       2. `chunk_document_task(document_id)`
       3. `embed_document_task(document_id)`
       4. `build_tree_task(document_id)`
       5. `store_vectors_task(document_id)`
     * Each task:

       * Reads `document_tasks` to fetch any paused state (e.g., partial offsets).
       * Writes partial results to the database when reaching a logical sub‐step (checkpoint).
       * Calls `task_state_manager` to update `task_status` and metadata.
       * Upon completion, sets `current_stage` to the next stage.
   * `services/document/stage_processors.py`

     * Refactor existing synchronous logic: instead of performing work inline, delegate to Celery tasks by invoking `extract_document_task.delay(…)` and returning the Celery task ID.
     * Remove blocking logic; no synchronous fallback.
   * `services/celery/task_utils.py` (new)

     * Common helper functions:

       * Fetching and updating checkpoint metadata from `document_tasks`.
       * Graceful sleep loops or “yield” points to allow pausing.
       * Small retry/backoff wrapper if external calls fail (e.g., embedding API).

2. **Implementation Details**

   * **Task Chaining**

     * In each task’s `run()` method, after finishing its work, enqueue the next stage’s task (e.g., at end of `extract_document_task`, call `chunk_document_task.delay(document_id)`).
     * Alternatively, use Celery’s `chain()` primitive when kicking off the pipeline.
   * **Pause/Resume Hooks**

     * Immediately after each major sub‐operation (e.g., after extracting all raw text, after every N chunks embedded), check the `task_status` in `document_tasks`. If it’s been set to “PAUSED,” save your current offset to the database and exit.
     * On a resume request, fetch that offset from `document_tasks`, and continue from there.
   * **Cancel Handling**

     * If `task_status` is set to “CANCELLED” at any time, clean up partial data (e.g., delete any rows created for that document in intermediate tables), then mark `task_status = "CANCELLED"` and stop further chaining.
   * **Error Recovery**

     * Use Celery’s built‐in retry: for network or API errors, annotate tasks with `autoretry_for=[TransientError]`, `retry_backoff=True`, `max_retries=3`. Fatal errors (malformed document) should be caught and immediately mark as “FAILURE” without retry.
   * **Checkpoints**

     * Example: In `embed_document_task`, after every 100 chunks embedded, write “last\_chunk\_index” to a JSON column in `document_tasks`. On resume, start embedding at the next index.

3. **Unit Tests**

   * For each task:

     1. Simulate a successful run for a small test document; verify that after completion, `document_tasks.current_stage` is set correctly.
     2. Simulate a pause: manually set `task_status = "PAUSED"` mid‐task; assert that the function exits at the next checkpoint and leaves valid metadata.
     3. Simulate a resume: enqueue the same task again, verify it picks up from the saved offset.
     4. Simulate a forced exception that should trigger retry; verify the retry count increments in Redis.

4. **Success Criteria**

   * Each stage runs as its own Celery task and writes progress to the database.
   * Pause/resume triggers at checkpoints without losing prior results.
   * Cancelling a task cleans up partial data and halts progression.

---

## Phase 1 – Step 4: Upload Service Integration

**Goal**: Refactor upload logic so that document ingestion immediately kicks off the Celery chain, rather than processing synchronously.

1. **Modify Files**

   * `services/document/persistent_upload_service.py`

     * Change `upload_document()` to:

       1. Save the raw document to storage.
       2. Insert a new record in `document_tasks` with `current_stage = "uploaded"`, `task_status = "PENDING"`.
       3. Kick off the Celery chain (e.g., `chain(extract_document_task.s(document_id), chunk_document_task.s(document_id), …).delay()`).
       4. Return a response containing `document_id` and initial `celery_task_ids` (or just the root task ID).
   * `api/routes/document_routes.py`

     * Update the “POST /upload” endpoint:

       * Accept file and metadata.
       * Call `persistent_upload_service.upload_document()`.
       * Return JSON:

         ```json
         {
           "document_id": "...",
           "task_id": "...",
           "status_endpoint": "/tasks/{document_id}/status"
         }
         ```
   * `api/routes/admin_routes.py` (modify)

     * If there are existing admin routes that assume synchronous processing, remove or adapt them:

       * No “upload and wait” endpoint.
       * Instead, provide:

         * `GET /tasks/{document_id}/status` → returns current `task_status`, `current_stage`, and progress metrics.
         * `GET /tasks/{document_id}/logs` (optional) → fetch worker logs or error messages.

2. **Implementation Details**

   * **Non‐Blocking Upload**

     * As soon as the file lands in object storage or on disk, the API layer returns 202 Accepted with a payload containing the `task_id`.
   * **Workflow Orchestration**

     * Use a Celery chain in `persistent_upload_service`. That single chain call returns a group of linked task IDs. Store the root task ID (or a composite list) in `document_tasks.celery_task_id` at insert time.
     * For simplicity, store only the first (root) task ID; subsequent stages inherit their task IDs implicitly.
   * **Progress Tracking**

     * The API’s status endpoint will query `document_tasks` for a given `document_id` and return:

       ```json
       {
         "current_stage": "chunking",
         "task_status": "STARTED",
         "percent_complete": 40,
         "can_pause": true,
         "can_resume": false,
         "can_cancel": true
       }
       ```
     * “percent\_complete” can be calculated by reading checkpoint fields (e.g., number of chunks embedded vs. total).

3. **Unit Tests**

   * **Upload Endpoint**

     1. POST a small test document; assert the response is 202 and contains a valid `task_id`.
     2. Immediately `GET /tasks/{document_id}/status`; assert that `task_status = "PENDING"`.
   * **Status Endpoint**

     1. Simulate state transitions in `document_tasks` (via the state manager); call the endpoint at each stage; assert the JSON fields match expectations.

4. **Success Criteria**

   * Upload endpoint returns immediately with a valid `task_id`.
   * The Celery chain is enqueued (visible via `redis-cli` or `celery -A core.celery_app status`).
   * Status endpoint shows correct `task_status` and `current_stage`.

---

## Phase 1 – Step 5: Pause/Resume/Cancel Implementation

**Goal**: Allow clients to pause, resume, or cancel any in‐progress document pipeline.

1. **Modify/Add Files**

   * `services/celery/task_controller.py` (new)

     * Expose methods:

       * `pause_task(document_id)`
       * `resume_task(document_id)`
       * `cancel_task(document_id)`
     * Each method:

       * Update `document_tasks.task_status` to “PAUSED”/“RESUMED”/“CANCELLED.”
       * For pause: set a `pause_requested = true` flag in the DB.
       * For resume: clear `pause_requested`, set `task_status = "STARTED"`, then re‐enqueue the appropriate Celery task at its last checkpoint.
       * For cancel: set `task_status = "CANCELLED"`, set `cancel_requested = true`, and clean up partial data.
   * `services/celery/checkpoint_manager.py` (new)

     * Manage checkpoint files/DB fields:

       * Fetch or store “last completed offset” for each stage.
       * Provide `load_checkpoint(document_id, stage)` and `save_checkpoint(document_id, stage, metadata_dict)`.
   * `services/document/stage_processors.py` (update)

     * Insert calls to `checkpoint_manager.save_checkpoint(...)` at each logical sub‐step.
     * Before starting a sub‐step, call `checkpoint_manager.load_checkpoint(...)` to resume from the correct point.
   * `api/routes/task_management_routes.py` (new)

     * **POST** `/tasks/{document_id}/pause` → calls `task_controller.pause_task(document_id)`; returns 200 OK with updated flags.
     * **POST** `/tasks/{document_id}/resume` → calls `task_controller.resume_task(document_id)`; returns 200 OK.
     * **POST** `/tasks/{document_id}/cancel` → calls `task_controller.cancel_task(document_id)`; returns 200 OK.
     * **GET** `/tasks/{document_id}/status` (already added in Step 4).

2. **Implementation Details**

   * **Pause Flow**

     1. Client calls `/tasks/{document_id}/pause`.
     2. `task_controller.pause_task()` sets `pause_requested = true`, `can_pause = false`, `can_resume = true`, and `task_status = "PAUSED"`.
     3. Within each running Celery task, before beginning the next sub‐operation, check `document_tasks.pause_requested`. If true → call `checkpoint_manager.save_checkpoint(...)`, exit early.
   * **Resume Flow**

     1. Client calls `/tasks/{document_id}/resume`.
     2. `task_controller.resume_task()` clears `pause_requested`, sets `task_status = "STARTED"`, `can_pause = true`, `can_resume = false`.
     3. Determine which stage was last active (via `current_stage` and checkpoint metadata). Enqueue the corresponding Celery task with its saved offset.
   * **Cancel Flow**

     1. Client calls `/tasks/{document_id}/cancel`.
     2. `task_controller.cancel_task()` sets `cancel_requested = true`, `task_status = "CANCELLED"`, `can_pause = false`, `can_resume = false`, `can_cancel = false`.
     3. Each Celery task checks `cancel_requested` at checkpoints. If true:

        * Delete any partially persisted data (e.g., partially inserted chunks, embeddings).
        * Exit immediately and do not enqueue further tasks.
   * **Checkpoint Granularity**

     * At least one checkpoint per major sub‐operation (e.g., after text extraction, after every N chunk embeddings, after building each subtree).
     * The exact “N” can be tuned later; initially, choose something like “once per 50 items” or “after each file write.”

3. **Unit Tests**

   * **Pause Test**

     1. Enqueue a fake “long” task (e.g., simulate with `time.sleep` loops).
     2. While it’s running, call `/tasks/{document_id}/pause`; assert the task exits at a checkpoint, and DB reflects `task_status = "PAUSED"`.
   * **Resume Test**

     1. After pausing, call `/tasks/{document_id}/resume`; assert that the next Celery task picks up from the correct checkpoint.
     2. Let it finish; assert `task_status = "SUCCESS"`.
   * **Cancel Test**

     1. Enqueue a fake “long” task.
     2. Call `/tasks/{document_id}/cancel`; assert partial data is cleaned up and `task_status = "CANCELLED"`.

4. **Success Criteria**

   * Pause/resume calls return immediately with updated flags.
   * Celery tasks respect pause/cancel signals at checkpoints.
   * Resumed tasks pick up exactly where they left off (no re‐processing prior work).
   * Cancelled tasks remove all intermediate rows and do not trigger downstream stages.

---

## Phase 1 – Step 6: Admin Interface & Monitoring (Minimal MVP)

**Goal**: Provide basic endpoints and minimal tooling to inspect running tasks; defer full monitoring/alerts to Phase 2.

1. **Modify/Add Files**

   * `api/routes/admin_routes.py` (update)

     * Add endpoints to:

       1. `GET /admin/tasks` → list all `document_tasks` (with pagination).
       2. `GET /admin/tasks/{document_id}/history` → return audit trail (all status changes).
       3. (Optional) `GET /admin/workers` → return a snapshot of currently connected Celery workers (via `Inspect().active()` calls).
   * `services/celery/monitor.py` (new)

     * Lightweight helper to fetch:

       * Worker status via Celery’s `app.control.inspect()`.
       * Queue lengths: use Redis keys (e.g., `llen “celery”`).
   * `api/models/task_models.py` (new)

     * Define Pydantic models for:

       * `TaskSummary` (id, document\_id, current\_stage, task\_status, created\_at)
       * `TaskDetail` (+ pause/cancel flags, timestamps, percent\_complete)

2. **Implementation Details**

   * **Admin Endpoints**

     * Implement simple SQL queries or ORM calls to read from `document_tasks`.
     * Return JSON arrays of task summaries.
     * For “history,” store status changes in a separate table (`document_task_events`) or simply join on `updated_at` in `document_tasks`–for MVP, capture only the latest record.
   * **Worker Snapshot**

     * In `monitor.py`, define a function `get_worker_list()` that returns `[worker.hostname for worker in inspect().active_queues().keys()]`.
     * Expose via `GET /admin/workers`.

3. **Unit Tests**

   * **Admin Tasks List**

     1. Seed the `document_tasks` table with example rows.
     2. Call `GET /admin/tasks`; assert response matches seeded data.
   * **Worker Snapshot**

     1. Mock `app.control.inspect()` → return a fake dict.
     2. Assert that `GET /admin/workers` returns the expected JSON array.

4. **Success Criteria**

   * `/admin/tasks` returns a paginated list of all tasks.
   * `/admin/tasks/{id}/history` returns at least the current status (future work can expand history).
   * `/admin/workers` returns a list of active worker hostnames.

---
