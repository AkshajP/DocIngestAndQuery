import os
from services.document.upload import upload_document
import logging
logging.basicConfig(level=logging.DEBUG)
upload_document('/Users/vikas/Downloads/docllm/output.pdf')