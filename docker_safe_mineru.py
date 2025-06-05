import os
import signal
import logging
import tempfile
import multiprocessing
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def safe_signal_context():
    """Context manager to handle signals safely in Docker/Celery environment"""
    original_handlers = {}
    
    try:
        # Store original signal handlers
        for sig in [signal.SIGTERM, signal.SIGINT]:
            try:
                original_handlers[sig] = signal.signal(sig, signal.SIG_DFL)
            except (ValueError, OSError):
                # Signal handling might not be available in some contexts
                pass
        
        yield
        
    finally:
        # Restore original handlers
        for sig, handler in original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

def configure_for_docker():
    """Configure environment for Docker-safe MinerU execution"""
    
    # Disable multiprocessing if requested
    if os.getenv('MINERU_DISABLE_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes'):
        # Force single-threaded execution
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['TORCH_NUM_THREADS'] = '1'
        
        # Disable PyTorch parallelism
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except ImportError:
            pass
    
    # Set multiprocessing start method to 'spawn' for Docker compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass
    
    logger.info("Configured environment for Docker-safe MinerU execution")

def ingest_pdf_docker_safe(pdf_file_path, lang='en', dump_intermediate=False, output_dir=None, storage_adapter=None):
    """
    Docker and Celery-safe version of PDF ingestion.
    Handles signal conflicts and resource constraints.
    """
    
    logger.info(f"Starting Docker-safe PDF ingestion for: {pdf_file_path}")
    
    # Configure environment for Docker
    configure_for_docker()
    
    # Validate input
    if not os.path.exists(pdf_file_path):
        logger.error(f"PDF file not found: {pdf_file_path}")
        return None
    
    try:
        with safe_signal_context():
            # Import MinerU components inside the safe context
            try:
                from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
                from magic_pdf.data.dataset import PymuDocDataset
                from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
                from magic_pdf.config.enums import SupportedPdfParseMethod
                import magic_pdf.model as model_config
                from magic_pdf.config.ocr_content_type import BlockType, ContentType
                from magic_pdf.post_proc.para_split_v3 import ListLineTag
                
                logger.info("Successfully imported MinerU components in safe context")
                
            except ImportError as e:
                logger.error(f"Failed to import MinerU components: {str(e)}")
                return None
            
            # Set model mode to full
            model_config.__model_mode__ = 'full'
            
            content_list_with_bbox = []
            images = {}
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    local_image_dir = os.path.join(temp_dir, "images")
                    os.makedirs(local_image_dir, exist_ok=True)
                    
                    # Initialize data writers
                    image_writer = FileBasedDataWriter(local_image_dir)
                    md_writer = FileBasedDataWriter(temp_dir)
                    
                    name_without_suff = os.path.splitext(os.path.basename(pdf_file_path))[0]
                    
                    # Read PDF content
                    reader = FileBasedDataReader("")
                    pdf_bytes = reader.read(pdf_file_path)
                    
                    logger.info(f"Read PDF bytes: {len(pdf_bytes)} bytes")
                    
                    # Create Dataset Instance with error handling
                    try:
                        ds = PymuDocDataset(pdf_bytes)
                        logger.info("Created PymuDocDataset successfully")
                    except Exception as e:
                        logger.error(f"Failed to create PymuDocDataset: {str(e)}")
                        return None
                    
                    # Process based on PDF classification with timeout protection
                    try:
                        classification = ds.classify()
                        logger.info(f"PDF classification: {classification}")
                        
                        if classification == SupportedPdfParseMethod.OCR:
                            logger.info("Using OCR mode")
                            infer_result = ds.apply(doc_analyze, lang=lang, ocr=True)
                            pipe_result = infer_result.pipe_ocr_mode(image_writer)
                        else:
                            logger.info("Using text mode")
                            infer_result = ds.apply(doc_analyze, ocr=False)
                            pipe_result = infer_result.pipe_txt_mode(image_writer)
                        
                        logger.info("PDF processing completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Error during PDF processing: {str(e)}", exc_info=True)
                        return None
                    
                    # Generate middle.json with error handling
                    middle_json_file = os.path.join(temp_dir, f"{name_without_suff}_middle.json")
                    try:
                        if hasattr(pipe_result, 'dump_middle_json'):
                            pipe_result.dump_middle_json(md_writer, middle_json_file)
                        else:
                            import json
                            pdf_info_to_dump = pipe_result.to_dict().get('pdf_info', []) if hasattr(pipe_result, 'to_dict') else pipe_result.get('pdf_info', [])
                            with open(middle_json_file, 'w', encoding='utf-8') as f:
                                json.dump({'pdf_info': pdf_info_to_dump}, f, indent=4, ensure_ascii=False)
                        
                        logger.info(f"Generated middle.json at: {middle_json_file}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate middle.json: {str(e)}")
                        # Continue without middle.json
                    
                    # Read and process results
                    try:
                        import json
                        with open(middle_json_file, 'r', encoding='utf-8') as f:
                            loaded_middle_data = json.load(f)
                        
                        pdf_info_list = loaded_middle_data.get('pdf_info', [])
                        if not pdf_info_list:
                            logger.warning("No pdf_info found in middle.json")
                            return {
                                "content_list": [],
                                "images": {}
                            }
                        
                        # Import the content extraction function
                        from mineru_ingester import extract_content_with_bbox
                        
                        # Process content with bbox
                        image_dir_prefix = "images"
                        content_list_with_bbox = extract_content_with_bbox(pdf_info_list, image_dir_prefix)
                        
                        logger.info(f"Extracted {len(content_list_with_bbox)} content items")
                        
                    except Exception as e:
                        logger.error(f"Error processing extraction results: {str(e)}")
                        return None
                    
                    # Collect images with error handling
                    try:
                        for root, _, files in os.walk(local_image_dir):
                            for file in files:
                                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                    try:
                                        with open(os.path.join(root, file), 'rb') as img_file:
                                            relative_img_path = os.path.relpath(os.path.join(root, file), local_image_dir)
                                            images[relative_img_path.replace('\\', '/')] = img_file.read()
                                    except IOError as e:
                                        logger.warning(f"Could not read image file {file}: {e}")
                        
                        logger.info(f"Collected {len(images)} images")
                        
                    except Exception as e:
                        logger.warning(f"Error collecting images: {str(e)}")
                        # Continue without images
                    
                    # Save images to output directory if specified
                    if images and output_dir:
                        try:
                            if storage_adapter:
                                storage_adapter.create_directory(output_dir)
                            else:
                                os.makedirs(output_dir, exist_ok=True)
                            
                            for filename, image_data in images.items():
                                filepath = os.path.join(output_dir, filename)
                                try:
                                    if storage_adapter:
                                        storage_adapter.write_file(image_data, filepath)
                                    else:
                                        if not os.path.exists(os.path.dirname(filepath)):
                                            os.makedirs(os.path.dirname(filepath), exist_ok=True)
                                        with open(filepath, 'wb') as f:
                                            f.write(image_data)
                                    logger.debug(f"Saved image to: {filepath}")
                                except Exception as e:
                                    logger.warning(f"Error saving image '{filename}': {e}")
                        except Exception as e:
                            logger.warning(f"Error creating output directory '{output_dir}': {e}")
                    
                    # Return results
                    result = {
                        "content_list": content_list_with_bbox,
                        "images": images
                    }
                    
                    logger.info(f"PDF ingestion completed successfully: {len(content_list_with_bbox)} items, {len(images)} images")
                    return result
                    
            except Exception as e:
                logger.error(f"Error in temporary directory operations: {str(e)}", exc_info=True)
                return None
    
    except Exception as e:
        logger.error(f"Error in safe signal context: {str(e)}", exc_info=True)
        return None