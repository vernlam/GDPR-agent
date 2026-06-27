"""
Full end-to-end compliance document ingestion pipeline.
Orchestrates ingestion, translation, vectorization, and index synchronization.
"""

import argparse
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from ingest_documents import ingest_all_sources
from translate_enforcement import translate_enforcement_documents
from vectorize_documents import vectorize_all_sources
from sync_vector_indices import sync_all_indices, check_all_indices_status
from utils.spark_helpers import get_or_create_spark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    sources: Optional[List[str]] = None,
    skip_translation: bool = False,
    skip_sync: bool = False,
    check_status_after: bool = True
) -> Dict[str, Any]:
    """
    Run the complete compliance document pipeline.
    
    Pipeline stages:
    1. Ingest documents (GDPR, Policy, Enforcement PDFs)
    2. Translate enforcement documents (optional)
    3. Vectorize all documents
    4. Sync vector search indices
    5. Check index status (optional)
    
    Args:
        sources: List of sources to process (gdpr, policy, enforcement)
                 If None, processes all sources
        skip_translation: Skip enforcement translation step
        skip_sync: Skip vector index sync step
        check_status_after: Check index status after sync
        
    Returns:
        Dictionary with pipeline execution results
    """
    start_time = datetime.now()
    spark = get_or_create_spark()
    
    # Default to all sources
    if sources is None:
        sources = ["gdpr", "policy", "enforcement"]
    
    results = {
        "start_time": start_time.isoformat(),
        "sources": sources,
        "stages": {}
    }
    
    logger.info("=" * 80)
    logger.info("STARTING FULL COMPLIANCE DOCUMENT PIPELINE")
    logger.info("=" * 80)
    logger.info("Pipeline started at: %s", start_time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("Processing sources: %s", ', '.join(sources))
    logger.info("=" * 80)
    
    # ============================================================================
    # STAGE 1: INGEST DOCUMENTS
    # ============================================================================
    logger.info("=" * 80)
    logger.info("STAGE 1: DOCUMENT INGESTION")
    logger.info("=" * 80)
    
    try:
        ingestion_results = ingest_all_sources(sources=sources)
        results["stages"]["ingestion"] = {
            "status": "success",
            "tables": ingestion_results
        }
        logger.info("Ingestion stage completed successfully: %d sources processed", len(ingestion_results))
    except Exception as e:
        results["stages"]["ingestion"] = {
            "status": "failed",
            "error": str(e)
        }
        logger.exception("Ingestion stage failed: %s", e)
        return results
    
    # ============================================================================
    # STAGE 2: TRANSLATE ENFORCEMENT DOCUMENTS (if enforcement is included)
    # ============================================================================
    if "enforcement" in sources and not skip_translation:
        logger.info("=" * 80)
        logger.info("STAGE 2: ENFORCEMENT DOCUMENT TRANSLATION")
        logger.info("=" * 80)
        
        try:
            translation_result = translate_enforcement_documents(spark=spark)
            results["stages"]["translation"] = {
                "status": "success",
                "table": translation_result
            }
            logger.info("Translation stage completed successfully")
        except Exception as e:
            results["stages"]["translation"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.exception("Translation stage failed: %s", e)
            logger.warning("Pipeline will continue with raw enforcement documents")
    else:
        if "enforcement" in sources and skip_translation:
            logger.info("Skipping translation stage (skip_translation flag enabled)")
            results["stages"]["translation"] = {"status": "skipped"}
    
    # ============================================================================
    # STAGE 3: VECTORIZE DOCUMENTS
    # ============================================================================
    logger.info("=" * 80)
    logger.info("STAGE 3: DOCUMENT VECTORIZATION")
    logger.info("=" * 80)
    
    try:
        # Use translated enforcement docs if translation succeeded
        use_translated = (
            "enforcement" in sources 
            and not skip_translation 
            and results["stages"].get("translation", {}).get("status") == "success"
        )
        
        vectorization_results = vectorize_all_sources(
            sources=sources,
            use_translated_enforcement=use_translated
        )
        results["stages"]["vectorization"] = {
            "status": "success",
            "tables": vectorization_results
        }
        logger.info("Vectorization stage completed successfully: %d sources processed", len(vectorization_results))
    except Exception as e:
        results["stages"]["vectorization"] = {
            "status": "failed",
            "error": str(e)
        }
        logger.exception("Vectorization stage failed: %s", e)
        return results
    
    # ============================================================================
    # STAGE 4: SYNC VECTOR INDICES
    # ============================================================================
    if not skip_sync:
        logger.info("=" * 80)
        logger.info("STAGE 4: VECTOR INDEX SYNCHRONIZATION")
        logger.info("=" * 80)
        
        try:
            sync_results = sync_all_indices(sources=sources)
            results["stages"]["sync"] = {
                "status": "success",
                "indices": sync_results
            }
            logger.info("Vector index synchronization completed successfully: %d indices synced", len(sync_results))
        except Exception as e:
            results["stages"]["sync"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.exception("Vector index synchronization failed: %s", e)
    else:
        logger.info("Skipping index sync stage (skip_sync flag enabled)")
        results["stages"]["sync"] = {"status": "skipped"}
    
    # ============================================================================
    # STAGE 5: CHECK INDEX STATUS (optional)
    # ============================================================================
    if check_status_after and not skip_sync:
        logger.info("=" * 80)
        logger.info("STAGE 5: INDEX STATUS CHECK")
        logger.info("=" * 80)
        
        try:
            status_results = check_all_indices_status(sources=sources)
            results["stages"]["status_check"] = {
                "status": "success",
                "statuses": status_results
            }
            logger.info("Index status check completed successfully")
        except Exception as e:
            results["stages"]["status_check"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.exception("Index status check failed: %s", e)
    
    # ============================================================================
    # PIPELINE SUMMARY
    # ============================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = duration.total_seconds()
    
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 80)
    logger.info("Pipeline finished at: %s", end_time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("Total pipeline duration: %s", duration)
    logger.info("Stage Summary:")
    
    for stage, info in results["stages"].items():
        status = info.get("status", "unknown")
        logger.info("  %s: %s", stage.upper(), status)
    
    logger.info("=" * 80)
    
    return results


def run_ingestion_only(sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run only the ingestion stage.
    
    Args:
        sources: List of sources to ingest
        
    Returns:
        Ingestion results
    """
    logger.info("Executing ingestion-only pipeline")
    return ingest_all_sources(sources=sources)


def run_vectorization_only(sources: Optional[List[str]] = None, use_translated: bool = True) -> Dict[str, Any]:
    """
    Run only the vectorization stage.
    
    Args:
        sources: List of sources to vectorize
        use_translated: Use translated enforcement docs
        
    Returns:
        Vectorization results
    """
    logger.info("Executing vectorization-only pipeline")
    return vectorize_all_sources(sources=sources, use_translated_enforcement=use_translated)


def run_sync_only(sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run only the index sync stage.
    
    Args:
        sources: List of sources to sync
        
    Returns:
        Sync results
    """
    logger.info("Executing sync-only pipeline")
    return sync_all_indices(sources=sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full compliance document ingestion pipeline"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["gdpr", "policy", "enforcement"],
        help="Specific sources to process (default: all)"
    )
    parser.add_argument(
        "--stage",
        choices=["full", "ingest", "vectorize", "sync"],
        default="full",
        help="Pipeline stage to run (default: full pipeline)"
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip enforcement document translation"
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip vector index synchronization"
    )
    parser.add_argument(
        "--no-status-check",
        action="store_true",
        help="Skip final index status check"
    )
    
    args = parser.parse_args()
    
    if args.stage == "full":
        run_full_pipeline(
            sources=args.sources,
            skip_translation=args.skip_translation,
            skip_sync=args.skip_sync,
            check_status_after=not args.no_status_check
        )
    elif args.stage == "ingest":
        run_ingestion_only(sources=args.sources)
    elif args.stage == "vectorize":
        run_vectorization_only(sources=args.sources)
    elif args.stage == "sync":
        run_sync_only(sources=args.sources)
