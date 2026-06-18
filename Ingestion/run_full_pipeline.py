"""
Full end-to-end compliance document ingestion pipeline.
Orchestrates ingestion, translation, vectorization, and index synchronization.
"""

import argparse
from typing import List, Optional
from datetime import datetime

from ingest_documents import ingest_all_sources
from translate_enforcement import translate_enforcement_documents
from vectorize_documents import vectorize_all_sources
from sync_vector_indices import sync_all_indices, check_all_indices_status
from utils.spark_helpers import get_or_create_spark


def run_full_pipeline(
    sources: Optional[List[str]] = None,
    skip_translation: bool = False,
    skip_sync: bool = False,
    check_status_after: bool = True
) -> dict:
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
    
    print("\n" + "=" * 80)
    print("🚀 STARTING FULL COMPLIANCE DOCUMENT PIPELINE")
    print("=" * 80)
    print(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📦 Processing sources: {', '.join(sources)}")
    print("=" * 80 + "\n")
    
    # ============================================================================
    # STAGE 1: INGEST DOCUMENTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("📥 STAGE 1: DOCUMENT INGESTION")
    print("=" * 80 + "\n")
    
    try:
        ingestion_results = ingest_all_sources(sources=sources)
        results["stages"]["ingestion"] = {
            "status": "success",
            "tables": ingestion_results
        }
        print(f"✅ Ingestion completed: {len(ingestion_results)} sources processed\n")
    except Exception as e:
        results["stages"]["ingestion"] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"❌ Ingestion failed: {e}\n")
        return results
    
    # ============================================================================
    # STAGE 2: TRANSLATE ENFORCEMENT DOCUMENTS (if enforcement is included)
    # ============================================================================
    if "enforcement" in sources and not skip_translation:
        print("\n" + "=" * 80)
        print("🌐 STAGE 2: ENFORCEMENT DOCUMENT TRANSLATION")
        print("=" * 80 + "\n")
        
        try:
            translation_result = translate_enforcement_documents(spark=spark)
            results["stages"]["translation"] = {
                "status": "success",
                "table": translation_result
            }
            print(f"✅ Translation completed\n")
        except Exception as e:
            results["stages"]["translation"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"❌ Translation failed: {e}\n")
            print("⚠️  Pipeline will continue with raw enforcement documents\n")
    else:
        if "enforcement" in sources and skip_translation:
            print("\n⏭️  Skipping translation (--skip-translation flag)\n")
            results["stages"]["translation"] = {"status": "skipped"}
    
    # ============================================================================
    # STAGE 3: VECTORIZE DOCUMENTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("🔢 STAGE 3: DOCUMENT VECTORIZATION")
    print("=" * 80 + "\n")
    
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
        print(f"✅ Vectorization completed: {len(vectorization_results)} sources processed\n")
    except Exception as e:
        results["stages"]["vectorization"] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"❌ Vectorization failed: {e}\n")
        return results
    
    # ============================================================================
    # STAGE 4: SYNC VECTOR INDICES
    # ============================================================================
    if not skip_sync:
        print("\n" + "=" * 80)
        print("🔄 STAGE 4: VECTOR INDEX SYNCHRONIZATION")
        print("=" * 80 + "\n")
        
        try:
            sync_results = sync_all_indices(sources=sources)
            results["stages"]["sync"] = {
                "status": "success",
                "indices": sync_results
            }
            print(f"✅ Index sync completed: {len(sync_results)} indices synced\n")
        except Exception as e:
            results["stages"]["sync"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"❌ Index sync failed: {e}\n")
    else:
        print("\n⏭️  Skipping index sync (--skip-sync flag)\n")
        results["stages"]["sync"] = {"status": "skipped"}
    
    # ============================================================================
    # STAGE 5: CHECK INDEX STATUS (optional)
    # ============================================================================
    if check_status_after and not skip_sync:
        print("\n" + "=" * 80)
        print("📊 STAGE 5: INDEX STATUS CHECK")
        print("=" * 80 + "\n")
        
        try:
            status_results = check_all_indices_status(sources=sources)
            results["stages"]["status_check"] = {
                "status": "success",
                "statuses": status_results
            }
        except Exception as e:
            results["stages"]["status_check"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"⚠️  Status check failed: {e}\n")
    
    # ============================================================================
    # PIPELINE SUMMARY
    # ============================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = duration.total_seconds()
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"⏰ Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total duration: {duration}")
    print("\n📋 Stage Summary:")
    
    for stage, info in results["stages"].items():
        status = info.get("status", "unknown")
        emoji = "✅" if status == "success" else "❌" if status == "failed" else "⏭️"
        print(f"   {emoji} {stage.upper()}: {status}")
    
    print("=" * 80 + "\n")
    
    return results


def run_ingestion_only(sources: Optional[List[str]] = None) -> dict:
    """
    Run only the ingestion stage.
    
    Args:
        sources: List of sources to ingest
        
    Returns:
        Ingestion results
    """
    print("\n🔹 Running ingestion-only pipeline\n")
    return ingest_all_sources(sources=sources)


def run_vectorization_only(sources: Optional[List[str]] = None, use_translated: bool = True) -> dict:
    """
    Run only the vectorization stage.
    
    Args:
        sources: List of sources to vectorize
        use_translated: Use translated enforcement docs
        
    Returns:
        Vectorization results
    """
    print("\n🔹 Running vectorization-only pipeline\n")
    return vectorize_all_sources(sources=sources, use_translated_enforcement=use_translated)


def run_sync_only(sources: Optional[List[str]] = None) -> dict:
    """
    Run only the index sync stage.
    
    Args:
        sources: List of sources to sync
        
    Returns:
        Sync results
    """
    print("\n🔹 Running sync-only pipeline\n")
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