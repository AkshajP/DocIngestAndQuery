# cli/migration_commands.py
import click
import logging
from sparse_migration import migrate_collection

logger = logging.getLogger(__name__)

@click.command()
@click.option('--source', default="document_store", help="Source collection name")
@click.option('--target', default="document_store", help="Target collection name")
@click.option('--batch-size', default=100, help="Batch size for processing")
@click.option('--max-documents', default=None, type=int, help="Maximum documents to migrate")
@click.option('--document-id', multiple=True, help="Specific document IDs to migrate")
@click.option('--dry-run', is_flag=True, help="Dry run without actual migration")
def migrate(source, target, batch_size, max_documents, document_id, dry_run):
    """Migrate data from existing collection to a hybrid-enabled collection."""
    if dry_run:
        click.echo(f"DRY RUN: Would migrate from {source} to {target}")
        click.echo(f"  Batch size: {batch_size}")
        click.echo(f"  Max documents: {max_documents or 'ALL'}")
        click.echo(f"  Document IDs: {list(document_id) or 'ALL'}")
        return
    
    click.echo(f"Starting migration from {source} to {target}...")
    
    result = migrate_collection(
        source_collection_name=source,
        target_collection_name=target,
        batch_size=batch_size,
        max_documents=max_documents,
        document_ids=list(document_id) if document_id else None
    )
    
    if result.get("error"):
        click.echo(f"Migration failed: {result['error']}")
    else:
        click.echo(f"Migration completed successfully:")
        click.echo(f"  Documents processed: {result['documents_processed']}")
        click.echo(f"  Chunks migrated: {result['chunks_migrated']}")
        click.echo(f"  Errors: {result['errors']}")
        click.echo(f"  Total time: {result['total_time']:.2f} seconds")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    migrate()