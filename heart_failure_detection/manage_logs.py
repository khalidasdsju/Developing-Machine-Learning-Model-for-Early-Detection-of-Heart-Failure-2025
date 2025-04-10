import os
import sys
import argparse
from src.logger import get_logger
from src.exception import CustomException
from src.utils.log_utils import archive_old_logs, clean_old_logs, get_log_stats

logger = get_logger(__name__)

def main():
    """
    Main function to manage log files
    """
    try:
        parser = argparse.ArgumentParser(description="Manage log files")
        parser.add_argument("--archive", action="store_true", help="Archive old log files")
        parser.add_argument("--clean", action="store_true", help="Clean old archived log files")
        parser.add_argument("--stats", action="store_true", help="Get log statistics")
        parser.add_argument("--archive-days", type=int, default=30, help="Days threshold for archiving logs (default: 30)")
        parser.add_argument("--clean-days", type=int, default=90, help="Days threshold for cleaning logs (default: 90)")
        
        args = parser.parse_args()
        
        if args.archive:
            print(f"Archiving logs older than {args.archive_days} days...")
            archived_count = archive_old_logs(days_threshold=args.archive_days)
            print(f"Archived {archived_count} log files")
        
        if args.clean:
            print(f"Cleaning archived logs older than {args.clean_days} days...")
            deleted_count = clean_old_logs(days_threshold=args.clean_days)
            print(f"Deleted {deleted_count} log files")
        
        if args.stats:
            print("Getting log statistics...")
            stats = get_log_stats()
            print("\nLog Statistics:")
            print(f"Total files: {stats['total_files']}")
            print(f"Total size: {stats['total_size_mb']} MB")
            print(f"Oldest log: {stats['oldest_log']}")
            print(f"Newest log: {stats['newest_log']}")
        
        # If no arguments provided, show help
        if not (args.archive or args.clean or args.stats):
            parser.print_help()
        
        return True
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
