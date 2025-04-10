import os
import sys
import shutil
from datetime import datetime, timedelta
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def archive_old_logs(log_dir="logs", archive_dir="logs/archive", days_threshold=30):
    """
    Archives log files older than the specified threshold
    
    Args:
        log_dir (str): Directory containing log files
        archive_dir (str): Directory to move archived log files to
        days_threshold (int): Number of days after which logs should be archived
    
    Returns:
        int: Number of files archived
    """
    try:
        log_dir_path = os.path.join(os.getcwd(), log_dir)
        archive_dir_path = os.path.join(os.getcwd(), archive_dir)
        
        # Create archive directory if it doesn't exist
        os.makedirs(archive_dir_path, exist_ok=True)
        
        # Get current date
        current_date = datetime.now()
        threshold_date = current_date - timedelta(days=days_threshold)
        
        # Get all log files
        log_files = [f for f in os.listdir(log_dir_path) if f.endswith(".log") and os.path.isfile(os.path.join(log_dir_path, f))]
        
        archived_count = 0
        for log_file in log_files:
            try:
                # Extract date from log file name (assuming format: YYYY-MM-DD_HH-MM-SS.log)
                file_date_str = log_file.split(".")[0]
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d_%H-%M-%S")
                
                # Check if file is older than threshold
                if file_date < threshold_date:
                    # Move file to archive
                    src_path = os.path.join(log_dir_path, log_file)
                    dst_path = os.path.join(archive_dir_path, log_file)
                    shutil.move(src_path, dst_path)
                    archived_count += 1
                    logger.info(f"Archived log file: {log_file}")
            except Exception as e:
                logger.warning(f"Could not process log file {log_file}: {str(e)}")
                continue
        
        logger.info(f"Archived {archived_count} log files")
        return archived_count
    except Exception as e:
        logger.error(f"Error archiving log files: {str(e)}")
        raise CustomException(e, sys)

def clean_old_logs(log_dir="logs/archive", days_threshold=90):
    """
    Deletes log files older than the specified threshold
    
    Args:
        log_dir (str): Directory containing log files to clean
        days_threshold (int): Number of days after which logs should be deleted
    
    Returns:
        int: Number of files deleted
    """
    try:
        log_dir_path = os.path.join(os.getcwd(), log_dir)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir_path, exist_ok=True)
        
        # Get current date
        current_date = datetime.now()
        threshold_date = current_date - timedelta(days=days_threshold)
        
        # Get all log files
        log_files = [f for f in os.listdir(log_dir_path) if f.endswith(".log") and os.path.isfile(os.path.join(log_dir_path, f))]
        
        deleted_count = 0
        for log_file in log_files:
            try:
                # Extract date from log file name (assuming format: YYYY-MM-DD_HH-MM-SS.log)
                file_date_str = log_file.split(".")[0]
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d_%H-%M-%S")
                
                # Check if file is older than threshold
                if file_date < threshold_date:
                    # Delete file
                    os.remove(os.path.join(log_dir_path, log_file))
                    deleted_count += 1
                    logger.info(f"Deleted log file: {log_file}")
            except Exception as e:
                logger.warning(f"Could not process log file {log_file}: {str(e)}")
                continue
        
        logger.info(f"Deleted {deleted_count} log files")
        return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning log files: {str(e)}")
        raise CustomException(e, sys)

def get_log_stats(log_dir="logs"):
    """
    Gets statistics about log files
    
    Args:
        log_dir (str): Directory containing log files
    
    Returns:
        dict: Statistics about log files
    """
    try:
        log_dir_path = os.path.join(os.getcwd(), log_dir)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir_path, exist_ok=True)
        
        # Get all log files
        log_files = [f for f in os.listdir(log_dir_path) if f.endswith(".log") and os.path.isfile(os.path.join(log_dir_path, f))]
        
        total_size = 0
        oldest_date = None
        newest_date = None
        
        for log_file in log_files:
            try:
                # Get file size
                file_path = os.path.join(log_dir_path, log_file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                # Extract date from log file name (assuming format: YYYY-MM-DD_HH-MM-SS.log)
                file_date_str = log_file.split(".")[0]
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d_%H-%M-%S")
                
                # Update oldest and newest dates
                if oldest_date is None or file_date < oldest_date:
                    oldest_date = file_date
                
                if newest_date is None or file_date > newest_date:
                    newest_date = file_date
            except Exception as e:
                logger.warning(f"Could not process log file {log_file}: {str(e)}")
                continue
        
        stats = {
            "total_files": len(log_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_log": oldest_date.strftime("%Y-%m-%d %H:%M:%S") if oldest_date else None,
            "newest_log": newest_date.strftime("%Y-%m-%d %H:%M:%S") if newest_date else None,
        }
        
        logger.info(f"Log statistics: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error getting log statistics: {str(e)}")
        raise CustomException(e, sys)
