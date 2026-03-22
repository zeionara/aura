from json import dump, load

# import os
# import tempfile
# import signal
# 
# 
# class TempFileManager:
#     """Context manager that creates a temporary file and cleans it up on signals."""
#     def __init__(self, directory):
#         self.directory = directory
#         self.tmp_path = None
#         self.tmp_file = None
#         self.original_handlers = {}
#         self.signals = (signal.SIGINT, signal.SIGTERM)
# 
#     def __enter__(self):
#         # Create the temporary file in the same directory as the target
#         self.tmp_file = tempfile.NamedTemporaryFile(
#             mode='w',
#             dir=self.directory,
#             delete=False,
#             encoding='utf-8'
#         )
#         self.tmp_path = self.tmp_file.name
# 
#         # Set signal handlers to delete the temporary file on interrupt
#         for sig in self.signals:
#             self.original_handlers[sig] = signal.signal(sig, self._handler)
#         return self.tmp_file
# 
#     def _handler(self, signum, _frame):
#         # Delete the temporary file if it exists
#         if self.tmp_path and os.path.exists(self.tmp_path):
#             os.unlink(self.tmp_path)
#         # Restore original handlers and re-raise the signal to exit
#         for sig, handler in self.original_handlers.items():
#             signal.signal(sig, handler)
#         signal.raise_signal(signum)  # Re-raise the signal to terminate normally
# 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # Restore original signal handlers
#         for sig, handler in self.original_handlers.items():
#             signal.signal(sig, handler)
#         # The temporary file is either already removed or will be renamed later
# 
# 
# def dict_to_json_file(data: dict, path: str):
#     """
#     Write a dictionary to a JSON file safely, ensuring no incomplete file is left
#     if the process is killed during writing.
#     """
#     # Ensure we work in the same directory as the target for atomic rename
#     dirname = os.path.dirname(path) or '.'
# 
#     # Use the context manager to handle the temporary file and signals
#     with TempFileManager(dirname) as tmp_file:
#         # Write JSON to the temporary file
#         dump(data, tmp_file, indent=2, ensure_ascii=False)
#         tmp_file.close()  # Ensure all data is flushed
# 
#         # Atomically replace the target file with the temporary one
#         # os.replace works atomically on Unix and Windows
#         os.replace(tmp_file.name, path)


def dict_to_json_file(data: dict, path: str):
    with open(path, 'w', encoding = 'utf-8') as file:
        dump(
            data,
            file,
            indent = 2,
            ensure_ascii = False
        )


def dict_from_json_file(path: str):
    with open(path, 'r', encoding = 'utf-8') as file:
        return load(file)
