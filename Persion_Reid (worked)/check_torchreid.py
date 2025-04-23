import torchreid
import inspect

# Print the directory structure
print("Available modules/attributes in torchreid:")
for item in dir(torchreid):
    if not item.startswith("__"):
        print(f"- {item}")

# Print the file location
print(f"\nTorchreid is installed at: {inspect.getfile(torchreid)}")

# Try to find utils
print("\nTrying to find utils:")
try:
    # Try different potential paths
    from torchreid.reid import utils
    print("Found utils at torchreid.reid.utils")
except ImportError:
    print("Not found at torchreid.reid.utils")

try:
    import torchreid.utils
    print("Found utils at torchreid.utils")
except ImportError:
    print("Not found at torchreid.utils")