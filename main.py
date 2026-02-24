from src import process_manager
from src.visualization import maps
import sys

def main():
    try:
        ans = process_manager.choose_option()
        if ans == 0:
            sys.exit(0)
            
        process_manager.choose_sub_option(ans)
    except KeyboardInterrupt:
        print("Interrupted signal by keyboard")
        sys.exit(1)
    finally:
        print("Cleaning processes")
if __name__ == "__main__":
    main()
    sys.exit(1)    