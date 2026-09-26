from src import process_manager
import sys

def main():
    try:
        ans = process_manager.choose_option()
        if ans == 0:
            return 0            
        process_manager.choose_sub_option(ans)
        return 0;
    except KeyboardInterrupt:
        print("\nInterrupted signal by keyboard")
        return 1
    finally:
        print("Cleaning processes")
if __name__ == "__main__":
    sys.exit(main())    