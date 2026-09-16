import os
import subprocess
import platform
from finder import DuplicateMediaFinder

def clean_path(raw_path: str) -> str:
    """Automatically strips outer quotes (single/double), escape backslashes, and trailing spaces from dropped paths."""
    if not raw_path:
        return ""
    path = raw_path.strip().strip("'\"").strip()
    path = os.path.expanduser(path)
    return path

def open_media(media_path: str):
    """Cross-platform function to open an image or video file with the system default player/viewer."""
    try:
        system = platform.system()
        cleaned_path = clean_path(media_path)
        if system == 'Darwin':  # macOS
            subprocess.run(['open', cleaned_path], check=False)
        elif system == 'Windows':  # Windows
            os.startfile(cleaned_path)
        else:  # Linux
            subprocess.run(['xdg-open', cleaned_path], check=False)
    except Exception as e:
        print(f"[Error opening file]: {e}")

def main():
    finder = DuplicateMediaFinder()

    while True:
        print("\n=== DUPLICATE MEDIA (IMAGE & VIDEO) FINDER ===")
        print("1. Scan Directory for Duplicates")
        print("2. Exit")
        
        choice = input("Select an option (1-2): ").strip()

        if choice == "1":
            raw_target = input("Enter path to media folder (drag & drop supported): ")
            target_dir = clean_path(raw_target)

            if not os.path.isdir(target_dir):
                print(f"[Error]: Directory '{target_dir}' not found.")
                continue

            # STEP 1: Analyze folder contents first before asking for threshold
            print(f"\n[Analyzing Folder]: Inspecting contents of '{target_dir}'...")
            profile = finder.analyze_folder_profile(target_dir)

            if profile["total"] < 2:
                print(f"[Info]: Found only {profile['total']} media file(s). Need at least 2 files to compare.")
                continue

            print(f"\n--- Folder Profile Report ---")
            print(f"  • Detected Profile : {profile['profile_type']}")
            print(f"  • Total Files Found: {profile['images']} image(s), {profile['videos']} video(s)")
            print(f"  • AI Recommendation: {profile['advice']}")
            print(f"  • Suggested Value  : {profile['recommended_threshold']}")

            # STEP 2: Prompt user with the smart recommendation pre-loaded
            try:
                threshold_input = input(f"\nEnter similarity threshold [0.70 - 0.99] (default {profile['recommended_threshold']}): ").strip()
                threshold = float(threshold_input) if threshold_input else profile['recommended_threshold']
            except ValueError:
                threshold = profile['recommended_threshold']

            print(f"\n[Scanning]: Processing visual embeddings at a {threshold * 100}% match threshold...")
            groups = finder.scan_directory(target_dir, threshold=threshold)

            if groups:
                print(f"\n[Found]: {len(groups)} set(s) of duplicate media files:")
                for idx, group in enumerate(groups, 1):
                    print(f"\n--- Duplicate Group {idx} ---")
                    for file_idx, path in enumerate(group, 1):
                        print(f"  [{file_idx}] {path}")
                
                # Interactive inspection loop
                inspect = input("\nWould you like to open any of these duplicate groups to inspect them? (y/n): ").strip().lower()
                if inspect == 'y':
                    group_num_input = input(f"Enter group number to open (1 to {len(groups)}), or 'all': ").strip().lower()
                    
                    if group_num_input == 'all':
                        print("[Opening all duplicate media files...]")
                        for group in groups:
                            for path in group:
                                open_media(path)
                    else:
                        try:
                            g_idx = int(group_num_input) - 1
                            if 0 <= g_idx < len(groups):
                                print(f"[Opening media in Group {g_idx + 1}...] ")
                                for path in groups[g_idx]:
                                    open_media(path)
                            else:
                                print("[Error]: Invalid group number.")
                        except ValueError:
                            print("[Error]: Please enter a valid number or 'all'.")
            else:
                print("\n[Result]: No duplicate or near-identical media files found!")

        elif choice == "2":
            print("\nExiting Duplicate Media Finder. Goodbye!")
            break
        else:
            print("[Error]: Invalid option. Please enter 1 or 2.")

if __name__ == "__main__":
    main()