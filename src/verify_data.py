import os

RAW_DATA_PATH = os.path.join('data', 'raw')

def verify_dataset(path):
    if not os.path.exists(path):
        print(f"Error: The path '{path}' does not exist!")
        return

    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"Found {len(classes)} classes (letters/symbols).")
    
    total_images = 0
    for char in sorted(classes):
        char_path = os.path.join(path, char)
        count = len([f for f in os.listdir(char_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"   - {char}: {count} images")
        total_images += count

    print(f"\nTotal Images in Dataset: {total_images}")

if __name__ == "__main__":
    verify_dataset(RAW_DATA_PATH)