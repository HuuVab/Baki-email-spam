import gdown
import os

# Extract the file ID from your Google Drive URL
file_id = "1BTug8TPRwAgMGxM7NBH0T-G_gzM73PKK"
url = f"https://drive.google.com/uc?id={file_id}"

# Download to current directory with the correct filename
output_path = "best_spam_classifier.pth"
gdown.download(url, output_path, quiet=False)

print(f"File downloaded to: {os.path.abspath(output_path)}")